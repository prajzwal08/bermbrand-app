"""Bermbrand Site Signals — roadside fires on Rijkswaterstaat roads, and every hazard indicator
around the ten sites with the most fires. Streamlit version of the claude.ai dashboard.

Data: ./data (snapshot written by scripts/build_app_snapshot.py in the hackathon repo).
Run:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="Bermbrand Site Signals", page_icon="🔥", layout="wide")
st.markdown("""<style>
html, body, [class*="css"], .stMarkdown, .stDataFrame {font-family: "Times New Roman", Times, Georgia, serif;}
h1 {font-size: 1.6rem; font-weight: 600; margin-bottom: 0.1rem;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; padding: 0.4rem 0.8rem;}
[data-testid="stMetricValue"] {font-size: 1.3rem;}
[data-testid="stMetricLabel"] {font-size: 0.82rem; color: #666;}
.caption {font-size: 0.85rem; color: #666;}
</style>""", unsafe_allow_html=True)

FIRE = "#c8372d"
COL = {"knmi": "#1a1a1a", "s2": "#2e7d32", "s1": "#6b4f9a", "s1b": "#a58fce", "modis": "#b07d12", "modis2": "#d9a93c", "era5": "#8aa1b4"}
BAND = "rgba(26,26,26,0.09)"
RCOL = {50: "#1b5e20", 500: "#2a78d6", 1000: "#c28a1e"}
GROUPS = {"all_veg": "All vegetation", "grass": "Grass", "berm": "Berm (verge)", "heath_shrub": "Heath / shrub",
          "forest": "Forest", "arable": "Arable", "wet": "Wet"}
BUF_S2 = [("ndvi", "NDVI"), ("evi", "EVI"), ("savi", "SAVI"), ("msavi", "MSAVI"), ("gndvi", "GNDVI"), ("nirv", "NIRv (NDVI × NIR)"),
          ("ndre", "NDRE (red-edge)"), ("cire", "CIre (red-edge chlorophyll)"), ("ndmi", "NDMI (fuel water)"),
          ("msi", "MSI (moisture stress, ↑ drier)"), ("nmdi", "NMDI (drought)"), ("nbr", "NBR"), ("ndti", "NDTI (dry / dead matter)"),
          ("ndwi", "NDWI (water / wetness)"), ("bsi", "BSI (bare / burnt ground)")]
BUF_S1 = [("rvi", "RVI = 4·VH/(VV+VH)"), ("cr_db", "VH/VV (dB)"), ("vh_db", "VH (dB)"), ("vv_db", "VV (dB)"), ("power_db", "VV + VH total power (dB)")]
FONT = dict(family="Times New Roman, Times, Georgia, serif", size=12, color="#666")
VEG = {"heide": "#8e3b8a", "grasland overig": "#9ccc65", "grasland agrarisch": "#c5e1a5", "berm (green verge)": "#43a047",
       "groenvoorziening": "#66bb6a", "gemengd bos": "#2e7d32", "naaldbos": "#1b5e20", "loofbos": "#558b2f", "struiken": "#827717",
       "houtwal": "#6d4c41", "bouwland": "#e0c068", "rietland": "#a1887f", "moeras": "#4db6ac", "duin": "#e6d3a3", "transitie": "#bdbdbd"}


# ------------------------------------------------------------------ data

@st.cache_data
def load_sites():
    s = pd.read_parquet(DATA / "sites.parquet")
    s["fires"] = s.fire_times.str.split(";").map(lambda v: [pd.Timestamp(t) for t in v])
    s["ids"] = s.fire_ids.str.split(";")
    return s.set_index("site_id")


@st.cache_data
def load_global():
    return (json.loads((DATA / "panels.json").read_text()), json.loads((DATA / "outline.json").read_text()),
            json.loads((DATA / "prefire.json").read_text()))


@st.cache_data
def load_site(site_id):
    d = DATA / "sites" / site_id
    ser = pd.read_parquet(d / "series.parquet")
    clim = pd.read_parquet(d / "clim.parquet")
    climd = {v: g.sort_values("doy")[["p10", "med", "p90"]].to_numpy() for v, g in clim.groupby("variable")}
    buf = pd.read_parquet(d / "buffers.parquet")
    return (ser, climd, buf, json.loads((d / "cumclim.json").read_text()), json.loads((d / "map.json").read_text()),
            json.loads((d / "header.json").read_text()))


@st.cache_data
def load_fire_risk(site_ids):
    """Per past fire: vulnerability at its cell (default weights) x FWI percentile on its day (scripts/export_vulnerability_map.py)."""
    rows = []
    for sid in site_ids:
        f = DATA / "sites" / sid / "fire_risk.json"
        if f.exists():
            rows += [dict(r, site=sid) for r in json.loads(f.read_text())]
    return pd.DataFrame(rows, columns=["site", "id", "start", "minutes", "vulnerability", "fwi", "fwi_pct", "risk"])


def series(ser, var):
    return ser[ser.variable == var].sort_values("date")


def doy(dates):
    return np.minimum(pd.DatetimeIndex(dates).dayofyear.to_numpy(), 365)


@st.cache_data
def buf_clim(site_id, sensor, idx, radius, group):
    """Day-of-year p10/median/p90 of a radius series from all its years (±15 days, >= 10 values)."""
    buf = load_site(site_id)[2]
    x = buf[(buf.sensor == sensor) & (buf["index"] == idx) & (buf.radius == radius) & (buf.group == group)]
    if x.empty:
        return None
    dd, vv = doy(x.date), x.value.to_numpy()
    out = np.full((365, 3), np.nan)
    for t in range(1, 366):
        g = np.abs(dd - t)
        v = vv[np.minimum(g, 365 - g) <= 15]
        if len(v) >= 10:
            out[t - 1] = np.percentile(v, [10, 50, 90])
    return out


# ------------------------------------------------------------------ sidebar

sites = load_sites()
PANELS, OUTLINE, PF = load_global()
ids = list(sites.index)
if "site" not in st.session_state:
    st.session_state.site = ids[0]

st.sidebar.markdown("### BERMBRAND")
st.sidebar.caption("Roadside fires on Rijkswaterstaat roads: the ten sites with the most fires, and every hazard indicator around them.")
c1, c2 = st.sidebar.columns(2)
if c1.button("← Previous", width="stretch"):
    st.session_state.site = ids[(ids.index(st.session_state.site) - 1) % len(ids)]
if c2.button("Next →", width="stretch"):
    st.session_state.site = ids[(ids.index(st.session_state.site) + 1) % len(ids)]
label = lambda i: f"{i} · {sites.loc[i, 'routes'].replace(';', ' · ')} · {sites.loc[i, 'gemeente'].replace(';', ' / ')} ({sites.loc[i, 'n_fires']} fires)"
site_id = st.sidebar.selectbox("Location", ids, key="site", format_func=label)
S = sites.loc[site_id]
ser, CLIM, BUF, CUMCLIM, MAP, HEADER = load_site(site_id)
fires = S.fires

st.sidebar.divider()
mode = st.sidebar.segmented_control("Values", ["Raw", "Anomaly"], default="Raw") or "Raw"
period = st.sidebar.segmented_control("Period", ["Since 2000", "Since 2017", "Last 3 years"], default="Since 2000") or "Since 2000"
fire_opts = ["—"] + [f"{t:%Y-%m-%d %H:%M}" for t in fires]
zoom_fire = st.sidebar.selectbox("Zoom to fire (±90 days)", fire_opts)
st.sidebar.divider()
radius = st.sidebar.segmented_control("Radius around the fires", [50, 500, 1000], default=500,
                                      format_func=lambda r: "1 km" if r == 1000 else f"{r} m") or 500
group = st.sidebar.selectbox("Fuel group", list(GROUPS), format_func=GROUPS.get)
st.sidebar.caption("Radius and fuel group are used by “Fuel & radar by radius”, “Before fires”, the Summary table and the Verge map. "
                   "BGT-labelled vegetation only (road, water, buildings and paved surfaces excluded).")
st.sidebar.divider()
st.sidebar.caption("Weather: KNMI stations. Fuel: Sentinel-2 (pure verge pixels) and MODIS. Radar: Sentinel-1. "
                   "Satellite values are cloud-screened and never interpolated. Details in Method & QC.")

END = ser.date.max()
if zoom_fire != "—":
    t = pd.Timestamp(zoom_fire)
    XR = [t - pd.Timedelta(days=90), t + pd.Timedelta(days=90)]
else:
    XR = {"Since 2000": [pd.Timestamp("2000-01-01"), END], "Since 2017": [pd.Timestamp("2017-01-01"), END],
          "Last 3 years": [END - pd.DateOffset(years=3), END]}[period]
RLAB = "1 km" if radius == 1000 else f"{radius} m"


# ------------------------------------------------------------------ figure helpers

def base_layout(height=190, **kw):
    lay = dict(height=height, margin=dict(l=52, r=10, t=6, b=24), paper_bgcolor="#fff", plot_bgcolor="#fff", font=FONT,
               showlegend=False, hovermode="x unified",
               xaxis=dict(type="date", gridcolor="#f0f0ec", range=XR),
               yaxis=dict(gridcolor="#f0f0ec", zeroline=mode == "Anomaly", zerolinecolor="#999"),
               shapes=[dict(type="line", xref="x", yref="paper", x0=t, x1=t, y0=0, y1=1, line=dict(color=FIRE, width=1.2)) for t in fires])
    lay.update(kw)
    return lay


def band_traces(dates, table, anom=False):
    if table is None:
        return []
    d = doy(dates)
    p10, med, p90 = (table[d - 1, i] for i in range(3))
    if anom:
        p10, p90, med = p10 - med, p90 - med, None
    tr = [go.Scatter(x=dates, y=p90, mode="lines", line=dict(width=0), hoverinfo="skip"),
          go.Scatter(x=dates, y=p10, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=BAND, hoverinfo="skip")]
    if med is not None:
        tr.append(go.Scatter(x=dates, y=med, mode="lines", line=dict(width=1, color="#999", dash="dot"), hoverinfo="skip"))
    return tr


DAYS = pd.date_range("2000-01-01", END, freq="D")


def panel_figure(items, band_var):
    anom = mode == "Anomaly"
    tr = band_traces(DAYS, CLIM.get(band_var), anom) if band_var else []
    for v, lab, kind, role in items:
        x = series(ser, v)
        if x.empty:
            continue
        y = x.value.to_numpy()
        if anom and v in CLIM:
            y = y - CLIM[v][doy(x.date) - 1, 1]
        if kind == "marker":
            out = x.qc.fillna("").str.contains("outlier")
            tr.append(go.Scatter(x=x.date, y=y, mode="markers", name=lab, customdata=np.c_[x.pct.round(0), x.qc.fillna("")],
                                 marker=dict(size=5, color=COL.get(role, "#333"), symbol=np.where(out, "circle-open", "circle")),
                                 hovertemplate=lab + ": %{y:.3f}<br>percentile %{customdata[0]} · %{customdata[1]}<extra></extra>"))
        else:
            full = x.set_index("date").value.reindex(DAYS)
            yy = full.to_numpy() - (CLIM[v][doy(DAYS) - 1, 1] if anom and v in CLIM else 0)
            tr.append(go.Scatter(x=DAYS, y=yy, mode="lines", name=lab, line=dict(width=1.1, color=COL.get(role, "#333")),
                                 hovertemplate=lab + ": %{y:.3~f}<extra></extra>"))
    return go.Figure(tr, base_layout())


def plot(fig, key):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=key)


# ------------------------------------------------------------------ header

st.title(f"{site_id} · {S.routes.replace(';', ' · ')} near {S.gemeente.replace(';', ' / ')}")
st.caption(f"{S.provincie} · {S.lat:.4f}° N, {S.lon:.4f}° E")
m = st.columns(4)
first, last = fires[0], fires[-1]
m[0].metric("Fires", S.n_fires, f"last on {last:%Y-%m-%d}", delta_color="off")
m[1].metric("Years with fires", S.n_years, f"{first:%Y}–{last:%Y}" if first.year != last.year else f"{first:%Y}", delta_color="off")
fp = HEADER.get("Satellite footprint", "")
m[2].metric("Green verge analysed", (fp.split(", ")[-1] if "m²" in fp else "–"), "BGT berm within 50 m", delta_color="off")
stn = (HEADER.get("KNMI stations (nearest)", "").split(";") or ["–"])[0].strip()
m[3].metric("KNMI station", stn.split(" ", 1)[-1] if stn else "–", "for temperature", delta_color="off")

tabs = st.tabs(["Summary", "Verge map", "Vulnerability map", "Fuel & radar by radius", "Before fires (90 d)", "Fire year", "Weather", "Fire weather",
                "Dryness", "Fuel", "Radar", "All sites", "Method & QC"])


# ------------------------------------------------------------------ helpers: fire-day values and colours

def clim_class(table, value, dy, high_bad):
    if table is None or value is None or not np.isfinite(value):
        return ""
    p10, med, p90 = table[dy - 1]
    if not np.isfinite(p10):
        return ""
    lvl = 2 if value > p90 else 1 if value > med else -2 if value < p10 else -1
    lvl = lvl if high_bad else -lvl
    return {2: "background-color:#b2182b;color:white", 1: "background-color:#f4a582", -1: "background-color:#d1e5f0",
            -2: "background-color:#2166ac;color:white"}[lvl]


def last_before(dates, values, t, max_days=16):
    m_ = (dates <= t) & (dates >= t - pd.Timedelta(days=max_days))
    if not m_.any():
        return None, None
    i = np.where(m_)[0][-1]
    return float(values[i]), int((t - dates[i]) / np.timedelta64(1, "D"))


# ------------------------------------------------------------------ Summary

with tabs[0]:
    left, right = st.columns([0.8, 1.3], gap="large")
    with left:
        st.subheader("Where")
        fr = load_fire_risk(tuple(sites.index))
        risk = fr.groupby("site").risk.mean().reindex(sites.index)
        hover = [f"<b>{i}</b> · mean fire-day risk {risk[i]:.2f}<br>" + "<br>".join(
                     f"{r.start[:16].replace('T', ' ')} · risk {r.risk:.2f} (vuln. {r.vulnerability:.2f} × FWI pct {r.fwi_pct})"
                     for r in fr[fr.site == i].itertuples()) if i in risk.dropna().index else i for i in sites.index]
        fig = go.Figure([
            go.Scattermap(lat=[sites.lat[site_id]], lon=[sites.lon[site_id]], mode="markers", hoverinfo="skip", showlegend=False,
                          marker=dict(size=14 + 3 * sites.n_fires[site_id], color="#2a78d6", opacity=0.9)),
            go.Scattermap(lat=sites.lat, lon=sites.lon, mode="markers+text", text=sites.index, textposition="middle right",
                          marker=dict(size=8 + 3 * sites.n_fires, color=risk.fillna(0), colorscale=[[0, "#fcbba1"], [0.5, "#ef3b2c"], [1, "#67000d"]], cmin=0,
                                      cmax=max(float(risk.max()), 0.05) if risk.notna().any() else 1, opacity=0.95,
                                      colorbar=dict(title=dict(text="fire-day<br>risk", font=dict(size=11)), thickness=12, len=0.6, x=0.98)),
                          customdata=sites.index, hovertext=hover, hovertemplate="%{hovertext}<extra></extra>", showlegend=False)])
        fig.update_layout(map=dict(style="open-street-map", center=dict(lat=52.1, lon=5.1), zoom=6.2), height=520,
                          margin=dict(l=0, r=0, t=0, b=0), font=FONT)
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="overview")
        pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) or hasattr(ev, "get") else []
        if pts:
            new = pts[0].get("customdata")
            if new and new != site_id:
                st.session_state.site = new
                st.rerun()
        st.caption("Dot size = number of fires. Colour = mean risk on the fire days: vulnerability at each fire's cell (default "
                   "weights of the Vulnerability map) × that day's FWI percentile for the date. Darker red = the fire happened where and "
                   "when the map rates risk higher. Blue ring = selected site. Click a dot to switch location.")
    with right:
        st.subheader("Conditions on each fire day")
        st.caption("Coloured by where each value sits in the climatology for that time of year, turned so that red always means more fire-prone.")
        cols = [("knmi_tmax_c", "Tmax °C", 1, True), ("knmi_rh_min_pct", "RH min %", 0, False), ("knmi_wind_mean_ms", "Wind m/s", 1, True),
                ("knmi_fwi", "FWI", 1, True), ("knmi_ffmc", "FFMC", 0, True), ("knmi_precip_deficit_since_apr_mm", "Deficit mm", 0, True),
                ("knmi_days_since_rain", "Dry days", 0, True), ("s2_ndmi", "S2 NDMI (pure verge)", 2, False), ("s2_ndvi", "S2 NDVI (pure verge)", 2, False),
                ("s1_ssm_pct", "S1 soil moist. %", 0, False)]
        bcols = [("s2", "ndvi", f"S2 NDVI {RLAB} {group}", False), ("s2", "ndmi", f"S2 NDMI {RLAB} {group}", False),
                 ("s2", "msi", f"S2 MSI {RLAB} {group}", True), ("s1", "rvi", f"S1 RVI {RLAB} {group}", False)]
        rows, styles = [], []
        for t in fires:
            day = t.normalize()
            dy = min(day.dayofyear, 365)
            r, s_ = {"Fire start": f"{t:%Y-%m-%d %H:%M}"}, {"Fire start": ""}
            for v, lab, dg, hb in cols:
                x = series(ser, v)
                if x.empty:
                    continue
                if x.source.iloc[0].startswith(("knmi", "openmeteo", "era5")):
                    hit = x[x.date == day]
                    val, ago = (float(hit.value.iloc[0]), None) if len(hit) else (None, None)
                else:
                    val, ago = last_before(x.date.to_numpy(), x.value.to_numpy(), day.to_datetime64())
                r[lab] = "–" if val is None else f"{val:.{dg}f}" + (f" (−{ago} d)" if ago else "")
                s_[lab] = clim_class(CLIM.get(v), val, dy, hb)
            for sn, ix, lab, hb in bcols:
                x = BUF[(BUF.sensor == sn) & (BUF["index"] == ix) & (BUF.radius == radius) & (BUF.group == group)].sort_values("date")
                if x.empty:
                    continue
                val, ago = last_before(x.date.to_numpy(), x.value.to_numpy(), day.to_datetime64())
                r[lab] = "–" if val is None else f"{val:.2f}" + (f" (−{ago} d)" if ago else "")
                s_[lab] = clim_class(buf_clim(site_id, sn, ix, radius, group), val, dy, hb)
            rows.append(r); styles.append(s_)
        df = pd.DataFrame(rows)
        sty = pd.DataFrame(styles).reindex(columns=df.columns).fillna("")
        st.dataframe(df.style.apply(lambda _: sty, axis=None), hide_index=True, width="stretch")
        st.caption("Dark red: beyond 90th percentile, fire-prone side · light red: fire-prone side of median · light blue: safer side · "
                   "dark blue: beyond 10th percentile, safer side · (−n d): satellite image n days before.")
        st.subheader("Around the site")
        for key in ["BGT vegetation within 100 m", "BGT vegetation within 500 m"]:
            txt = HEADER.get(key, "")
            parts = [(p.rsplit(" ", 1)[0], float(p.rsplit(" ", 1)[1].rstrip("%"))) for p in txt.split(", ") if p.endswith("%")]
            if parts:
                fig = go.Figure([go.Bar(y=[""], x=[v], orientation="h", name=n, marker_color=VEG.get(n, "#bdbdbd"),
                                        hovertemplate=f"{n}: {v:.0f}%<extra></extra>") for n, v in parts])
                fig.update_layout(barmode="stack", height=110, margin=dict(l=0, r=0, t=22, b=0), font=FONT, showlegend=True,
                                  legend=dict(orientation="h", y=-0.3), title=dict(text=key + " (rest: road, water, buildings, bare)", font=dict(size=12)),
                                  xaxis=dict(range=[0, 100], visible=False), plot_bgcolor="#fff", paper_bgcolor="#fff")
                plot(fig, "veg" + key)
        osm = HEADER.get("Distance to nearest (m, OSM)", "")
        if osm:
            st.dataframe(pd.DataFrame([p.rsplit(" ", 1) for p in osm.split(", ")], columns=["Nearest", "Distance (m)"]), hide_index=True)


# ------------------------------------------------------------------ Verge map (SVG with pan/zoom, same as the artifact)

VERGE_MAP_HTML = """<div style="font-family:Times New Roman,serif;display:grid;grid-template-columns:1fr 240px;gap:16px">
<div style="border:1px solid #e3e3de;border-radius:6px;background:#fbfbf9;overflow:hidden;cursor:grab">%(svg)s</div>
<div style="font-size:14px"><b>Layers</b><div id="layers" style="display:flex;flex-direction:column;gap:4px;margin:6px 0 12px"></div>
<b>Key</b><div style="font-size:12px;display:flex;flex-direction:column;gap:2px;margin-top:4px">%(key)s</div></div></div>
<script>
const el=document.getElementById('vm');const home=%(home)s;let vb=home.slice();
const setVB=()=>el.setAttribute('viewBox',vb.join(' '));setVB();
const L=%(layers)s;const box=document.getElementById('layers');
L.forEach(([k,l])=>{const lab=document.createElement('label');lab.innerHTML=`<input type=checkbox checked> ${l}`;
 lab.querySelector('input').onchange=e=>{el.querySelector(`g[data-l="${k}"]`).style.display=e.target.checked?'':'none'};box.appendChild(lab);});
el.addEventListener('wheel',e=>{e.preventDefault();const r=el.getBoundingClientRect(),k=e.deltaY>0?1.2:1/1.2;
 const sc=Math.max(vb[2]/r.width,vb[3]/r.height);const ox=vb[0]+(e.clientX-r.left-(r.width-vb[2]/sc)/2)*sc,oy=vb[1]+(e.clientY-r.top-(r.height-vb[3]/sc)/2)*sc;
 const w=Math.min(Math.max(vb[2]*k,20),%(w)s*1.5),h=w*vb[3]/vb[2];vb=[ox-(ox-vb[0])*w/vb[2],oy-(oy-vb[1])*h/vb[3],w,h];setVB();},{passive:false});
let dr=null;el.addEventListener('pointerdown',e=>{dr={x:e.clientX,y:e.clientY,vb:vb.slice()};el.setPointerCapture(e.pointerId)});
el.addEventListener('pointermove',e=>{if(!dr)return;const r=el.getBoundingClientRect(),sc=Math.max(vb[2]/r.width,vb[3]/r.height);
 vb=[dr.vb[0]-(e.clientX-dr.x)*sc,dr.vb[1]-(e.clientY-dr.y)*sc,vb[2],vb[3]];setVB();});
el.addEventListener('pointerup',()=>dr=null);el.addEventListener('dblclick',()=>{vb=home.slice();setVB()});
</script>"""


def verge_map_html(m):
    def fill(layer, lab):
        if layer == "veg":
            return VEG.get(lab, "#cfe3b8")
        if layer == "water":
            return "#a9d0f0"
        if layer == "roads":
            return "#c9c9c4" if "gesloten" in lab else "#e3dfd2"
        return "#43a047" if lab.startswith("berm (green") else "#e8e4d8"

    def polys(layer):
        return "".join(f'<path d="{d}" fill="{fill(layer, lab)}" fill-rule="evenodd" stroke="rgba(0,0,0,.18)" stroke-width="0.3" '
                       f'vector-effect="non-scaling-stroke"><title>{lab}</title></path>' for lab, d in m.get(layer, []))
    circles = "".join(f'<path d="{d}" fill="none" stroke="{RCOL[int(r)]}" stroke-width="{3 if int(r) == radius else 1.4}" '
                      f'stroke-dasharray="{"" if int(r) == radius else "6 4"}" vector-effect="non-scaling-stroke"><title>{r} m</title></path>'
                      for r, d in m.get("radii", {}).items())
    firesvg = "".join(f'<circle cx="{x}" cy="{y}" r="4" fill="{FIRE}" stroke="#fff" stroke-width="1.2" vector-effect="non-scaling-stroke"/>'
                      for x, y in m["fires"])
    svg = (f'<svg id="vm" viewBox="0 0 {m["w"]} {m["h"]}" style="width:100%;height:600px;display:block">'
           f'<g data-l="veg">{polys("veg")}</g><g data-l="water">{polys("water")}</g><g data-l="roads">{polys("roads")}</g>'
           f'<g data-l="support">{polys("support")}</g><g data-l="radii">{circles}</g>'
           f'<g data-l="zone"><path d="{m["zone"]}" fill="none" stroke="#555" stroke-dasharray="4 3" stroke-width="1.2" vector-effect="non-scaling-stroke"/></g>'
           f'<g data-l="footprint"><path d="{m["footprint"]}" fill="rgba(27,94,32,.25)" fill-rule="evenodd" stroke="#1b5e20" stroke-width="1.6" vector-effect="non-scaling-stroke"/></g>'
           f'<g data-l="pure"><path d="{m["pure"]}" fill="rgba(42,120,214,.18)" stroke="#2a78d6" stroke-width="1" vector-effect="non-scaling-stroke"/></g>'
           f'<g data-l="fires">{firesvg}</g></svg>')
    fx, fy, pad = [f[0] for f in m["fires"]], [f[1] for f in m["fires"]], radius + 60
    x0, y0 = max(0, min(fx) - pad), max(0, min(fy) - pad)
    home = [x0, y0, min(m["w"], max(fx) + pad) - x0, min(m["h"], max(fy) + pad) - y0]
    layers = [["veg", "BGT vegetation"], ["water", "Water"], ["roads", "Roads and paved"], ["support", "Road-side parts (green = berm)"],
              ["radii", "Radius circles (50 m, 500 m, 1 km)"], ["zone", "50 m around the fire points"],
              ["footprint", "Verge analysed"], ["pure", "Pure Sentinel-2 pixels (≥ 70 % verge)"], ["fires", "Fire points"]]
    classes = sorted({lab for lab, _ in m.get("veg", [])})
    key = "".join(f'<span><i style="display:inline-block;width:12px;height:10px;background:{c};margin-right:5px"></i>{lab}</span>'
                  for lab, c in [("berm (green verge)", "#43a047"), ("verge analysed", "rgba(27,94,32,.45)"), ("pure S2 pixel", "rgba(42,120,214,.35)"),
                                 ("road (sealed)", "#c9c9c4"), ("paved (open)", "#e3dfd2"), ("water", "#a9d0f0")] + [(c, VEG.get(c, "#cfe3b8")) for c in classes])
    return VERGE_MAP_HTML % dict(svg=svg, key=key, home=json.dumps(home), layers=json.dumps(layers), w=m["w"])


with tabs[1]:
    st.subheader(f"Verge boundary at {site_id}")
    st.caption("Source: BGT (Kadaster, via PDOK), today's map, up to 1.1 km around the fires. Opens on the sidebar radius (solid circle). "
               "Scroll to zoom, drag to pan, double-click to reset; hover a shape for its BGT class.")
    # Raw HTML string with its own script (pan/zoom, layer toggles), rendered in an iframe.
    st.iframe("<!doctype html><html><body style='margin:0'>" + verge_map_html(MAP) + "</body></html>", height=640)


# ------------------------------------------------------------------ Vulnerability map

@st.cache_data
def load_nearest():
    rows = []
    for sid in sites.index:
        f = DATA / "sites" / sid / "nearest.json"
        if not f.exists():
            continue
        n = json.loads(f.read_text())
        rt, fs = n.get("station_route") or {}, n.get("fire_station") or {}
        km = lambda o: round(o["m"] / 1000, 2) if o else None  # noqa: E731
        rows.append({"site": sid, "quickest station": rt.get("name"), "drive min": round(rt["minutes"], 1) if rt else None,
                     "road km": round(rt["km"], 1) if rt else None, "closest station (line)": fs.get("name"),
                     "line km": km(fs), "campsite km": km(n.get("campsite")), "parking km": km(n.get("parking")),
                     "rest area km": km(n.get("rest_area"))})
    return pd.DataFrame(rows)


with tabs[2]:
    vpage = DATA / "sites" / site_id / "vulnerability.html"
    st.subheader(f"Where could a fire start and spread at {site_id}?")
    st.caption("Vulnerability = fuel (land-cover % × flammability) × soil drying × ignition (road, traffic, people), lowered by "
               "suppression (water, fire station). Every weight is a slider: set what you think is more or less vulnerable and the map "
               "recolours. Pick a day and hour below the map to see risk under that day's fire weather. Black dots = past fires (size = duration); "
               "the red F and red dashed line = the fire station with the quickest road route.")
    if vpage.exists():
        st.iframe(vpage.read_text(), height=900)
    else:
        st.info("No vulnerability map for this site yet (scripts/export_vulnerability_map.py).")
    near = load_nearest()
    if len(near):
        st.markdown("**Nearest facilities, all sites** <span class='caption'>from the closest past fire; drive time at normal car speeds "
                    "(OSRM), indicative for a fire engine; OSM coverage of stations and campsites is incomplete</span>", unsafe_allow_html=True)
        st.dataframe(near, hide_index=True, width="stretch")


# ------------------------------------------------------------------ Fuel & radar by radius

def buf_series(sensor, idx, r, g):
    return BUF[(BUF.sensor == sensor) & (BUF["index"] == idx) & (BUF.radius == r) & (BUF.group == g)].sort_values("date")


def buf_traces(sensor, idx, r, g, colour, with_band):
    x = buf_series(sensor, idx, r, g)
    if x.empty:
        return []
    tr = []
    if with_band:
        tab = buf_clim(site_id, sensor, idx, r, g)
        days = pd.date_range(x.date.min(), x.date.max(), freq="D")
        tr += band_traces(days, tab)
    for o, xo in (x.groupby("orbit") if sensor == "s1" else [(None, x)]):
        xo = xo.sort_values("date")
        gap = xo.date.diff().dt.days.gt(45)
        dates = xo.date.where(~gap)  # break lines across gaps
        xs = pd.concat([xo.date, xo.date[gap] - pd.Timedelta(days=1)]).sort_values()
        ys = xo.set_index("date").value.reindex(xs)
        ys[xs.isin(xo.date[gap] - pd.Timedelta(days=1)).to_numpy()] = np.nan
        tr.append(go.Scatter(x=xs, y=ys.to_numpy(), mode="lines+markers", connectgaps=False, line=dict(width=1, color=colour),
                             marker=dict(size=3.5, color=colour),
                             hovertemplate=f"{r} m · {GROUPS[g]}{'' if o is None else f' · orbit {o}'}<br>%{{x|%Y-%m-%d}}: %{{y:.3f}}<extra></extra>"))
    return tr


with tabs[3]:
    compare = st.toggle("Compare radii (50 m, 500 m, 1 km)", value=False)
    st.caption(f"Averages over BGT-labelled {GROUPS[group].lower()} pixels (≥ 70 % one class; road, water, buildings, paved excluded). "
               "Sentinel-2: mean reflectance, then the index. Sentinel-1: mean of linear VV and VH, then dB and ratios; one line per orbit. "
               "Change radius and group in the sidebar." + ("" if compare else " Grey band: climatology p10–p90 of this series, dotted median."))
    xr17 = [max(XR[0], pd.Timestamp("2017-01-01")), XR[1]]
    for title, sensor, lst in [("Sentinel-2 (optical, cloud-screened)", "s2", BUF_S2), ("Sentinel-1 (radar)", "s1", BUF_S1)]:
        st.markdown(f"#### {title}")
        for idx, lab in lst:
            tr = ([t for r in (50, 500, 1000) for t in buf_traces(sensor, idx, r, group, RCOL[r], False)] if compare
                  else buf_traces(sensor, idx, radius, group, RCOL[radius], True))
            if not tr:
                continue
            extra = ""
            if sensor == "s1" and idx == "rvi" and not compare:
                x = buf_series("s1", "rvi", radius, group).set_index("date").value
                roll = x.rolling("30D").agg(["mean", "count"])
                tr.append(go.Scatter(x=roll.index, y=roll["mean"].where(roll["count"] >= 5), mode="lines", line=dict(width=2.6, color=FIRE),
                                     hovertemplate="30-day trailing mean<br>%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>"))
                extra = " · thick red line: 30-day trailing rolling mean (all orbits, ≥ 5 images)"
            st.markdown(f"**{lab}** <span class='caption'>{'all radii' if compare else RLAB} · {GROUPS[group].lower()}{extra}</span>", unsafe_allow_html=True)
            fig = go.Figure(tr, base_layout(hovermode="closest"))
            fig.update_xaxes(range=xr17)
            plot(fig, f"buf-{sensor}-{idx}")


# ------------------------------------------------------------------ Before fires (90 d)

with tabs[4]:
    set_ = PF["sets"].get(f"{radius}|{group}", {})
    allr = {**PF["weather"], **set_}
    names = {**PF["labels"], **{f"s2|{k}": "S2 " + l for k, l in BUF_S2}, **{f"s1|{k}": "S1 " + l for k, l in BUF_S1}}
    rows = []
    for k, v in allr.items():
        sm = v.get("summary") or {}
        if not sm:
            continue
        rows.append({"key": k, "Index": names.get(k, k) + ("" if k.startswith("knmi") else f" · {RLAB}"),
                     "Last 30 days vs normal": "▲ higher" if sm["direction"] == "up" else "▼ lower",
                     "Median anomaly": sm["median_z"], "Fires agreeing": f"{sm['fires_agree']}/{sm['n_fires']} ({sm['fires_pct']}%)",
                     "Sites agreeing": f"{sm['sites_agree']}/{sm['n_sites']}", "p (fires)": sm["p_fires"], "p (sites)": sm["p_sites"],
                     "Consistent": sm["sites_agree"] >= 0.8 * sm["n_sites"] and sm["p_fires"] < 0.05, "_s": sm["sites_agree"], "_f": sm["fires_pct"]})
    tab = pd.DataFrame(rows).sort_values(["_s", "_f"], ascending=False).reset_index(drop=True)
    st.subheader(f"What happens in the 90 days before a fire, across all {len({f[0] for f in PF['fires']})} sites")
    st.caption(f"Each fire day ({len(PF['fires'])} in total) is lined up at day 0. Every observation in the 90 days before is compared with the same site's "
               "other years (the fire year left out) within ±15 days of the same date, as a standardised anomaly (0 = normal, ±1 ≈ one standard "
               f"deviation; radar per orbit). Satellite indices use the sidebar radius ({RLAB}) and fuel group ({GROUPS[group].lower()}). "
               "Consistent: ≥ 80 % of sites move the same way in the last 30 days and p < 0.05 over fires.")
    show = tab.drop(columns=["key", "_s", "_f"])
    st.dataframe(show.style.apply(lambda r: ["font-weight:bold;color:#1b5e20" if r["Consistent"] else "" for _ in r], axis=1),
                 hide_index=True, width="stretch", height=min(40 + 35 * len(show), 700))
    default = next((i for i, k in enumerate(tab.key) if not k.startswith("knmi")), 0)
    pick = st.selectbox("Plot index", list(tab.key), index=default, format_func=lambda k: names.get(k, k))
    sel = allr[pick]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if "rolling" in sel:
            R = sel["rolling"]
            dx = list(range(-90, 1))
            tr = [go.Scatter(x=dx, y=c, mode="lines", line=dict(width=1, color="rgba(26,26,26,.2)"), hovertemplate=f"{lab}<br>%{{x}} d: %{{y:.2f}}<extra></extra>")
                  for c, lab in zip(R["curves"], R["labels"])]
            b = R["band"]
            tr += [go.Scatter(x=dx, y=b["hi"], mode="lines", line=dict(width=0), hoverinfo="skip"),
                   go.Scatter(x=dx, y=b["lo"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(42,120,214,.22)", hoverinfo="skip"),
                   go.Scatter(x=dx, y=b["mean"], mode="lines", line=dict(width=3, color="#2a78d6"), hovertemplate="mean %{x} d: %{y:.2f}<extra></extra>")]
            st.markdown(f"**{names.get(pick, pick)}** <span class='caption'>30-day trailing rolling mean of the anomalies (only images before each day; each orbit "
                        f"against its own normal; ≥ {R['min_obs']} images per window). Thin: individual fires; thick: mean; band: 90 % bootstrap interval.</span>",
                        unsafe_allow_html=True)
            yr = [-2, 2]
        else:
            xs = [b + 5 for b in PF["bins"]]
            arr = np.array([[np.nan if v is None else v for v in f["bins"]] for f in sel["per_fire"]])
            tr = [go.Scatter(x=xs, y=f["bins"], mode="lines", line=dict(width=1, color="rgba(26,26,26,.22)"), connectgaps=True,
                             hovertemplate=f"{f['site']} {f['fire']}<br>%{{x}} d: %{{y:.2f}}<extra></extra>") for f in sel["per_fire"]]
            with np.errstate(all="ignore"):
                q25, q50, q75 = (np.nanpercentile(arr, p, axis=0) for p in (25, 50, 75))
            tr += [go.Scatter(x=xs, y=q75, mode="lines", line=dict(width=0), hoverinfo="skip"),
                   go.Scatter(x=xs, y=q25, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(42,120,214,.18)", hoverinfo="skip"),
                   go.Scatter(x=xs, y=q50, mode="lines+markers", line=dict(width=3, color="#2a78d6"), hovertemplate="median %{x} d: %{y:.2f}<extra></extra>")]
            st.markdown(f"**{names.get(pick, pick)}** <span class='caption'>10-day means per fire (thin); median across fires (thick); band 25th–75th percentile.</span>",
                        unsafe_allow_html=True)
            yr = [-3, 3]
        fig = go.Figure(tr)
        fig.update_layout(height=380, margin=dict(l=56, r=10, t=8, b=36), paper_bgcolor="#fff", plot_bgcolor="#fff", font=FONT, showlegend=False,
                          xaxis=dict(title="days before the fire", range=[-90, 0], gridcolor="#f0f0ec"),
                          yaxis=dict(title="anomaly (z)", range=yr, gridcolor="#f0f0ec", zeroline=True, zerolinecolor="#999"),
                          shapes=[dict(type="line", x0=0, x1=0, y0=yr[0], y1=yr[1], line=dict(color=FIRE, width=2))])
        plot(fig, "pf-line")
    with c2:
        st.markdown("**All indices, 10-day bins before the fire** <span class='caption'>median anomaly across fires; red above normal, blue below.</span>",
                    unsafe_allow_html=True)
        xs = [b + 5 for b in PF["bins"]]
        z = []
        for k in tab.key:
            arr = np.array([[np.nan if v is None else v for v in f["bins"]] for f in allr[k]["per_fire"]])
            with np.errstate(all="ignore"):
                z.append(np.nanmedian(arr, axis=0))
        fig = go.Figure(go.Heatmap(x=xs, y=[names.get(k, k) for k in tab.key], z=z, zmid=0, zmin=-1.5, zmax=1.5,
                                   colorscale=[[0, "#2166ac"], [0.25, "#92c5de"], [0.5, "#f7f7f7"], [0.75, "#f4a582"], [1, "#b2182b"]],
                                   hovertemplate="%{y}<br>%{x} d: %{z:.2f}<extra></extra>", colorbar=dict(title="z", thickness=12)))
        fig.update_layout(height=max(380, 17 * len(tab) + 60), margin=dict(l=230, r=10, t=8, b=36), font=FONT, paper_bgcolor="#fff",
                          xaxis=dict(title="days before the fire (bin centre)"), yaxis=dict(autorange="reversed", tickfont=dict(size=11)))
        plot(fig, "pf-heat")


# ------------------------------------------------------------------ Fire year

FY = [("Max temperature (°C)", "knmi_tmax_c", "line", 1, True), ("Min relative humidity (%)", "knmi_rh_min_pct", "line", 0, False),
      ("Max VPD (kPa)", "knmi_vpd_max_kpa", "line", 2, True), ("Mean wind (m/s)", "knmi_wind_mean_ms", "line", 1, True),
      ("Fire Weather Index", "knmi_fwi", "line", 1, True), ("FFMC (fine fuel moisture code)", "knmi_ffmc", "line", 0, True),
      ("Drought Code", "knmi_dc", "line", 0, True), ("Days since rain > 1 mm", "knmi_days_since_rain", "line", 0, True),
      ("Rain since 1 January (mm)", "cum_precip_mm", "cum", 0, False), ("Makkink evaporation since 1 January (mm)", "cum_et_mm", "cum", 0, True),
      ("P − ET since 1 January (mm)", "cum_p_minus_et_mm", "cum", 0, False), ("Rain deficit since 1 April (mm)", "knmi_precip_deficit_since_apr_mm", "line", 0, True),
      ("S1 soil moisture index (%)", "s1_ssm_pct", "obs", 0, False), ("S2 NDVI (pure verge)", "s2_ndvi", "obs", 2, False),
      ("S2 NDMI (fuel water)", "s2_ndmi", "obs", 2, False), ("S2 MSI (moisture stress)", "s2_msi", "obs", 2, True),
      ("S2 NDTI (dry / dead matter)", "s2_ndti", "obs", 3, True), ("S1 cross-ratio VH − VV (dB)", "s1_cr_db", "obs", 1, False)]

with tabs[5]:
    fsel = st.segmented_control("Fire", [f"{t:%Y-%m-%d %H:%M}" for t in fires], default=f"{fires[-1]:%Y-%m-%d %H:%M}")
    fday = pd.Timestamp(fsel or f"{fires[-1]:%Y-%m-%d %H:%M}").normalize()
    y0 = pd.Timestamp(year=fday.year, month=1, day=1)
    days = pd.date_range(y0, fday, freq="D")
    st.caption(f"{fday.year}, from 1 January to the fire day (black), against the climatology for the same calendar days (grey band p10–p90, dotted median).")
    wide = ser[ser.source == "knmi_station"].pivot_table(index="date", columns="variable", values="value").reindex(days)
    cols_ = st.columns(2, gap="large")
    k = 0
    for title, v, kind, dg, hb in FY:
        if kind == "cum":
            src = {"cum_precip_mm": wide.get("knmi_precip_mm"), "cum_et_mm": wide.get("knmi_makkink_et_mm"),
                   "cum_p_minus_et_mm": (wide.get("knmi_precip_mm") - wide.get("knmi_makkink_et_mm"))
                   if {"knmi_precip_mm", "knmi_makkink_et_mm"} <= set(wide.columns) else None}[v]
            if src is None or v not in CUMCLIM:
                continue
            cs = src.cumsum()
            cs[src.isna().cummax()] = np.nan
            xd, yd = days, cs.to_numpy()
            c = CUMCLIM[v]
            n = len(days)
            tab_ = np.c_[c["p10"][:n], c["med"][:n], c["p90"][:n]].astype(float)
            band = [go.Scatter(x=days, y=tab_[:, 2], mode="lines", line=dict(width=0), hoverinfo="skip"),
                    go.Scatter(x=days, y=tab_[:, 0], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=BAND, hoverinfo="skip"),
                    go.Scatter(x=days, y=tab_[:, 1], mode="lines", line=dict(width=1, color="#999", dash="dot"), hoverinfo="skip")]
            med_end, p10e, p90e = tab_[-1, 1], tab_[-1, 0], tab_[-1, 2]
        else:
            x = series(ser, v)
            x = x[(x.date >= y0) & (x.date <= fday)]
            if x.empty:
                continue
            xd, yd = x.date, x.value.to_numpy()
            band = band_traces(days, CLIM.get(v))
            t_ = CLIM.get(v)
            med_end, p10e, p90e = (t_[min(fday.dayofyear, 365) - 1, 1], t_[min(fday.dayofyear, 365) - 1, 0], t_[min(fday.dayofyear, 365) - 1, 2]) if t_ is not None else (np.nan,) * 3
        ok = np.isfinite(yd)
        if not ok.any():
            continue
        fdv = yd[ok][-1]
        lvl = "" if not np.isfinite(med_end) else ("↑ above p90" if fdv > p90e else "below p10" if fdv < p10e else "")
        colour = FIRE if ((fdv > med_end) == hb) and np.isfinite(med_end) else "#2166ac"
        tr = band + ([go.Scatter(x=xd, y=yd, mode="markers", marker=dict(size=6, color=COL["s1" if v.startswith("s1") else "s2"]),
                                 hovertemplate="%{x|%d %b}: %{y:.3f}<extra></extra>")] if kind == "obs" else
                     [go.Scatter(x=xd, y=yd, mode="lines", line=dict(width=2, color="#1a1a1a"), hovertemplate="%{x|%d %b}: %{y:.3~f}<extra></extra>")])
        fig = go.Figure(tr)
        fig.update_layout(height=210, margin=dict(l=46, r=8, t=4, b=24), paper_bgcolor="#fff", plot_bgcolor="#fff", font=FONT, showlegend=False,
                          xaxis=dict(range=[y0, fday + pd.Timedelta(days=4)], tickformat="%d %b", gridcolor="#f0f0ec"),
                          yaxis=dict(gridcolor="#f0f0ec", zeroline=False),
                          shapes=[dict(type="line", xref="x", yref="paper", x0=fday, x1=fday, y0=0, y1=1, line=dict(color=FIRE, width=1.5))])
        with cols_[k % 2]:
            st.markdown(f"**{title}** <span class='caption'>on fire day: <span style='color:{colour}'>{fdv:.{dg}f}</span>"
                        f"{'' if not np.isfinite(med_end) else f', median {med_end:.{dg}f}'}</span>", unsafe_allow_html=True)
            plot(fig, f"fy-{v}")
        k += 1


# ------------------------------------------------------------------ time-series tabs (from panels.json)

for tab_i, grp in zip(range(6, 11), ["C spread", "B/C fire weather", "B dryness", "A fuel", "A/B radar"]):
    with tabs[tab_i]:
        st.caption("Red lines are fire starts. Grey band: 10th–90th percentile of the ±15-day climatology (KNMI 1991–2020; satellites all years); "
                   "dotted: median. Satellite points are single cloud-screened images; hollow markers are flagged outliers. Use Values and Period in the sidebar.")
        for g, title, items, band_var in PANELS:
            if g != grp or not any(not series(ser, v).empty for v, *_ in items):
                continue
            present = " · ".join(lab for v, lab, *_ in items if not series(ser, v).empty)
            st.markdown(f"**{title}** <span class='caption'>{present}</span>", unsafe_allow_html=True)
            plot(panel_figure(items, band_var), f"p-{grp}-{title}")


# ------------------------------------------------------------------ All sites

with tabs[11]:
    st.subheader("All ten sites")
    st.caption("Incidents within 1 km are grouped into one site (DBSCAN on the RWS bermbrand points) and ranked by number of fires. Select a row to open it.")
    t = sites.reset_index()[["site_id", "routes", "gemeente", "provincie", "n_fires", "n_years"]].copy()
    t["last fire"] = sites.fires.map(lambda f: f"{f[-1]:%Y-%m-%d}").to_numpy()
    ev = st.dataframe(t, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="allsites")
    rows_sel = ev.selection.rows if hasattr(ev, "selection") else []
    if rows_sel and t.site_id.iloc[rows_sel[0]] != site_id:
        st.session_state.site = t.site_id.iloc[rows_sel[0]]
        st.rerun()


# ------------------------------------------------------------------ Method & QC

with tabs[12]:
    st.subheader(f"Sources for {site_id}")
    st.dataframe(pd.DataFrame([(k, v) for k, v in HEADER.items() if not k.startswith("BGT vegetation") and k != "Site"],
                              columns=["Item", "Value"]), hide_index=True, width="stretch")
    st.markdown("""
**Weather (KNMI).** Validated daily station data. For each variable, the nearest station with ≥ 95 % of days present in the 3 years
before the site's first fire; among those within 25 km, the one with the longest record. Gaps filled from another station within 25 km
and flagged; nothing interpolated. P − ET uses Makkink evaporation; the rain deficit runs 1 April – 30 September and never drops below zero.
The Fire Weather Index (Canadian system) uses KNMI 12:00 UT temperature, humidity and wind plus daily rain, restarting after gaps.

**Satellites, verge.** The footprint is the BGT green verge (berm, groenvoorziening) within 50 m of each fire point. Sentinel-2 uses pure
verge pixels (≥ 70 % verge, weighted by verge share), SCL classes 4/5, cloud and shadow grown by 60 m, haze test, the 2022 baseline offset
corrected; scenes with > 20 % bad pixels are dropped; hollow markers are outliers (> 3 robust SD) kept for review. Sentinel-1: soil-moisture
change index per orbit (5th–95th percentile scaling, frost days removed); RVI and VH/VV per pixel. MODIS: good-quality pixels only (250 m).

**Satellites, radius × fuel group.** Local QC'd image cubes (1.1 km around each site). Pixels labelled with BGT classes (≥ 70 % one
vegetation class, < 10 % road/water/building/paved/bare). Sentinel-2: the buffer must be ≥ 80 % clear; mean reflectance per band, then
the index. Sentinel-1: mean of linear VV and VH (pixels inside a 3 × 3 area of the same group), then dB, RVI and VH/VV.

**Climatology band.** 10th–90th percentile of the ±15-day window around each calendar day: KNMI 1991–2020 (all years if a station has
< 10 years in that period); satellites all their years. The Anomaly view subtracts the median.
""")
