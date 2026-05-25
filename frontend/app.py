import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time
import streamlit.components.v1 as components


st.set_page_config(
    page_title="DisasterSense — India Flood Alert Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://localhost:8000"

DISTRICTS = [
    "Jaipur","Mumbai","Delhi","Kolkata","Chennai",
    "Bangalore","Hyderabad","Ahmedabad","Pune","Patna",
    "Bhopal","Lucknow","Kochi","Guwahati","Bhubaneswar"
]

RISK_COLORS = {"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"}
RISK_EMOJI  = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

    :root {
        --bg: #0d1117;
        --card: #161b22;
        --border: #30363d;
        --green: #22c55e;
        --yellow: #eab308;
        --red: #ef4444;
        --blue: #3b82f6;
        --text: #e6edf3;
        --muted: #8b949e;
    }

    .main { background: var(--bg); }
    .block-container { padding: 1.5rem 2rem; }

    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .metric-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'DM Sans', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
    }
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .alert-card {
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
        font-family: 'DM Sans', sans-serif;
    }
    .alert-danger  { border-color: var(--red);    background: rgba(239,68,68,0.08); }
    .alert-warning { border-color: var(--yellow);  background: rgba(234,179,8,0.08); }
    .alert-info    { border-color: var(--green);   background: rgba(34,197,94,0.08); }
    .resource-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0;
        border-bottom: 1px solid var(--border);
        font-family: 'DM Sans', sans-serif;
    }
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--muted);
        margin: 1.2rem 0 0.6rem;
    }
    .stMetric label { font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; }
    .stSelectbox label { font-family: 'Space Mono', monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_dashboard(district: str):
    try:
        r = requests.get(f"{API_BASE}/dashboard/{district}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}. Make sure FastAPI is running at {API_BASE}")
        return None

@st.cache_data(ttl=300)
def fetch_map_data():
    try:
        r = requests.get(f"{API_BASE}/map-data", timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def leaflet_map_html(data: dict, selected_district: str, relief_centers: list) -> str:
    """Generate Leaflet map HTML."""
    center_lat = data["lat"]
    center_lon = data["lon"]

    markers_js = ""
    for d in (fetch_map_data() or {}).get("districts", []):
        color = d["color"]
        risk  = d["risk_level"]
        is_selected = "★ " if d["district"] == selected_district else ""
        markers_js += f"""
        L.circleMarker([{d["lat"]}, {d["lon"]}], {{
            radius: {"16" if d["district"] == selected_district else "10"},
            fillColor: "{color}",
            color: "{"#fff" if d["district"] == selected_district else color}",
            weight: {"3" if d["district"] == selected_district else "1.5"},
            opacity: 1,
            fillOpacity: 0.8
        }}).addTo(map)
        .bindPopup(`<b>{is_selected}{d["district"]}</b><br>Risk: <b style="color:{color}">{risk}</b><br>~{d["annual_rainfall_mm"]}mm/yr`);
        """

    for rc in relief_centers:
        icon_color = {"Medical":"red","Shelter":"blue","Food":"green"}.get(rc["type"],"gray")
        markers_js += f"""
        L.marker([{rc["lat"]}, {rc["lon"]}], {{
            icon: L.divIcon({{
                html: '<div style="background:{icon_color};color:white;border-radius:50%;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font-size:12px;border:2px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.4)">{"🏥" if rc["type"]=="Medical" else "🏠" if rc["type"]=="Shelter" else "🍱"}</div>',
                iconSize: [24,24], iconAnchor: [12,12]
            }})
        }}).addTo(map)
        .bindPopup(`<b>{rc["name"]}</b><br>Type: {rc["type"]}`);
        """

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    #map {{ width:100%; height:420px; border-radius:12px; }}
    .legend {{
      background: rgba(22,27,34,0.92);
      padding: 10px 14px;
      border-radius: 8px;
      color: #e6edf3;
      font-family: monospace;
      font-size: 12px;
      line-height: 1.8;
      border: 1px solid #30363d;
    }}
    .legend-dot {{
      display:inline-block; width:10px; height:10px;
      border-radius:50%; margin-right:6px; vertical-align:middle;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    var map = L.map('map', {{zoomControl:true}}).setView([{center_lat}, {center_lon}], 6);

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '© OpenStreetMap contributors © CARTO',
      maxZoom: 18
    }}).addTo(map);

    {markers_js}

    // Legend
    var legend = L.control({{position: 'bottomright'}});
    legend.onAdd = function(map) {{
      var div = L.DomUtil.create('div', 'legend');
      div.innerHTML = '<b>Risk Zones</b><br>'
        + '<span class="legend-dot" style="background:#22c55e"></span>Safe<br>'
        + '<span class="legend-dot" style="background:#eab308"></span>Medium<br>'
        + '<span class="legend-dot" style="background:#ef4444"></span>Danger<br>'
        + '<br><b>Relief Centers</b><br>'
        + '🏥 Medical &nbsp; 🏠 Shelter &nbsp; 🍱 Food';
      return div;
    }};
    legend.addTo(map);
  </script>
</body>
</html>
"""

def render_rainfall_chart(forecast: list) -> go.Figure:
    dates  = [f["date"] for f in forecast]
    precip = [f["precipitation_mm"] for f in forecast]
    
    # Color bars by intensity
    bar_colors = []
    for p in precip:
        if p > 15:   bar_colors.append("#ef4444")
        elif p > 8:  bar_colors.append("#eab308")
        else:        bar_colors.append("#3b82f6")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=precip,
        marker_color=bar_colors,
        marker_line_width=0,
        name="Rainfall",
        hovertemplate="%{x}<br>%{y} mm<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=precip,
        mode="lines+markers",
        line=dict(color="#60a5fa", width=2, dash="dot"),
        marker=dict(size=6, color="#60a5fa"),
        name="Trend",
        hovertemplate="%{x}<br>%{y} mm<extra></extra>"
    ))

    fig.update_layout(
        plot_bgcolor="#161b22",
        paper_bgcolor="#161b22",
        font=dict(color="#8b949e", family="Space Mono, monospace", size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        height=220,
        title=dict(text="7-Day Rainfall Forecast (mm)", font=dict(size=12, color="#e6edf3")),
        xaxis=dict(showgrid=False, linecolor="#30363d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d", linecolor="#30363d"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.15),
        bargap=0.3
    )
    return fig

def render_risk_gauge(proba: dict) -> go.Figure:
    risk_score = proba["Low"] * 0 + proba["Medium"] * 50 + proba["High"] * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score, 1),
        domain={"x": [0,1], "y": [0,1]},
        title={"text": "RISK SCORE", "font": {"size": 11, "color": "#8b949e", "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#8b949e", "size": 9}},
            "bar": {"color": "#3b82f6"},
            "steps": [
                {"range": [0, 33],  "color": "rgba(34,197,94,0.2)"},
                {"range": [33, 66], "color": "rgba(234,179,8,0.2)"},
                {"range": [66, 100],"color": "rgba(239,68,68,0.2)"}
            ],
            "threshold": {
                "line": {"color": "#fff", "width": 3},
                "thickness": 0.75,
                "value": risk_score
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        height=200,
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:1.2rem; font-weight:700; 
                color:#e6edf3; padding-bottom:0.5rem; border-bottom:1px solid #30363d; margin-bottom:1rem">
        🌊 DisasterSense
    </div>
    <div style="font-family:'DM Sans',sans-serif; font-size:0.8rem; color:#8b949e; margin-bottom:1.5rem">
        AI-Powered Flood Alert Dashboard<br>India — Real-Time Risk Monitoring
    </div>
    """, unsafe_allow_html=True)

    district = st.selectbox("📍 Select District", DISTRICTS, index=0)

    auto_refresh = st.toggle("Auto Refresh (2 min)", value=False)
    if auto_refresh:
        st.info("Auto-refresh active. Data updates every 2 minutes.")

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#8b949e">
    <b>DATA SOURCES</b><br>
    • Weather: Open-Meteo API (free)<br>
    • ML Model: XGBoost/Random Forest<br>
    • Rainfall: India 1901-2015 dataset<br><br>
    <b>RISK THRESHOLDS</b><br>
    • Low:    &lt; 1000 mm/yr<br>
    • Medium: 1000–1800 mm/yr<br>
    • High:   &gt; 1800 mm/yr
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── Main Dashboard ───────────────────────────────────────────────────────────
st.markdown(f"""
<h1 style="font-family:'Space Mono',monospace; font-size:1.5rem; 
           color:#e6edf3; margin-bottom:0.2rem">
    🌊 Disaster Alert Dashboard
</h1>
<div style="font-family:'DM Sans',sans-serif; color:#8b949e; font-size:0.9rem; margin-bottom:1.5rem">
    District: <b style="color:#3b82f6">{district}</b> &nbsp;|&nbsp; 
    Updated: {datetime.now().strftime("%b %d, %Y %H:%M")}
</div>
""", unsafe_allow_html=True)

data = fetch_dashboard(district)

if data is None:
    st.warning("Could not connect to backend. Please run `uvicorn main:app --reload` in the backend folder.")
    st.stop()

risk    = data["risk"]
weather = data["weather"]
resources = data["resources"]
alerts  = data["alerts"]
relief  = data["relief_centers"]

c1, c2, c3, c4, c5 = st.columns(5)

risk_color = risk["color"]
risk_level = risk["level"]

with c1:
    st.markdown(f"""
    <div class="metric-card" style="border-color:{risk_color}40">
        <div class="metric-label">Risk Level</div>
        <div class="risk-badge" style="background:{risk_color}20; color:{risk_color}; border:1px solid {risk_color}60">
            {RISK_EMOJI[risk_level]} {risk_level.upper()} RISK
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">7-Day Rainfall</div>
        <div class="metric-value">{weather['total_7day_rain_mm']}<span style="font-size:1rem;color:#8b949e"> mm</span></div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Temperature</div>
        <div class="metric-value">{weather['current_temp_c']}<span style="font-size:1rem;color:#8b949e"> °C</span></div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Wind Speed</div>
        <div class="metric-value">{weather['current_wind_kmh']}<span style="font-size:1rem;color:#8b949e"> km/h</span></div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    est = weather['annual_est_mm']
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Annual Est.</div>
        <div class="metric-value">{est:.0f}<span style="font-size:1rem;color:#8b949e"> mm</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Content ─────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown('<div class="section-title">📍 Risk Zone Map</div>', unsafe_allow_html=True)
    map_html = leaflet_map_html(data, district, relief)
    components.html(map_html, height=440)

with right:
    # Risk Gauge
    st.markdown('<div class="section-title">⚡ Risk Score</div>', unsafe_allow_html=True)
    st.plotly_chart(render_risk_gauge(risk["probability"]), use_container_width=True, config={"displayModeBar": False})

    # Probability bars
    for lvl, pct in risk["probability"].items():
        color = RISK_COLORS[lvl]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-family:'DM Sans',sans-serif;font-size:0.85rem">
            <span style="width:55px;color:#8b949e">{lvl}</span>
            <div style="flex:1;background:#21262d;border-radius:999px;height:8px;overflow:hidden">
                <div style="width:{pct*100:.1f}%;height:100%;background:{color};border-radius:999px"></div>
            </div>
            <span style="width:40px;text-align:right;color:{color}">{pct*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Row 2: Rainfall chart + Alerts + Resources ───────────────────────────────
col_chart, col_alerts, col_res = st.columns([2, 1.5, 1.5])

with col_chart:
    st.markdown('<div class="section-title">📈 Rainfall Trend — Last 7 Days</div>', unsafe_allow_html=True)
    fig = render_rainfall_chart(weather["forecast"])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_alerts:
    st.markdown('<div class="section-title">🚨 Active Alerts</div>', unsafe_allow_html=True)
    for alert in alerts:
        css_class = f"alert-{alert['type']}"
        border_color = {"danger": "#ef4444", "warning": "#eab308", "info": "#22c55e"}.get(alert["type"], "#3b82f6")
        st.markdown(f"""
        <div class="alert-card {css_class}" style="border-left-color:{border_color}">
            <div style="font-weight:700; font-size:0.82rem; color:#e6edf3; margin-bottom:3px">
                {alert['icon']} {alert['title']}
            </div>
            <div style="font-size:0.78rem; color:#8b949e">{alert['message']}</div>
            <div style="font-size:0.7rem; color:#484f58; margin-top:4px; font-family:'Space Mono',monospace">{alert['time']}</div>
        </div>
        """, unsafe_allow_html=True)

with col_res:
    st.markdown('<div class="section-title">🏥 Resource Panel</div>', unsafe_allow_html=True)

    food_color = {
        "Good": "#22c55e", "Adequate": "#3b82f6",
        "Low": "#eab308", "Critical": "#ef4444"
    }.get(resources.get("food_status","Adequate"), "#8b949e")

    items = [
        ("🏥", "Hospitals", str(resources.get("hospitals","—"))),
        ("🏠", "Shelters", str(resources.get("shelters","—"))),
        ("🍱", "Food Supply", resources.get("food_status","—")),
        ("⛺", "Relief Centers", str(resources.get("relief_centers","—"))),
    ]
    for icon, label, val in items:
        color = food_color if label == "Food Supply" else "#e6edf3"
        st.markdown(f"""
        <div class="resource-row">
            <span style="color:#8b949e">{icon} {label}</span>
            <span style="font-weight:700; color:{color}; font-family:'Space Mono',monospace">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📍 Relief Centers</div>', unsafe_allow_html=True)
    for rc in relief[:3]:
        type_icon = {"Medical":"🏥","Shelter":"🏠","Food":"🍱"}.get(rc["type"],"📍")
        st.markdown(f"""
        <div style="font-family:'DM Sans',sans-serif;font-size:0.78rem;
                    padding:5px 0;border-bottom:1px solid #21262d;color:#8b949e">
            {type_icon} <b style="color:#e6edf3">{rc['name']}</b><br>
            &nbsp;&nbsp;&nbsp;&nbsp;{rc['type']} · {rc['lat']:.4f}, {rc['lon']:.4f}
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; font-family:'Space Mono',monospace; font-size:0.65rem; 
            color:#484f58; border-top:1px solid #21262d; padding-top:1rem">
    DisasterSense v1.0 · ML Model: XGBoost · Weather: Open-Meteo API · 
    Trained on India Rainfall 1901–2015 · {datetime.now().strftime('%Y')}
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if auto_refresh:
    time.sleep(120)
    st.cache_data.clear()
    st.rerun()
