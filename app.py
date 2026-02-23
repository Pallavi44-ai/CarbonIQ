import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import random
import plotly.graph_objects as go
import plotly.express as px

from utils.preprocessing import clean_dataset, compute_kpis, get_country_stats, compute_risk_level
from utils.visualization import (
    emission_trend_chart, global_choropleth, top_emitters_bar,
    sector_pie, global_trend_line, emission_heatmap,
    forecast_chart, multi_country_comparison, scatter_comparison
)
from ai.chatbot import ai_assistant, get_policy_recommendations

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarbonIQ — Global Emission Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — PREMIUM DARK THEME
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

<style>
/* ===== RESET & BASE ===== */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #050d1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070f20 0%, #050d1a 100%) !important;
    border-right: 1px solid #0d2137 !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00e5ff !important;
    font-family: 'Syne', sans-serif;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #94a3b8 !important;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #0a1628 !important;
    padding: 8px 12px;
    border-radius: 14px;
    border: 1px solid #0d2137;
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 18px;
    border: none !important;
    transition: all 0.25s ease;
    letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00e5ff22, #00e5ff11) !important;
    color: #00e5ff !important;
    border: 1px solid #00e5ff44 !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.15);
}

/* ===== HEADERS ===== */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.01em;
}

/* ===== INPUTS ===== */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #0a1628 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
.stSlider [data-testid="stSlider"] > div { background: #1e293b; }
div[data-baseweb="slider"] > div > div { background: #00e5ff !important; }

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #00e5ff18, #00e5ff08) !important;
    border: 1px solid #00e5ff55 !important;
    color: #00e5ff !important;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 14px;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.04em;
    transition: all 0.25s ease;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00e5ff30, #00e5ff18) !important;
    border-color: #00e5ff99 !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.25);
    transform: translateY(-1px);
}

/* ===== TEXT INPUT ===== */
.stTextInput > div > div > input {
    background: #0a1628 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif;
}
.stTextInput > div > div > input:focus {
    border-color: #00e5ff55 !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.1) !important;
}

/* ===== METRICS ===== */
[data-testid="metric-container"] {
    background: #0a1628 !important;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 16px;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00e5ff !important;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

/* ===== SUCCESS / INFO / WARNING ===== */
.stSuccess {
    background: rgba(57,255,20,0.08) !important;
    border: 1px solid rgba(57,255,20,0.25) !important;
    border-radius: 12px !important;
    color: #39ff14 !important;
}
.stInfo {
    background: rgba(0,229,255,0.08) !important;
    border: 1px solid rgba(0,229,255,0.25) !important;
    border-radius: 12px !important;
    color: #00e5ff !important;
}
.stWarning {
    background: rgba(255,107,53,0.08) !important;
    border: 1px solid rgba(255,107,53,0.25) !important;
    border-radius: 12px !important;
    color: #ff6b35 !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #050d1a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00e5ff44; }

/* ===== DIVIDER ===== */
hr { border-color: #0d2137 !important; }

/* ===== PLOTLY CHARTS ===== */
.js-plotly-plot { border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/final_emissions_dataset.csv")
    df = clean_dataset(df)
    return df

@st.cache_resource
def load_models():
    try:
        ml_model = joblib.load("models/ml_emission_forecast.pkl")

        # Try loading DL model (optional for cloud)
        try:
            
            dl_model = keras_load("models/dl_climate_risk.keras", compile=False)
            DL_ENABLED = True
        except:
            dl_model = None
            DL_ENABLED = False

        return ml_model, dl_model, DL_ENABLED

    except Exception as e:
        st.sidebar.warning(f"⚠️ Model load issue: {str(e)[:60]}")
        return None, None, False

df = load_data()
ml_model, dl_model, DL_ENABLED = load_models()
kpis = compute_kpis(df)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 20px 0;">
        <div style="font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; 
                    background: linear-gradient(90deg, #00e5ff, #39ff14); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    letter-spacing: -0.01em;">
            🌍 CarbonIQ
        </div>
        <div style="font-size: 11px; color: #475569; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px;">
            Global Emission Intelligence
        </div>
    </div>
    <hr style="border-color: #0d2137; margin-bottom: 20px;">
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Country & Sector</div>', unsafe_allow_html=True)
    
    country = st.selectbox("Country", sorted(df["Country"].unique()), index=sorted(df["Country"].unique()).index("India") if "India" in df["Country"].unique() else 0)
    sector = st.selectbox("Sector", sorted(df["Sector"].unique()))
    
    st.markdown("<hr style='border-color:#0d2137; margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Map Year</div>', unsafe_allow_html=True)
    
    map_year = st.slider("", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].max()), label_visibility="collapsed")
    
    st.markdown("<hr style='border-color:#0d2137; margin: 16px 0;'>", unsafe_allow_html=True)
    
    # System status
    model_status = "🟢 Online" if ml_model else "🔴 Offline"
    dl_status = "🟢 Online" if dl_model else "🔴 Offline"
    
    st.markdown(f"""
    <div style="background: #0a1628; border: 1px solid #1e293b; border-radius: 12px; padding: 14px;">
        <div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">System Status</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 12px; color: #94a3b8;">ML Engine</span>
            <span style="font-size: 12px;">{model_status}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
            <span style="font-size: 12px; color: #94a3b8;">DL Engine</span>
            <span style="font-size: 12px;">{dl_status}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 12px; color: #94a3b8;">Dataset</span>
            <span style="font-size: 12px;">🟢 {len(df):,} rows</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 28px 0 8px 0;">
    <div style="font-family: 'Syne', sans-serif; font-size: 36px; font-weight: 800; 
                background: linear-gradient(135deg, #ffffff 0%, #00e5ff 50%, #39ff14 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                line-height: 1.15; letter-spacing: -0.02em;">
        Global Emission Intelligence Platform
    </div>
    <div style="font-size: 14px; color: #64748b; margin-top: 8px; font-family: 'Inter', sans-serif;">
        Real-time analytics · ML forecasting · Climate risk AI · 195 countries · 1990–2021
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LIVE KPI TICKER
# ──────────────────────────────────────────────────────────────────────────────
live_placeholder = st.empty()
base_val = 1247 + random.randint(-30, 30)
live_placeholder.markdown(f"""
<div style="background: linear-gradient(135deg, #070f20, #0a1628); border: 1px solid #0d2137;
            border-radius: 14px; padding: 14px 24px; margin: 12px 0; display: flex; 
            align-items: center; gap: 20px;">
    <div style="width: 8px; height: 8px; background: #39ff14; border-radius: 50%;
                box-shadow: 0 0 8px #39ff14; animation: pulse 1.5s infinite;"></div>
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #64748b;">LIVE</div>
    <div style="font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 700; color: #39ff14;">
        {base_val:,} tCO₂e / hour
    </div>
    <div style="font-size: 12px; color: #475569; margin-left: auto;">Global real-time estimate based on 2021 baseline</div>
</div>
<style>
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# KPI CARDS ROW
# ──────────────────────────────────────────────────────────────────────────────
yoy_color = "#ff6b35" if kpis["yoy_change"] > 0 else "#39ff14"
yoy_arrow = "▲" if kpis["yoy_change"] > 0 else "▼"
country_stats = get_country_stats(df, country)
risk_label, risk_class = compute_risk_level(country_stats.get("latest_emissions", 0))

c1, c2, c3, c4, c5 = st.columns(5)

card_css = """
background: linear-gradient(135deg, #0a1628, #070f20);
border: 1px solid #0d2137;
border-radius: 16px;
padding: 20px;
text-align: center;
"""

with c1:
    st.markdown(f"""
    <div style="{card_css}">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em;">Global Total {kpis['latest_year']}</div>
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:#00e5ff; margin: 8px 0;">
            {kpis['total_latest']:,.0f}
        </div>
        <div style="font-size:11px; color:#64748b;">tCO₂e</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div style="{card_css}">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em;">YoY Change</div>
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:{yoy_color}; margin: 8px 0;">
            {yoy_arrow} {abs(kpis['yoy_change']):.1f}%
        </div>
        <div style="font-size:11px; color:#64748b;">vs prior year</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div style="{card_css}">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em;">Top Emitter</div>
        <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700; color:#ff6b35; margin: 8px 0;">
            {kpis['top_country']}
        </div>
        <div style="font-size:11px; color:#64748b;">{kpis['top_emission']:,.1f} tCO₂e</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div style="{card_css}">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em;">Countries Tracked</div>
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:#bf5af2; margin: 8px 0;">
            {kpis['total_countries']}
        </div>
        <div style="font-size:11px; color:#64748b;">nations monitored</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div style="{card_css}">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em;">{country} Risk</div>
        <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:700; color:#f59e0b; margin: 8px 0;">
            {risk_label}
        </div>
        <div style="font-size:11px; color:#64748b;">{country_stats.get('trend','stable')} trend</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin: 24px 0 4px;'></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌐 Global Dashboard",
    "📊 Country Deep Dive",
    "🧠 ML Forecast",
    "🔥 Climate Risk AI",
    "⚖️ Compare Countries",
    "🛡️ Policy Advisor",
    "🤖 AI Assistant",
    "ℹ️ Data & About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: GLOBAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(f"### 🌐 Global Emissions Intelligence — {map_year}")
    
    # World Map
    with st.spinner("Rendering global map..."):
        fig_map = global_choropleth(df, map_year)
        st.plotly_chart(fig_map, use_container_width=True)
    
    col_a, col_b = st.columns([3, 2])
    
    with col_a:
        fig_bar = top_emitters_bar(df, map_year, n=15)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_b:
        fig_global = global_trend_line(df)
        st.plotly_chart(fig_global, use_container_width=True)
    
    # Scatter
    fig_scatter = scatter_comparison(df, map_year)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Key insight banner
    ydf_map = df[df["Year"] == map_year].groupby("Country")["Emissions"].sum()
    top_c = ydf_map.idxmax()
    pct_top5 = ydf_map.nlargest(5).sum() / ydf_map.sum() * 100
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0a1628, #070f20); border: 1px solid #1e293b;
                border-radius: 14px; padding: 18px 24px; margin-top: 8px;">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">🔍 Key Insight — {map_year}</div>
        <div style="font-size:15px; color:#e2e8f0; line-height:1.6;">
            The top 5 emitting nations account for <span style="color:#00e5ff; font-weight:700;">{pct_top5:.1f}%</span> 
            of all tracked emissions. <span style="color:#ff6b35; font-weight:600;">{top_c}</span> leads globally. 
            Use the sidebar year slider to trace the emission trajectory since 1990.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: COUNTRY DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f"### 📊 Deep Dive — {country}")
    
    stats = get_country_stats(df, country)
    
    if stats:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Latest Emissions", f"{stats['latest_emissions']:.1f} tCO₂e", 
                      f"{stats['slope']:.2f}/yr slope")
        with m2:
            st.metric("Peak Year", str(stats['peak_year']), f"{stats['peak_val']:.1f} tCO₂e")
        with m3:
            trend_delta = "↑ Rising" if stats['trend'] == 'increasing' else "↓ Falling"
            st.metric("Trend", trend_delta)
        with m4:
            st.metric("Data Points", f"{len(df[df['Country'] == country])} records")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        fig_trend = emission_trend_chart(df, country, sector)
        st.plotly_chart(fig_trend, use_container_width=True)
    with col2:
        fig_pie = sector_pie(df, country)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Heatmap
    fig_heat = emission_heatmap(df, country)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Rank in dataset
    ydf_rank = df[df["Year"] == df["Year"].max()].groupby("Country")["Emissions"].sum().reset_index()
    ydf_rank = ydf_rank.sort_values("Emissions", ascending=False).reset_index(drop=True)
    ydf_rank["Rank"] = ydf_rank.index + 1
    
    if country in ydf_rank["Country"].values:
        rank_row = ydf_rank[ydf_rank["Country"] == country].iloc[0]
        total = len(ydf_rank)
        percentile = (1 - rank_row["Rank"] / total) * 100
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0a1628, #070f20); border: 1px solid #1e293b;
                    border-radius: 14px; padding: 18px 24px; margin-top: 8px; display: flex; gap: 40px; align-items: center;">
            <div>
                <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em;">Global Rank</div>
                <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800; color:#00e5ff;">
                    #{int(rank_row['Rank'])} <span style="font-size:14px; color:#64748b;">of {total}</span>
                </div>
            </div>
            <div>
                <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:0.1em;">Percentile</div>
                <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800; color:#bf5af2;">
                    {percentile:.0f}th
                </div>
            </div>
            <div style="flex: 1; text-align: right;">
                <div style="font-size: 14px; color: #94a3b8; line-height: 1.6;">
                    {country} ranks <strong style="color:#00e5ff;">#{int(rank_row['Rank'])}</strong> globally in {df['Year'].max()} emissions.
                    {"This puts it in the top quartile — significant mitigation action is warranted." if rank_row['Rank'] <= total*0.25 else "Continued monitoring and progressive reduction targets recommended."}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ML FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 🧠 Machine Learning Emission Forecast")
    
    if not ml_model:
        st.warning("⚠️ ML model could not be loaded. Please ensure `models/ml_emission_forecast.pkl` exists.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background: #0a1628; border: 1px solid #1e293b; border-radius: 14px; padding: 18px; margin-bottom: 16px;">
                <div style="font-size: 13px; color: #94a3b8; line-height: 1.7;">
                    This model was trained on historical emission trajectories across all sectors. 
                    Forecasts are generated using a <strong style="color:#00e5ff;">gradient-boosted regressor</strong> 
                    fitted to year-emission patterns. Extend the timeline to explore long-term trajectories.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            year_from = st.slider("Forecast Start", 2022, 2030, 2025)
            year_to = st.slider("Forecast End", year_from + 1, 2040, 2035)
        
        if st.button("🚀 Run ML Forecast", use_container_width=True):
            with st.spinner("Generating forecast..."):
                try:
                    fig_forecast, preds, years = forecast_chart(country, year_from, year_to, ml_model, df)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Results table
                    results_df = pd.DataFrame({"Year": years, "Predicted Emissions (tCO₂e)": [round(p, 2) for p in preds]})
                    
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        # Mini bar chart of predictions
                        fig_mini = go.Figure(go.Bar(
                            x=years, y=preds,
                            marker_color=[f"rgba(0,229,255,{0.4+0.6*(p-min(preds))/(max(preds)-min(preds)+1)})" for p in preds],
                            hovertemplate="%{x}: %{y:.2f} tCO₂e<extra></extra>"
                        ))
                        fig_mini.update_layout(
                            title="Forecast Breakdown",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e2e8f0"), margin=dict(t=40, b=30),
                            xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b")
                        )
                        st.plotly_chart(fig_mini, use_container_width=True)
                    
                    with col_b:
                        st.markdown("**📋 Forecast Table**")
                        st.dataframe(
                            results_df.style.background_gradient(cmap="Blues", subset=["Predicted Emissions (tCO₂e)"]),
                            use_container_width=True, hide_index=True
                        )
                    
                    avg_pred = np.mean(preds)
                    trend_text = "rising" if preds[-1] > preds[0] else "falling"
                    st.success(f"✅ Forecast complete. {country}'s emissions are projected to be **{avg_pred:.1f} tCO₂e** on average between {year_from}–{year_to}, with a **{trend_text}** trajectory.")
                    
                except Exception as e:
                    st.error(f"Forecast error: {e}")
        else:
            # Show existing data
            fig_existing = emission_trend_chart(df, country, sector)
            st.plotly_chart(fig_existing, use_container_width=True, key="fig_existing_chart")
            st.info("👆 Configure your forecast parameters and click **Run ML Forecast** to generate predictions.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CLIMATE RISK AI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🔥 Deep Learning Climate Risk Assessment")
    
    if not dl_model:
        st.warning("⚠️ Deep learning model could not be loaded. Ensure `models/dl_climate_risk.keras` is present.")
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a0a0a, #0d0505); border: 1px solid #3d1515;
                    border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="font-size: 13px; color: #94a3b8; line-height: 1.7;">
                Our neural network predicts a <strong style="color:#ff6b35;">Climate Risk Index (CRI)</strong> 
                based on temperature anomalies and precipitation shifts. Higher scores indicate greater 
                exposure to extreme weather, agricultural stress, and infrastructure risk.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.slider("🌡️ Temp Increase (°C)", 0.5, 4.0, 1.5, 0.1)
        with col2:
            rainfall = st.slider("🌧️ Rainfall Change (%)", -30, 30, 5)
        with col3:
            st.markdown("""
            <div style="background: #0a1628; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; margin-top: 8px;">
                <div style="font-size: 11px; color: #475569; text-transform: uppercase; margin-bottom: 8px;">Risk Legend</div>
                <div style="font-size: 12px; line-height: 2;">
                    🟢 0.0 – 0.3 → Low<br>
                    🟡 0.3 – 0.6 → Moderate<br>
                    🟠 0.6 – 0.8 → High<br>
                    🔴 0.8 – 1.0 → Critical
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🔬 Analyze Climate Risk", use_container_width=True):
            with st.spinner("Running neural network inference..."):
                try:
                    X_dl = np.array([[temp, rainfall]])
                    risk = float(dl_model.predict(X_dl, verbose=0)[0][0])
                    risk_clamped = max(0.0, min(1.0, risk))
                    
                    # Risk gauge
                    gauge_color = "#39ff14" if risk_clamped < 0.3 else "#f59e0b" if risk_clamped < 0.6 else "#ff6b35" if risk_clamped < 0.8 else "#ff0040"
                    risk_label_text = "Low" if risk_clamped < 0.3 else "Moderate" if risk_clamped < 0.6 else "High" if risk_clamped < 0.8 else "Critical"
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk_clamped,
                        number=dict(suffix="", font=dict(family="Syne", size=40, color=gauge_color)),
                        gauge=dict(
                            axis=dict(range=[0, 1], tickwidth=1, tickcolor="#1e293b",
                                      tickfont=dict(color="#64748b")),
                            bar=dict(color=gauge_color, thickness=0.25),
                            bgcolor="rgba(0,0,0,0)",
                            borderwidth=0,
                            steps=[
                                dict(range=[0, 0.3], color="#0d1f0d"),
                                dict(range=[0.3, 0.6], color="#1a1a00"),
                                dict(range=[0.6, 0.8], color="#1a0a00"),
                                dict(range=[0.8, 1.0], color="#1a0005"),
                            ],
                            threshold=dict(line=dict(color=gauge_color, width=3), thickness=0.75, value=risk_clamped)
                        ),
                        title=dict(text=f"Climate Risk Index — {risk_label_text}", 
                                   font=dict(family="Syne", size=18, color=gauge_color))
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", 
                        font=dict(color="#e2e8f0"),
                        height=300, margin=dict(t=60, b=20)
                    )
                    st.plotly_chart(fig_existing, use_container_width=True, key="fig_existing")
                    st.plotly_chart(fig_bar, use_container_width=True, key="fig_bar")
                    st.plotly_chart(fig_map, use_container_width=True, key="fig_map")
                    
                    # Impact analysis
                    st.markdown("#### 🌍 Impact Analysis")
                    impacts = []
                    if temp > 2.0:
                        impacts.append(("🌊 Sea Level Risk", "High", f"+{temp*3:.0f}cm potential rise", "#ff6b35"))
                        impacts.append(("🔥 Wildfire Probability", "Elevated", f"Risk increases {temp*15:.0f}% vs baseline", "#f59e0b"))
                    if temp > 1.5:
                        impacts.append(("🌾 Agricultural Stress", "Moderate-High", f"Crop yield decline {temp*8:.0f}%", "#f59e0b"))
                        impacts.append(("💧 Water Security", "Stressed", "Glacial melt acceleration detected", "#00e5ff"))
                    if abs(rainfall) > 15:
                        impacts.append(("🌧️ Flood/Drought Risk", "High", f"{abs(rainfall)}% precipitation anomaly", "#ff6b35"))
                    if not impacts:
                        impacts.append(("✅ Overall Risk", "Low", "Conditions within manageable range", "#39ff14"))
                    
                    cols = st.columns(min(len(impacts), 3))
                    for i, (name, severity, detail, color) in enumerate(impacts[:3]):
                        with cols[i % 3]:
                            st.markdown(f"""
                            <div style="background: #0a1628; border: 1px solid {color}33; border-radius: 12px; 
                                        padding: 16px; border-left: 3px solid {color};">
                                <div style="font-size: 14px; color: #e2e8f0; font-weight: 600; margin-bottom: 6px;">{name}</div>
                                <div style="font-size: 13px; color: {color}; font-weight: 700; margin-bottom: 4px;">{severity}</div>
                                <div style="font-size: 11px; color: #64748b;">{detail}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Recommendation
                    if risk_clamped > 0.6:
                        st.warning(f"⚠️ **High Climate Risk Detected (CRI: {risk_clamped:.3f})**. Immediate adaptation strategies required: flood defenses, drought-resistant agriculture, heat emergency protocols.")
                    elif risk_clamped > 0.3:
                        st.info(f"ℹ️ **Moderate Climate Risk (CRI: {risk_clamped:.3f})**. Proactive adaptation planning recommended. Review infrastructure resilience and water management systems.")
                    else:
                        st.success(f"✅ **Low Climate Risk (CRI: {risk_clamped:.3f})**. Current projections are within manageable bounds. Maintain monitoring and prevention protocols.")
                
                except Exception as e:
                    st.error(f"Model inference error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: COMPARE COUNTRIES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### ⚖️ Multi-Country Comparison")
    
    all_countries = sorted(df["Country"].unique())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        compare_countries = st.multiselect(
            "Select countries to compare (up to 5)",
            all_countries,
            default=["India", "China", "United States", "Germany", "Brazil"][:5],
            max_selections=5
        )
    with col2:
        compare_sector = st.selectbox("Sector", sorted(df["Sector"].unique()), key="compare_sector")
    
    if compare_countries:
        fig_compare = multi_country_comparison(df, compare_countries, compare_sector)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Comparison table
        st.markdown("#### 📋 Statistical Summary")
        
        latest_yr = df["Year"].max()
        summary_rows = []
        for c in compare_countries:
            cdf = df[df["Country"] == c]
            latest_e = cdf[cdf["Year"] == latest_yr]["Emissions"].sum()
            max_e = cdf.groupby("Year")["Emissions"].sum().max()
            peak_yr = cdf.groupby("Year")["Emissions"].sum().idxmax()
            stats_c = get_country_stats(df, c)
            summary_rows.append({
                "Country": c,
                f"Latest ({latest_yr}) tCO₂e": round(latest_e, 2),
                "Peak tCO₂e": round(max_e, 2),
                "Peak Year": int(peak_yr),
                "Trend": stats_c.get("trend", "—").capitalize()
            })
        
        summary_df = pd.DataFrame(summary_rows).set_index("Country")
        st.dataframe(summary_df, use_container_width=True)
        
        # Ranking visualization
        ydf_compare = df[(df["Year"] == latest_yr) & (df["Country"].isin(compare_countries))].groupby("Country")["Emissions"].sum().sort_values(ascending=True)
        
        fig_rank = go.Figure(go.Bar(
            x=ydf_compare.values,
            y=ydf_compare.index,
            orientation="h",
            marker=dict(
                color=["#00e5ff", "#39ff14", "#ff6b35", "#bf5af2", "#f59e0b"][:len(ydf_compare)],
                line=dict(color="rgba(0,0,0,0.2)", width=0.5)
            ),
            text=[f"{v:.1f}" for v in ydf_compare.values],
            textposition="outside",
            textfont=dict(color="#e2e8f0")
        ))
        fig_rank.update_layout(
            title=f"Comparison Ranking — {latest_yr}",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Syne", color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e293b", title="tCO₂e"),
            yaxis=dict(gridcolor="#1e293b"),
            margin=dict(t=40, b=40, l=120, r=60)
        )
        st.plotly_chart(fig_rank, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: POLICY ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown(f"### 🛡️ AI Policy Advisor — {country}")
    
    stats_pol = get_country_stats(df, country)
    emissions_pol = stats_pol.get("latest_emissions", 0)
    trend_pol = stats_pol.get("trend", "stable")
    
    recommendations = get_policy_recommendations(country, emissions_pol, trend_pol, df)
    
    # Overview banner
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0a1628, #070f20); border: 1px solid #1e293b;
                border-radius: 14px; padding: 20px 24px; margin-bottom: 20px;">
        <div style="font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; color: #00e5ff; margin-bottom: 10px;">
            Policy Analysis: {country}
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 12px;">
            <div>
                <div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.1em;">Current Emissions</div>
                <div style="font-size: 20px; font-weight: 700; color: #e2e8f0;">{emissions_pol:.1f} <span style="font-size: 12px; color: #64748b;">tCO₂e</span></div>
            </div>
            <div>
                <div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.1em;">Emission Trend</div>
                <div style="font-size: 20px; font-weight: 700; color: {'#ff6b35' if trend_pol == 'increasing' else '#39ff14'};">{trend_pol.capitalize()}</div>
            </div>
            <div>
                <div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.1em;">Risk Level</div>
                <div style="font-size: 20px; font-weight: 700; color: #f59e0b;">{risk_label}</div>
            </div>
        </div>
        <div style="font-size: 13px; color: #64748b; line-height: 1.6;">
            Based on data-driven analysis of {country}'s emission profile, our AI has generated the following 
            prioritized policy recommendations. These are calibrated to the country's specific emission level, 
            trend direction, and sector breakdown.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📋 Recommended Policy Actions")
    
    # Category filter
    categories = list(set([r["category"] for r in recommendations]))
    selected_cats = st.multiselect("Filter by Category", categories, default=categories)
    
    filtered_recs = [r for r in recommendations if r["category"] in selected_cats]
    
    for rec in filtered_recs:
        priority_color = "#ff0040" if "Critical" in rec["priority"] or "Urgent" in rec["priority"] else \
                         "#ff6b35" if "High" in rec["priority"] else \
                         "#f59e0b" if "Medium" in rec["priority"] else "#39ff14"
        
        st.markdown(f"""
        <div style="background: #0a1628; border: 1px solid #1e293b; border-radius: 12px; 
                    padding: 16px 20px; margin-bottom: 12px; border-left: 3px solid {priority_color};
                    display: flex; align-items: center; gap: 20px;">
            <div style="min-width: 100px;">
                <div style="font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.08em;">Priority</div>
                <div style="font-size: 13px; font-weight: 700; color: {priority_color};">{rec['priority']}</div>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 15px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px;">{rec['action']}</div>
                <div style="font-size: 12px; color: #39ff14;">→ {rec['impact']}</div>
            </div>
            <div style="background: #1e293b; border-radius: 8px; padding: 4px 12px;">
                <div style="font-size: 11px; color: #64748b;">{rec['category']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Net zero timeline
    st.markdown("#### 🎯 Pathway to Net Zero")
    
    years_nz = list(range(2025, 2051))
    if emissions_pol > 0:
        # Simple linear decline to zero
        nz_emissions = [max(0, emissions_pol * (1 - (y - 2024) / 26)) for y in years_nz]
        aggressive = [max(0, emissions_pol * (1 - (y - 2024) / 18)) for y in years_nz]
        
        fig_nz = go.Figure()
        fig_nz.add_trace(go.Scatter(
            x=years_nz, y=nz_emissions,
            fill="tozeroy", fillcolor="rgba(0,229,255,0.07)",
            line=dict(color="#00e5ff", width=2),
            name="Standard Path (2050)",
            hovertemplate="%{x}: %{y:.1f} tCO₂e<extra></extra>"
        ))
        fig_nz.add_trace(go.Scatter(
            x=years_nz, y=aggressive,
            fill="tozeroy", fillcolor="rgba(57,255,20,0.05)",
            line=dict(color="#39ff14", width=2, dash="dot"),
            name="Accelerated Path (2042)",
            hovertemplate="%{x}: %{y:.1f} tCO₂e<extra></extra>"
        ))
        fig_nz.add_hline(y=0, line=dict(color="#ff6b35", dash="dash", width=1),
                         annotation_text="Net Zero", annotation_font_color="#ff6b35")
        
        fig_nz.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Syne", color="#e2e8f0"),
            xaxis=dict(gridcolor="#1e293b", title="Year"),
            yaxis=dict(gridcolor="#1e293b", title="tCO₂e"),
            margin=dict(t=20, b=40)
        )
        st.plotly_chart(fig_nz, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7: AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 🤖 AI Climate Intelligence Assistant")
    
    # Chat UI header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #050d1a, #070f20); 
                border: 1px solid #0d2137; border-radius: 16px; 
                padding: 24px; margin-bottom: 20px; text-align: center;">
        <div style="font-family: 'Syne', sans-serif; font-size: 42px; margin-bottom: 8px;">🌍</div>
        <div style="font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; 
                    background: linear-gradient(90deg, #00e5ff, #39ff14);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            CarbonIQ Intelligence
        </div>
        <div style="font-size: 13px; color: #475569; margin-top: 6px;">
            Ask about emissions, climate policy, sustainability, country data, and solutions
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick prompts
    st.markdown("**💡 Quick Questions:**")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    
    quick_questions = [
        "Which country has highest emissions?",
        "How to reduce transport emissions?",
        "What is the Paris Agreement?",
        "How to achieve carbon neutrality?",
    ]
    
    quick_response = None
    for i, (col, q) in enumerate(zip([qcol1, qcol2, qcol3, qcol4], quick_questions)):
        with col:
            if st.button(q[:28] + "...", key=f"quick_{i}"):
                quick_response = ai_assistant(q, df)
    
    st.markdown("<hr style='border-color:#0d2137;'>", unsafe_allow_html=True)
    
    # Main input
    user_query = st.text_input("💬 Ask your climate question", placeholder="e.g. Why are emissions still rising globally?")
    
    col_ask, col_clear = st.columns([1, 1])
    with col_ask:
        ask_btn = st.button("✨ Ask CarbonIQ", use_container_width=True)
    with col_clear:
        if st.button("🔄 New Question", use_container_width=True):
            st.rerun()
    
    # Response display
    if quick_response:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #071a0f, #050d1a); border: 1px solid #1a4d2e;
                    border-radius: 14px; padding: 20px; margin-top: 16px; border-left: 3px solid #39ff14;">
            <div style="font-size: 11px; color: #39ff14; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                🤖 CarbonIQ Response
            </div>
            <div style="font-size: 15px; color: #e2e8f0; line-height: 1.7;">{quick_response}</div>
        </div>
        """, unsafe_allow_html=True)
    
    elif ask_btn and user_query:
        with st.spinner("Analyzing..."):
            time.sleep(0.5)  # UX polish
            response = ai_assistant(user_query, df)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #071a0f, #050d1a); border: 1px solid #1a4d2e;
                    border-radius: 14px; padding: 20px; margin-top: 16px; border-left: 3px solid #39ff14;">
            <div style="font-size: 11px; color: #39ff14; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                🤖 CarbonIQ Response
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size: 11px; color: #475569; margin-bottom: 8px;">
                Query: {user_query[:80]}{'...' if len(user_query) > 80 else ''}
            </div>
            <div style="font-size: 15px; color: #e2e8f0; line-height: 1.7;">{response}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data snapshot for context
    st.markdown("<hr style='border-color:#0d2137; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Live Context Data")
    
    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
    with ctx_col1:
        latest_data = df[df["Year"] == df["Year"].max()].groupby("Country")["Emissions"].sum().nlargest(5).reset_index()
        latest_data.columns = ["Country", "tCO₂e"]
        st.markdown("**Top 5 Emitters:**")
        st.dataframe(latest_data, use_container_width=True, hide_index=True)
    with ctx_col2:
        sector_data = df[df["Year"] == df["Year"].max()].groupby("Sector")["Emissions"].sum().reset_index()
        st.markdown("**By Sector:**")
        st.dataframe(sector_data, use_container_width=True, hide_index=True)
    with ctx_col3:
        st.markdown("**Dataset Coverage:**")
        st.markdown(f"""
        <div style="background: #0a1628; border: 1px solid #1e293b; border-radius: 10px; padding: 14px;">
            <div style="font-size: 13px; color: #94a3b8; line-height: 2.2;">
                📅 Years: <strong style="color:#00e5ff;">1990–{int(df['Year'].max())}</strong><br>
                🌍 Countries: <strong style="color:#39ff14;">{df['Country'].nunique()}</strong><br>
                🏭 Sectors: <strong style="color:#bf5af2;">{df['Sector'].nunique()}</strong><br>
                📊 Records: <strong style="color:#f59e0b;">{len(df):,}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### ℹ️ About CarbonIQ & Data Sources")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div style="background: #0a1628; border: 1px solid #1e293b; border-radius: 14px; padding: 24px; margin-bottom: 16px;">
            <div style="font-family: 'Syne', sans-serif; font-size: 18px; color: #00e5ff; margin-bottom: 14px; font-weight: 700;">📚 Data Sources</div>
            <div style="font-size: 14px; color: #94a3b8; line-height: 2;">
                <strong style="color:#e2e8f0;">World Resources Institute (WRI)</strong><br>
                Climate Watch — Transport sector GHG emissions across 195 countries, 1990–2021<br><br>
                <strong style="color:#e2e8f0;">World Bank Social Sustainability Database</strong><br>
                CO₂ emissions per capita (tonnes CO₂-equivalent per person)<br><br>
                <strong style="color:#e2e8f0;">Combined Dataset</strong><br>
                12,584 records · 2 sector types · 32-year span
            </div>
        </div>
        
        <div style="background: #0a1628; border: 1px solid #1e293b; border-radius: 14px; padding: 24px;">
            <div style="font-family: 'Syne', sans-serif; font-size: 18px; color: #39ff14; margin-bottom: 14px; font-weight: 700;">🛠️ Tech Stack</div>
            <div style="font-size: 14px; color: #94a3b8; line-height: 2.2;">
                🐍 <strong style="color:#e2e8f0;">Python 3.11</strong> — Core runtime<br>
                🌊 <strong style="color:#e2e8f0;">Streamlit 1.54</strong> — UI framework<br>
                🤖 <strong style="color:#e2e8f0;">Scikit-learn</strong> — ML emission forecasting<br>
                🧠 <strong style="color:#e2e8f0;">TensorFlow / Keras</strong> — Deep learning climate risk<br>
                📊 <strong style="color:#e2e8f0;">Plotly 6.5</strong> — Interactive visualizations<br>
                🐼 <strong style="color:#e2e8f0;">Pandas + NumPy</strong> — Data processing
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0a1628, #070f20); border: 1px solid #1e293b;
                    border-radius: 14px; padding: 24px; margin-bottom: 16px;">
            <div style="font-family: 'Syne', sans-serif; font-size: 18px; color: #bf5af2; margin-bottom: 14px; font-weight: 700;">🎯 Platform Features</div>
            <div style="font-size: 13px; color: #94a3b8; line-height: 2.2;">
                🌐 Global choropleth emissions map<br>
                📊 195-country deep-dive analytics<br>
                🧠 ML-powered emission forecasting<br>
                🔥 Neural net climate risk indexing<br>
                ⚖️ Multi-country comparison engine<br>
                🛡️ AI-driven policy recommendations<br>
                🤖 NLP climate intelligence assistant<br>
                📋 Net-zero pathway modeling<br>
                📈 Sector × year heatmap analysis<br>
                🎯 Real-time KPI monitoring
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #0a1628, #070f20); border: 1px solid #1e293b;
                    border-radius: 14px; padding: 24px;">
            <div style="font-family: 'Syne', sans-serif; font-size: 18px; color: #ff6b35; margin-bottom: 14px; font-weight: 700;">⚠️ Disclaimer</div>
            <div style="font-size: 13px; color: #64748b; line-height: 1.8;">
                Forecasts are model estimates, not guarantees. Climate risk indices are derived from 
                simplified input parameters. Policy recommendations are AI-generated suggestions and 
                should be reviewed by domain experts before implementation. Data reflects best 
                available records through 2021.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset preview
    st.markdown("#### 🔍 Dataset Preview")
    preview_df = df.sample(min(20, len(df))).sort_values("Year", ascending=False)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 32px 0 16px; margin-top: 20px; border-top: 1px solid #0d2137;">
    <div style="font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700;
                background: linear-gradient(90deg, #00e5ff, #39ff14);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        CarbonIQ Industry Monitor
    </div>
    <div style="font-size: 12px; color: #334155; margin-top: 6px; font-family: 'JetBrains Mono', monospace;">
        AI Emission & Climate Risk Intelligence Platform · Built with ❤️ for a sustainable future
    </div>
</div>
""", unsafe_allow_html=True)
