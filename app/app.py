import streamlit as st
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="ElectroGuard — Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Google Fonts ────────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── Premium CSS (injected via st.html to avoid visible text bleed) ─────────────
_CSS = """
<style>
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main {
    background: radial-gradient(ellipse at 0% 0%, #0d1b2a 0%, #07080f 50%, #0a0f1a 100%);
    min-height: 100vh;
}
.block-container {
    padding: 1.2rem 2rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #111827 100%);
    border-right: 1px solid rgba(99,179,237,0.12);
}
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] p { color: #cbd5e1 !important; }

.sidebar-section {
    background: rgba(99,179,237,0.07);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 8px;
    padding: 7px 14px;
    margin: 14px 0 10px 0;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #63b3ed !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(17,24,39,0.92) 50%, rgba(10,15,30,0.95) 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -80px; left: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(233,69,96,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; bottom: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(233,69,96,0.15);
    border: 1px solid rgba(233,69,96,0.4);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #fc8181; margin-bottom: 12px;
}
.hero h1 {
    color: #f8fafc; font-size: 2.1rem; font-weight: 800;
    margin: 0 0 8px 0; letter-spacing: -0.5px; line-height: 1.2;
}
.hero h1 span { color: #63b3ed; }
.hero p { color: #94a3b8; margin: 0; font-size: 0.95rem; line-height: 1.6; }
.hero-stats { display: flex; gap: 24px; margin-top: 20px; flex-wrap: wrap; }
.hero-stat { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #64748b; }
.hero-stat .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

/* Tabs */
button[data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: #64748b !important; font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important; font-weight: 500 !important;
    padding: 10px 20px !important; border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
}
button[data-baseweb="tab"]:hover { color: #cbd5e1 !important; background: rgba(99,179,237,0.06) !important; }
button[aria-selected="true"][data-baseweb="tab"] {
    color: #63b3ed !important; background: rgba(99,179,237,0.1) !important;
    border-bottom: 2px solid #63b3ed !important; font-weight: 600 !important;
}

/* Section titles */
.section-title {
    color: #e2e8f0; font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.4px;
    padding-bottom: 8px; margin-bottom: 16px;
    border-bottom: 1px solid rgba(99,179,237,0.2);
    display: flex; align-items: center; gap: 8px;
}
.section-title::before {
    content: ''; display: inline-block;
    width: 3px; height: 14px;
    background: linear-gradient(180deg, #63b3ed, #e94560);
    border-radius: 2px; flex-shrink: 0;
}

/* Metric cards */
.metric-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 12px; padding: 20px 16px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: rgba(99,179,237,0.3); }
.metric-card .label { color: #64748b; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; }
.metric-card .value { font-size: 2rem; font-weight: 800; margin: 8px 0 4px 0; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.metric-card .sub { color: #475569; font-size: 0.72rem; font-weight: 500; }

/* Result banners */
.result-banner {
    border-radius: 12px; padding: 18px 24px;
    text-align: center; font-weight: 700;
    font-size: 1.25rem; letter-spacing: 0.5px; margin-bottom: 20px;
}
.result-banner.theft {
    background: linear-gradient(135deg, rgba(185,28,28,0.25), rgba(220,38,38,0.2));
    border: 1px solid rgba(248,113,113,0.4); color: #fca5a5;
    box-shadow: 0 0 30px rgba(220,38,38,0.15);
}
.result-banner.medium {
    background: linear-gradient(135deg, rgba(180,83,9,0.25), rgba(217,119,6,0.2));
    border: 1px solid rgba(251,191,36,0.35); color: #fcd34d;
    box-shadow: 0 0 30px rgba(217,119,6,0.15);
}
.result-banner.normal {
    background: linear-gradient(135deg, rgba(6,78,59,0.25), rgba(16,185,129,0.2));
    border: 1px solid rgba(52,211,153,0.35); color: #6ee7b7;
    box-shadow: 0 0 30px rgba(16,185,129,0.12);
}

/* Feature tags */
.feature-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.feature-tag {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(99,179,237,0.08); color: #93c5fd;
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.8rem; font-weight: 500; transition: all 0.2s;
}
.feature-tag:hover { background: rgba(99,179,237,0.15); border-color: rgba(99,179,237,0.4); }

/* Info / Placeholder */
.info-box {
    background: rgba(15,23,42,0.6); border: 1px solid rgba(99,179,237,0.1);
    border-radius: 12px; padding: 16px 20px;
    color: #64748b; font-size: 0.88rem; line-height: 1.7;
}
.placeholder-box {
    background: rgba(15,23,42,0.5); border: 1px dashed rgba(99,179,237,0.2);
    border-radius: 12px; padding: 52px 24px; text-align: center;
}
.placeholder-icon { font-size: 3.2rem; margin-bottom: 14px; }
.placeholder-title { color: #e2e8f0; font-size: 1.05rem; font-weight: 600; margin-bottom: 6px; }
.placeholder-sub { color: #475569; font-size: 0.85rem; line-height: 1.6; }

/* Risk legend */
.risk-legend {
    background: rgba(15,23,42,0.6); border: 1px solid rgba(99,179,237,0.1);
    border-radius: 10px; padding: 14px 16px;
}
.risk-legend-title { color: #94a3b8; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.risk-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.82rem; color: #94a3b8; }
.risk-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #e94560 0%, #b91c1c 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 13px 0 !important;
    font-size: 0.92rem !important; font-weight: 700 !important;
    letter-spacing: 0.5px !important; width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(233,69,96,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(233,69,96,0.45) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #07080f; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }

hr { border-color: rgba(99,179,237,0.1) !important; }
.footer { text-align: center; color: #334155; font-size: 0.78rem; padding: 18px 0 8px 0; letter-spacing: 0.3px; }
</style>
"""
# Use st.html if available (Streamlit >= 1.31), otherwise fall back to st.markdown
try:
    st.html(_CSS)
except AttributeError:
    st.markdown(_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — defined at top level (OUTSIDE tabs) to avoid Streamlit rendering bug
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="padding: 0 4px; margin-bottom: 4px;">
    <div style="font-size:1.1rem; font-weight:800; color:#f8fafc; letter-spacing:-0.3px; margin-bottom:4px;">
        ⚡ Consumer Profile
    </div>
    <div style="font-size:0.78rem; color:#64748b;">
        Adjust features and run prediction
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section">📈 Consumption Stats</div>', unsafe_allow_html=True)
mean_c   = st.sidebar.number_input("Mean Daily Consumption (kWh)", min_value=0.0, value=7.0,   step=0.5)
std_c    = st.sidebar.number_input("Std of Consumption",           min_value=0.0, value=4.0,   step=0.5)
max_c    = st.sidebar.number_input("Max Daily Consumption (kWh)",  min_value=0.0, value=28.0,  step=1.0)
min_c    = st.sidebar.number_input("Min Daily Consumption (kWh)",  min_value=0.0, value=0.5,   step=0.1)

st.sidebar.markdown('<div class="sidebar-section">⚖️ Derived Ratios</div>', unsafe_allow_html=True)
cv_c     = st.sidebar.number_input("CV of Consumption",            min_value=0.0, value=1.46,  step=0.1)
peak_op  = st.sidebar.number_input("Peak / Off-Peak Ratio",        min_value=0.0, value=15.0,  step=1.0)
wk_ratio = st.sidebar.number_input("Weekend / Weekday Ratio",      min_value=0.0, value=1.3,   step=0.1)
mtp      = st.sidebar.number_input("Mean-to-Peak Ratio",           min_value=0.0, max_value=1.0, value=0.25, step=0.01)
lf       = st.sidebar.number_input("Load Factor",                  min_value=0.0, max_value=1.0, value=0.25, step=0.01)

st.sidebar.markdown('<div class="sidebar-section">🕐 Temporal Features</div>', unsafe_allow_html=True)
sc_count = st.sidebar.number_input("Sudden Change Count",          min_value=0.0, value=115.0, step=1.0)
roll_m   = st.sidebar.number_input("Rolling Mean 7d (kWh)",        min_value=0.0, value=6.5,   step=0.5)
roll_s   = st.sidebar.number_input("Rolling Std 7d",               min_value=0.0, value=1.3,   step=0.1)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.sidebar.button("⚡  Run Prediction", type="primary", use_container_width=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="risk-legend">
    <div class="risk-legend-title">Risk Levels</div>
    <div class="risk-row">
        <div class="risk-dot" style="background:#34d399"></div>
        <div><b style="color:#e2e8f0">Low</b> &nbsp;— probability &lt; 40%</div>
    </div>
    <div class="risk-row">
        <div class="risk-dot" style="background:#fb923c"></div>
        <div><b style="color:#e2e8f0">Medium</b> — 40% – 70%</div>
    </div>
    <div class="risk-row">
        <div class="risk-dot" style="background:#f87171"></div>
        <div><b style="color:#e2e8f0">High</b> &nbsp;— probability &gt; 70%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input data dict (built from sidebar widgets) ──────────────────────────────
input_data = {
    "mean_consumption":      mean_c,
    "std_consumption":       std_c,
    "max_consumption":       max_c,
    "min_consumption":       min_c,
    "cv_consumption":        cv_c,
    "peak_offpeak_ratio":    peak_op,
    "weekend_weekday_ratio": wk_ratio,
    "mean_to_peak_ratio":    mtp,
    "load_factor":           lf,
    "sudden_change_count":   sc_count,
    "rolling_mean_7d":       roll_m,
    "rolling_std_7d":        roll_s,
}

FRIENDLY = {
    "mean_consumption":      "Mean Consumption (kWh)",
    "std_consumption":       "Std of Consumption",
    "max_consumption":       "Max Consumption (kWh)",
    "min_consumption":       "Min Consumption (kWh)",
    "cv_consumption":        "CV of Consumption",
    "peak_offpeak_ratio":    "Peak / Off-Peak Ratio",
    "weekend_weekday_ratio": "Weekend / Weekday Ratio",
    "mean_to_peak_ratio":    "Mean-to-Peak Ratio",
    "load_factor":           "Load Factor",
    "sudden_change_count":   "Sudden Change Count",
    "rolling_mean_7d":       "Rolling Mean 7d (kWh)",
    "rolling_std_7d":        "Rolling Std 7d",
}

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ ElectroGuard AI</div>
    <h1>Electricity Theft <span>Detection</span> System</h1>
    <p>Explainable AI-powered detection using Ensemble Learning &amp; SHAP analysis
    &nbsp;·&nbsp; Trained on SGCC Dataset &nbsp;·&nbsp; B III M.Sc. AI &amp; ML</p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="dot" style="background:#34d399"></div>Real-time prediction</div>
        <div class="hero-stat"><div class="dot" style="background:#63b3ed"></div>Random Forest + XGBoost</div>
        <div class="hero-stat"><div class="dot" style="background:#a78bfa"></div>SHAP Explainability</div>
        <div class="hero-stat"><div class="dot" style="background:#fb923c"></div>SGCC Open Dataset</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Predict", "📊  EDA & Model Plots", "📋  Model Comparison"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1.08], gap="large")

    with col1:
        st.markdown('<div class="section-title">Input Feature Summary</div>', unsafe_allow_html=True)
        df_input = pd.DataFrame(
            [(FRIENDLY.get(k, k), round(v, 3)) for k, v in input_data.items()],
            columns=["Feature", "Value"]
        )
        st.dataframe(
            df_input,
            use_container_width=True,
            hide_index=True,
            height=430,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="medium"),
                "Value":   st.column_config.NumberColumn("Value",  format="%.3f", width="small"),
            }
        )

    with col2:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            with st.spinner("Analyzing consumer pattern …"):
                try:
                    response = requests.post(API_URL, json=input_data, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        prob   = result["probability"] * 100
                        level  = result["risk_level"]

                        if level == "High":
                            banner_cls, icon = "theft",  "🚨"
                        elif level == "Medium":
                            banner_cls, icon = "medium", "⚠️"
                        else:
                            banner_cls, icon = "normal", "✅"

                        st.markdown(f"""
                        <div class="result-banner {banner_cls}">
                            {icon} &nbsp; {result['prediction']}
                        </div>
                        """, unsafe_allow_html=True)

                        risk_color = {"High": "#f87171", "Medium": "#fb923c"}.get(level, "#34d399")

                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="label">Risk Score</div>
                                <div class="value" style="color:{risk_color}">{result['risk_score']}</div>
                                <div class="sub">out of 100</div>
                            </div>""", unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="label">Probability</div>
                                <div class="value" style="color:{risk_color}">{prob:.1f}%</div>
                                <div class="sub">theft likelihood</div>
                            </div>""", unsafe_allow_html=True)
                        with m3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="label">Risk Level</div>
                                <div class="value" style="color:{risk_color}; font-size:1.4rem; padding-top:6px">{level}</div>
                                <div class="sub">classification</div>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # ── Risk Gauge ────────────────────────────────────
                        fig, ax = plt.subplots(figsize=(6, 1.1))
                        fig.patch.set_facecolor("#0d1b2a")
                        ax.set_facecolor("#0d1b2a")
                        ax.barh([""], [40], color="#065f46", height=0.55, alpha=0.85)
                        ax.barh([""], [30], color="#92400e", height=0.55, left=40,  alpha=0.85)
                        ax.barh([""], [30], color="#7f1d1d", height=0.55, left=70,  alpha=0.85)

                        score = result["risk_score"]
                        ax.axvline(score, color="white",    linewidth=2.5, zorder=5)
                        ax.axvline(score, color=risk_color, linewidth=6,   alpha=0.25, zorder=4)

                        ha = "left" if score < 85 else "right"
                        offset = 1 if ha == "left" else -1
                        ax.text(score + offset, 0.52, str(score),
                                color="white", fontsize=9, fontweight="bold",
                                va="bottom", ha=ha, fontfamily="monospace")

                        for txt, x in [("LOW", 20), ("MED", 55), ("HIGH", 85)]:
                            ax.text(x, -0.52, txt, color="#334155",
                                    fontsize=7, ha="center", va="top", fontfamily="monospace")

                        ax.set_xlim(0, 100)
                        ax.set_ylim(-0.75, 0.75)
                        ax.set_xlabel("Risk Score (0 – 100)", color="#475569", fontsize=8)
                        ax.tick_params(colors="#334155", labelsize=8)
                        ax.set_yticks([])
                        for sp in ax.spines.values():
                            sp.set_edgecolor("#1e293b")
                        plt.tight_layout(pad=0.5)
                        st.pyplot(fig)
                        plt.close()

                        # ── Top Features ──────────────────────────────────
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-title">Top Contributing Features</div>', unsafe_allow_html=True)
                        tags = "".join([
                            f'<span class="feature-tag">🔑 {f}</span>'
                            for f in result["top_features"]
                        ])
                        st.markdown(f'<div class="feature-tags">{tags}</div>', unsafe_allow_html=True)

                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("⚠️  Cannot connect to API. Run `start_api.bat` first.")
        else:
            st.markdown("""
            <div class="placeholder-box">
                <div class="placeholder-icon">⚡</div>
                <div class="placeholder-title">Ready to Analyze</div>
                <div class="placeholder-sub">
                    Adjust consumer features in the sidebar<br>
                    then click <b style="color:#e2e8f0">⚡ Run Prediction</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & MODEL PLOTS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    figures_dir = "results/figures"

    PLOT_DESCRIPTIONS = {
        "01_class_distribution.png":        "Class Distribution — Normal vs Theft Consumers",
        "02_consumption_distribution.png":  "Mean Consumption Distribution by Class",
        "03_feature_boxplots.png":          "Feature Boxplots — Normal vs Theft Comparison",
        "04_correlation_heatmap.png":       "Feature Correlation Heatmap",
        "05_sudden_changes.png":            "Sudden Consumption Change Count Distribution",
        "06_cv_vs_mean.png":                "Coefficient of Variation vs Mean Consumption",
        "07_load_factor.png":               "Load Factor Distribution by Class",
        "08_feature_label_correlation.png": "Feature Correlation with Theft Label",
        "09_roc_curves.png":                "ROC Curves — All 3 Models",
        "10_pr_curves.png":                 "Precision-Recall Curves — All 3 Models",
        "11_model_comparison.png":          "Model Comparison Bar Chart",
        "12_shap_global_bar.png":           "SHAP Global Feature Importance",
        "13_shap_summary.png":              "SHAP Beeswarm Summary Plot",
        "14_shap_waterfall_theft.png":      "SHAP Waterfall — Individual Theft Explanation",
        "cm_logistic_regression.png":       "Confusion Matrix — Logistic Regression",
        "cm_random_forest.png":             "Confusion Matrix — Random Forest",
        "cm_xgboost.png":                   "Confusion Matrix — XGBoost",
    }

    PLOT_ICONS = {"EDA Plots": "📊", "Model Evaluation": "🎯", "SHAP Explainability": "🔬"}

    if os.path.exists(figures_dir):
        plot_files = sorted([f for f in os.listdir(figures_dir) if f.endswith(".png")])

        groups = {
            "EDA Plots":           [f for f in plot_files if f[:2].isdigit() and int(f[:2]) < 9],
            "Model Evaluation":    [f for f in plot_files if f.startswith(("09","10","11","cm"))],
            "SHAP Explainability": [f for f in plot_files if f.startswith("1") and
                                    not f.startswith("10") and not f.startswith("11")],
        }

        selected_group = st.radio(
            "Select Category:",
            list(groups.keys()),
            horizontal=True,
            format_func=lambda x: f"{PLOT_ICONS.get(x,'')}  {x}"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        files_in_group = groups[selected_group]
        if files_in_group:
            cols = st.columns(2, gap="large")
            for i, fname in enumerate(files_in_group):
                with cols[i % 2]:
                    desc = PLOT_DESCRIPTIONS.get(fname, fname.replace("_"," ").replace(".png","").title())
                    st.markdown(f'<div class="section-title">{desc}</div>', unsafe_allow_html=True)
                    st.image(os.path.join(figures_dir, fname), use_column_width=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="placeholder-box">
                <div class="placeholder-icon">📊</div>
                <div class="placeholder-title">No Plots in This Category</div>
                <div class="placeholder-sub">Run the pipeline to generate plots.</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-icon">🗂️</div>
            <div class="placeholder-title">No Plots Found</div>
            <div class="placeholder-sub">
                The <code style="color:#63b3ed">results/figures/</code> directory was not found.<br>
                Run the pipeline scripts first to generate EDA and model plots.
            </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    metrics_path     = "results/metrics/model_comparison.csv"
    error_path       = "results/metrics/error_analysis.csv"
    benchmark_path   = "results/metrics/benchmark_comparison.csv"

    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)

        # ── Split our models vs benchmarks ───────────────────────────────────
        our_models = df_metrics[~df_metrics["Model"].str.startswith("---")].copy()
        our_models = our_models[~our_models["Model"].str.contains("Zheng|Nagi|Hu et al", na=False)]

        metric_cols = [c for c in ["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]
                       if c in our_models.columns]

        # ── Section 1: Our Models ─────────────────────────────────────────────
        st.markdown('<div class="section-title">🤖 Our Models — Performance Comparison (All 6)</div>', unsafe_allow_html=True)

        optuna_badge = lambda m: f"⚡ {m}" if "Optuna" in m else m
        display_df = our_models.copy()
        display_df["Model"] = display_df["Model"].apply(optuna_badge)

        def highlight_best(s):
            try:
                is_max = pd.to_numeric(s, errors='coerce') == pd.to_numeric(s, errors='coerce').max()
                return ["background-color: rgba(6,95,70,0.55); color:#6ee7b7; font-weight:700" if v else "" for v in is_max]
            except:
                return ["" for _ in s]

        fmt_dict = {c: "{:.4f}" for c in metric_cols if c in display_df.columns}
        if "Threshold" in display_df.columns:
            fmt_dict["Threshold"] = "{:.2f}"

        styled = (
            display_df.style
            .apply(highlight_best, subset=metric_cols)
            .format(fmt_dict)
            .set_properties(**{
                "background-color": "rgba(15,23,42,0.7)",
                "color": "#cbd5e1",
                "font-family": "'JetBrains Mono', monospace",
                "font-size": "0.84rem",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown('<div style="font-size:0.75rem;color:#475569;margin-top:-8px">⚡ = Optuna Bayesian Optimization &nbsp;|&nbsp; Green cell = best in column</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 2: Best Model Cards ───────────────────────────────────────
        num_cols = [c for c in metric_cols if c in our_models.columns]
        safe_df  = our_models.copy()
        for c in num_cols:
            safe_df[c] = pd.to_numeric(safe_df[c], errors="coerce")

        best_f1  = safe_df.loc[safe_df["F1"].idxmax(),      "Model"] if "F1"      in safe_df else "N/A"
        best_rec = safe_df.loc[safe_df["Recall"].idxmax(),  "Model"] if "Recall"  in safe_df else "N/A"
        best_auc = safe_df.loc[safe_df["ROC-AUC"].idxmax(), "Model"] if "ROC-AUC" in safe_df else "N/A"

        st.markdown('<div class="section-title">🏅 Best Models by Metric</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, label, val, color in [
            (c1, "🏆 Best F1 Score",  best_f1,  "#34d399"),
            (c2, "🎯 Best Recall",    best_rec, "#63b3ed"),
            (c3, "📈 Best ROC-AUC",   best_auc, "#fb923c"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value" style="color:{color}; font-size:1rem; padding-top:8px">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 3: Benchmark Comparison ──────────────────────────────────
        if os.path.exists(benchmark_path):
            st.markdown('<div class="section-title">📚 Benchmark Comparison — vs Published Papers (SGCC Dataset)</div>', unsafe_allow_html=True)
            df_bench = pd.read_csv(benchmark_path)
            df_bench_clean = df_bench[~df_bench["Model"].str.startswith("---", na=False)].copy()

            def highlight_our(row):
                is_ours = not any(x in str(row.get("Model","")) for x in ["Zheng","Nagi","Hu et al"])
                color = "background-color: rgba(99,179,237,0.12)" if is_ours else ""
                return [color] * len(row)

            bench_metric_cols = [c for c in ["Accuracy","Precision","Recall","F1","ROC-AUC"] if c in df_bench_clean.columns]
            bench_fmt = {c: "{:.4f}" for c in bench_metric_cols}
            if "Threshold" in df_bench_clean.columns:
                bench_fmt["Threshold"] = "{}"

            bench_styled = (
                df_bench_clean.style
                .apply(highlight_our, axis=1)
                .format(bench_fmt, na_rep="—")
                .set_properties(**{
                    "background-color": "rgba(15,23,42,0.7)",
                    "color": "#cbd5e1",
                    "font-family": "'JetBrains Mono', monospace",
                    "font-size": "0.84rem",
                })
            )
            st.dataframe(bench_styled, use_container_width=True, hide_index=True)
            st.markdown('<div style="font-size:0.75rem;color:#475569;margin-top:-8px">🔵 Highlighted rows = our models &nbsp;|&nbsp; Benchmarks from Zheng 2018, Nagi 2019, Hu 2020</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 4: Error Analysis ─────────────────────────────────────────
        if os.path.exists(error_path):
            st.markdown('<div class="section-title">🔬 Rigorous Error Analysis — FP / FN / FPR / FNR</div>', unsafe_allow_html=True)
            df_err = pd.read_csv(error_path)

            err_styled = (
                df_err.style
                .format({
                    "False Positive Rate": "{:.4f}",
                    "False Negative Rate": "{:.4f}",
                    "Miss Rate (FNR)":     "{:.4f}",
                }, na_rep="—")
                .set_properties(**{
                    "background-color": "rgba(15,23,42,0.7)",
                    "color": "#cbd5e1",
                    "font-family": "'JetBrains Mono', monospace",
                    "font-size": "0.84rem",
                })
                .apply(lambda s: ["color:#f87171" if ("False Positive" in s.name or "False Negative" in s.name) else "" for _ in s], axis=0)
            )
            st.dataframe(err_styled, use_container_width=True, hide_index=True)

            err_img = "results/figures/17_error_analysis.png"
            if os.path.exists(err_img):
                st.markdown("<br>", unsafe_allow_html=True)
                st.image(err_img, use_column_width=True)
        else:
            st.markdown("""
            <div class="info-box">
                ℹ️ Error analysis not yet generated. Run <code style="color:#63b3ed">src/evaluate.py</code> to produce it.
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 5: Model Selection Rationale ──────────────────────────────
        st.markdown('<div class="section-title">💡 Model Selection Rationale</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <ul style="margin:0; padding-left:20px; line-height:2.1">
                <li><b style="color:#e2e8f0">LightGBM (Optuna) — Best F1 (0.3415) & ROC-AUC (0.7834)</b> — Bayesian HPO found optimal hyperparameters in 30 trials</li>
                <li><b style="color:#e2e8f0">Random Forest — Best baseline model</b> — Highest F1 (0.3254) among GridSearch models, SHAP-compatible</li>
                <li>Logistic Regression: High Recall but very low Precision → too many false alarms for field investigation</li>
                <li>SMOTE applied to training only — no data leakage into test/val sets</li>
                <li>Threshold tuning (0.20–0.70) further improves F1 beyond default 0.5 cutoff</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-icon">📋</div>
            <div class="placeholder-title">No Metrics Found</div>
            <div class="placeholder-sub">
                Run <code style="color:#63b3ed">src/evaluate.py</code> first to generate model comparison data.
            </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    ElectroGuard &nbsp;·&nbsp; Explainable Electricity Theft Detection
    &nbsp;|&nbsp; B III M.Sc. AI &amp; ML &nbsp;·&nbsp; Advanced Machine Learning Lab
</div>
""", unsafe_allow_html=True)
