"""
streamlit_app.py
────────────────────────────────────────────────────────────────────────────
Churn-LTV Engine — Streamlit Frontend Dashboard

Pages:
  1. 🏠 Executive Overview  — KPI cards + portfolio-level charts
  2. 🔍 Single Prediction   — Real-time churn + LTV for one customer
  3. 📦 Batch Prediction    — Upload CSV, get table + charts for all rows
  4. 🔎 Customer Lookup     — Find a stored customer by ID
  5. 📊 Model Performance   — Classification + regression metrics

Run:
  streamlit run streamlit_app.py
"""

import os
import io
import json
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────
load_dotenv()

API_BASE   = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY    = os.getenv("API_KEY", "")
HEADERS    = {"X-API-Key": API_KEY}

# Churn risk colour map
RISK_COLORS = {
    "Low"      : "#22c55e",
    "Medium"   : "#f59e0b",
    "High"     : "#f97316",
    "Critical" : "#ef4444",
}
LTV_COLORS = {
    "Low"     : "#94a3b8",
    "Medium"  : "#60a5fa",
    "High"    : "#a78bfa",
    "Premium" : "#f59e0b",
}
PRIORITY_COLORS = {
    "Low"      : "#22c55e",
    "Medium"   : "#60a5fa",
    "High"     : "#f97316",
    "Critical" : "#ef4444",
}


# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Churn-LTV Engine",
    page_icon   = "📡",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Dark sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 4px 0; }

  /* ── Main background ── */
  .stApp { background: #0f172a; color: #e2e8f0; }

  /* ── KPI cards ── */
  .kpi-card {
    background: linear-gradient(135deg, #1e293b, #1e3a5f);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(0,0,0,0.6); }
  .kpi-label { font-size: 0.8rem; font-weight: 500; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
  .kpi-value { font-size: 2.2rem; font-weight: 700; color: #f8fafc; line-height: 1.1; }
  .kpi-sub   { font-size: 0.82rem; color: #64748b; margin-top: 6px; }

  /* ── Risk badge ── */
  .badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600; letter-spacing: 0.03em;
  }

  /* ── Section headers ── */
  .section-header {
    font-size: 1.3rem; font-weight: 700; color: #f1f5f9;
    padding-bottom: 8px; border-bottom: 2px solid #334155;
    margin-bottom: 20px;
  }

  /* ── Result panel ── */
  .result-panel {
    background: #1e293b; border: 1px solid #334155; border-radius: 16px;
    padding: 28px; margin-top: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }

  /* ── Prediction value ── */
  .pred-value { font-size: 3rem; font-weight: 800; line-height: 1; }
  .pred-label { font-size: 0.85rem; color: #94a3b8; font-weight: 500; margin-top: 4px; }

  /* ── Streamlit overrides ── */
  [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 2rem !important; }
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
  h1, h2, h3 { color: #f1f5f9 !important; }
  .stSelectbox label, .stSlider label, .stNumberInput label { color: #cbd5e1 !important; }
  div[data-testid="stDataFrame"] { background: #1e293b; border-radius: 12px; }
  .stAlert { border-radius: 10px; }
  [data-testid="stForm"] { background: #1e293b; border-radius: 16px; padding: 8px; border: 1px solid #334155; }
  .stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 10px 28px !important;
    transition: opacity 0.2s ease !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }
  .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
  .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }
  .stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #6366f1) !important; }
  div[data-testid="stFileUploaderDropzone"] {
    background: #1e293b !important; border: 2px dashed #334155 !important; border-radius: 12px !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────
def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach API. Make sure the FastAPI server is running on `localhost:8000`.")
    return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", headers=HEADERS, json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach API.")
    return None


def api_post_file(path: str, file_bytes: bytes, filename: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}{path}",
            headers=HEADERS,
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach API.")
    return None


def gauge_chart(value: float, title: str, color: str, min_val=0, max_val=1, fmt=".1%") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value * (100 if fmt == ".1%" else 1),
        title = {"text": title, "font": {"color": "#94a3b8", "size": 14}},
        number= {"suffix": "%" if fmt == ".1%" else "", "font": {"color": "#f1f5f9", "size": 36}},
        gauge = {
            "axis"     : {"range": [0, 100 if fmt == ".1%" else max_val], "tickcolor": "#475569", "tickfont": {"color": "#64748b"}},
            "bar"      : {"color": color},
            "bgcolor"  : "#1e293b",
            "bordercolor": "#334155",
            "steps"    : [
                {"range": [0,   25 if fmt == ".1%" else max_val * 0.25], "color": "rgba(30, 58, 95, 0.13)"},
                {"range": [25 if fmt == ".1%" else max_val * 0.25,
                           50 if fmt == ".1%" else max_val * 0.5], "color": "rgba(30, 58, 95, 0.07)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": value * (100 if fmt == ".1%" else 1)},
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        margin=dict(t=40, b=10, l=20, r=20), height=230,
    )
    return fig


def dark_fig(fig: go.Figure, height=360) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font_color="#94a3b8",
        xaxis=dict(gridcolor="#1e3a5f", linecolor="#334155", tickfont_color="#64748b"),
        yaxis=dict(gridcolor="#1e3a5f", linecolor="#334155", tickfont_color="#64748b"),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        margin=dict(t=32, b=32, l=16, r=16),
        height=height,
    )
    return fig


def risk_badge(label: str) -> str:
    color = RISK_COLORS.get(label, "#64748b")
    return f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">{label}</span>'


def ltv_badge(label: str) -> str:
    color = LTV_COLORS.get(label, "#64748b")
    return f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">{label}</span>'


def priority_badge(label: str) -> str:
    color = PRIORITY_COLORS.get(label, "#64748b")
    return f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">🎯 {label} Priority</span>'


# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px 0;">
      <div style="font-size:2.2rem">📡</div>
      <div style="font-size:1.15rem;font-weight:700;color:#f1f5f9;margin-top:6px">Churn-LTV Engine</div>
      <div style="font-size:0.78rem;color:#475569;margin-top:2px">v1.1.0 · XGBoost Powered</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Executive Overview",
         "🔍  Single Prediction",
         "📦  Batch Prediction",
         "🔎  Customer Lookup",
         "📊  Model Performance"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.75rem;color:#475569">API: <code style="color:#60a5fa">{API_BASE}</code></div>',
        unsafe_allow_html=True,
    )

    # Live API health ping
    try:
        health = requests.get(f"{API_BASE}/", timeout=3).json()
        st.markdown('<div style="font-size:0.75rem;color:#22c55e;margin-top:4px">● API Online</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div style="font-size:0.75rem;color:#ef4444;margin-top:4px">● API Offline</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Overview
# ═══════════════════════════════════════════════════════════════════════════
if "Executive Overview" in page:
    st.markdown("## 🏠 Executive Overview")
    st.markdown("Real-time snapshot of churn risk and lifetime value across the entire customer portfolio.")
    st.markdown("---")

    summary = api_get("/metrics/summary")

    if summary:
        # ── KPI cards ──────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        cards = [
            (k1, "Total Customers",     f"{summary['total_customers']:,}",       ""),
            (k2, "Churned Customers",   f"{summary['churned_customers']:,}",      ""),
            (k3, "Churn Rate",          f"{summary['churn_rate_pct']:.1f}%",      "⚠️ Benchmark < 10%"),
            (k4, "Avg Predicted LTV",   f"${summary['avg_predicted_ltv']:,.0f}",  "per customer"),
            (k5, "Revenue at Risk",     f"${summary['total_revenue_at_risk']:,.0f}", "from churning customers"),
        ]
        for col, label, value, sub in cards:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── LTV Segment Distribution + Churn Gauge ─────────────────────
        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown('<div class="section-header">LTV Segment Distribution</div>', unsafe_allow_html=True)
            segs = summary.get("ltv_segments", {})
            seg_df = pd.DataFrame([
                {"Segment": k, "Customers": v, "Color": LTV_COLORS.get(k, "#64748b")}
                for k, v in segs.items()
            ]).sort_values("Customers", ascending=False)
            fig_seg = px.bar(
                seg_df, x="Segment", y="Customers",
                color="Segment",
                color_discrete_map=LTV_COLORS,
                text="Customers",
            )
            fig_seg.update_traces(textposition="outside", marker_line_width=0)
            fig_seg = dark_fig(fig_seg)
            st.plotly_chart(fig_seg, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">Portfolio Churn Rate</div>', unsafe_allow_html=True)
            churn_pct = summary["churn_rate_pct"] / 100
            color = "#ef4444" if churn_pct > 0.25 else "#f97316" if churn_pct > 0.15 else "#f59e0b"
            fig_gauge = gauge_chart(churn_pct, "Churn Rate", color)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Revenue at risk donut
            total_ltv      = summary["avg_predicted_ltv"] * summary["total_customers"]
            revenue_safe   = total_ltv - summary["total_revenue_at_risk"]
            fig_donut = go.Figure(go.Pie(
                labels=["Safe Revenue", "Revenue at Risk"],
                values=[max(revenue_safe, 0), summary["total_revenue_at_risk"]],
                hole=0.65,
                marker_colors=["#22c55e", "#ef4444"],
                textinfo="none",
            ))
            fig_donut.add_annotation(
                text=f"${summary['total_revenue_at_risk']/1e6:.1f}M<br><span style='font-size:11px'>at risk</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color="#f1f5f9"),
            )
            fig_donut = dark_fig(fig_donut, height=220)
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center", font_color="#94a3b8"),
                margin=dict(t=0, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.warning("Could not load portfolio summary. Make sure the API and database are running.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Single Prediction
# ═══════════════════════════════════════════════════════════════════════════
elif "Single Prediction" in page:
    st.markdown("## 🔍 Single Customer Prediction")
    st.markdown("Enter customer attributes below to get a real-time churn probability and lifetime value estimate.")
    st.markdown("---")

    with st.form("single_predict_form"):
        st.markdown("### 👤 Demographics & Account")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        gender         = r1c1.selectbox("Gender", ["Male", "Female"])
        senior_citizen = r1c2.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner        = r1c3.selectbox("Partner", ["Yes", "No"])
        dependents     = r1c4.selectbox("Dependents", ["Yes", "No"])

        st.markdown("### 📋 Contract & Billing")
        r2c1, r2c2, r2c3 = st.columns(3)
        tenure          = r2c1.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = r2c2.number_input("Monthly Charges ($)", 0.0, 200.0, 65.50, step=0.5)
        total_charges   = r2c3.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), step=10.0)

        r3c1, r3c2, r3c3 = st.columns(3)
        contract          = r3c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment_method    = r3c2.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless_billing = r3c3.selectbox("Paperless Billing", ["Yes", "No"])

        st.markdown("### 🌐 Services")
        s1, s2, s3, s4 = st.columns(4)
        phone_service    = s1.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines   = s2.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = s3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security  = s4.selectbox("Online Security", ["Yes", "No", "No internet service"])

        s5, s6, s7, s8 = st.columns(4)
        online_backup      = s5.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection  = s6.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support       = s7.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv       = s8.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

        s9, *_ = st.columns(4)
        streaming_movies = s9.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True)

    if submitted:
        payload = {
            "gender": gender, "senior_citizen": senior_citizen,
            "partner": partner, "dependents": dependents,
            "tenure": tenure, "monthly_charges": monthly_charges,
            "total_charges": total_charges, "contract": contract,
            "payment_method": payment_method, "paperless_billing": paperless_billing,
            "phone_service": phone_service, "multiple_lines": multiple_lines,
            "internet_service": internet_service, "online_security": online_security,
            "online_backup": online_backup, "device_protection": device_protection,
            "tech_support": tech_support, "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
        }

        with st.spinner("Running prediction..."):
            result = api_post("/predict/churn-and-ltv", payload)

        if result:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)

            col_a, col_b, col_c, col_d = st.columns(4)

            churn_prob = result["churn_probability"]
            churn_color = RISK_COLORS.get(result["churn_risk_label"], "#64748b")
            ltv_color   = LTV_COLORS.get(result["ltv_segment"], "#64748b")

            with col_a:
                st.markdown(f"""
                <div style="text-align:center">
                  <div class="pred-value" style="color:{churn_color}">{churn_prob*100:.1f}%</div>
                  <div class="pred-label">Churn Probability</div>
                  <div style="margin-top:10px">{risk_badge(result["churn_risk_label"])}</div>
                </div>""", unsafe_allow_html=True)

            with col_b:
                outcome = "Will Churn" if result["churn_prediction"] else "Will Stay"
                outcome_color = "#ef4444" if result["churn_prediction"] else "#22c55e"
                outcome_icon  = "⚠️" if result["churn_prediction"] else "✅"
                st.markdown(f"""
                <div style="text-align:center">
                  <div class="pred-value" style="color:{outcome_color}">{outcome_icon}</div>
                  <div class="pred-label">Predicted Outcome</div>
                  <div style="margin-top:10px;font-size:1rem;font-weight:600;color:{outcome_color}">{outcome}</div>
                </div>""", unsafe_allow_html=True)

            with col_c:
                st.markdown(f"""
                <div style="text-align:center">
                  <div class="pred-value" style="color:{ltv_color}">${result["ltv_predicted"]:,.0f}</div>
                  <div class="pred-label">Predicted LTV</div>
                  <div style="margin-top:10px">{ltv_badge(result["ltv_segment"])}</div>
                </div>""", unsafe_allow_html=True)

            with col_d:
                priority_color = PRIORITY_COLORS.get(result["priority_score"], "#64748b")
                st.markdown(f"""
                <div style="text-align:center">
                  <div class="pred-value" style="color:{priority_color}">${result["revenue_at_risk"]:,.0f}</div>
                  <div class="pred-label">Revenue at Risk</div>
                  <div style="margin-top:10px">{priority_badge(result["priority_score"])}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Churn probability gauge
            col_gauge, col_breakdown = st.columns([1, 1.5])
            with col_gauge:
                fig_g = gauge_chart(churn_prob, "Churn Probability", churn_color)
                st.plotly_chart(fig_g, use_container_width=True)

            with col_breakdown:
                st.markdown("#### 📋 Prediction Breakdown")
                items = [
                    ("Churn Probability",  f"{churn_prob*100:.2f}%"),
                    ("Risk Label",         result["churn_risk_label"]),
                    ("Predicted LTV",      f"${result['ltv_predicted']:,.2f}"),
                    ("LTV Segment",        result["ltv_segment"]),
                    ("Revenue at Risk",    f"${result['revenue_at_risk']:,.2f}"),
                    ("Retention Priority", result["priority_score"]),
                    ("Model Used",         result.get("model_used", "XGBoost")),
                ]
                for label, val in items:
                    c1, c2 = st.columns([1, 1])
                    c1.markdown(f"<span style='color:#94a3b8;font-size:0.85rem'>{label}</span>", unsafe_allow_html=True)
                    c2.markdown(f"<span style='color:#f1f5f9;font-weight:600;font-size:0.85rem'>{val}</span>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════
elif "Batch Prediction" in page:
    st.markdown("## 📦 Batch Prediction")
    st.markdown("Upload a CSV file to get churn + LTV predictions for all customers at once.")
    st.markdown("---")

    st.info("**Required columns:** `tenure`, `monthly_charges`, `total_charges`, `contract`, `internet_service`, `payment_method`  \nAll other columns are optional (defaults applied if missing).")

    # Sample CSV download
    sample_csv = """\
tenure,monthly_charges,total_charges,contract,internet_service,payment_method
12,65.5,786.0,Month-to-month,Fiber optic,Electronic check
36,45.0,1620.0,One year,DSL,Bank transfer (automatic)
60,20.0,1200.0,Two year,No,Mailed check
3,89.5,268.5,Month-to-month,Fiber optic,Electronic check
24,55.0,1320.0,One year,DSL,Credit card (automatic)
"""
    st.download_button(
        "⬇️ Download Sample CSV", sample_csv,
        file_name="sample_batch.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        uploaded.seek(0)
        st.markdown(f"**Preview** — {len(df_preview)} rows × {len(df_preview.columns)} columns")
        st.dataframe(df_preview.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=False):
            with st.spinner(f"Processing {len(df_preview)} customers…"):
                result = api_post_file("/predict/batch/churn", uploaded.read(), uploaded.name)

            if result:
                st.markdown("---")

                # ── Summary KPIs ───────────────────────────────────────
                b1, b2, b3, b4 = st.columns(4)
                batch_cards = [
                    (b1, "Total Customers",   f"{result['total_customers']:,}", ""),
                    (b2, "Predicted Churners",f"{result['churners_predicted']:,}", ""),
                    (b3, "Batch Churn Rate",  f"{result['churn_rate']*100:.1f}%", ""),
                    (b4, "Total Revenue Risk",f"${result['total_revenue_at_risk']:,.0f}", ""),
                ]
                for col, label, value, sub in batch_cards:
                    with col:
                        st.markdown(f"""
                        <div class="kpi-card">
                          <div class="kpi-label">{label}</div>
                          <div class="kpi-value">{value}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                rows_df = pd.DataFrame([r.dict() if hasattr(r, "dict") else r for r in result["results"]])

                # ── Charts ────────────────────────────────────────────
                tab1, tab2, tab3 = st.tabs(["📊 Risk Distribution", "💰 LTV Segments", "📋 Full Results"])

                with tab1:
                    risk_counts = rows_df["churn_risk_label"].value_counts().reset_index()
                    risk_counts.columns = ["Risk", "Count"]
                    fig_risk = px.pie(risk_counts, names="Risk", values="Count",
                                     color="Risk", color_discrete_map=RISK_COLORS, hole=0.5)
                    fig_risk = dark_fig(fig_risk, 320)
                    st.plotly_chart(fig_risk, use_container_width=True)

                with tab2:
                    ltv_counts = rows_df["ltv_segment"].value_counts().reset_index()
                    ltv_counts.columns = ["Segment", "Count"]
                    fig_ltv = px.bar(ltv_counts, x="Segment", y="Count",
                                    color="Segment", color_discrete_map=LTV_COLORS, text="Count")
                    fig_ltv.update_traces(textposition="outside", marker_line_width=0)
                    fig_ltv = dark_fig(fig_ltv, 320)
                    st.plotly_chart(fig_ltv, use_container_width=True)

                with tab3:
                    display_df = rows_df.copy()
                    display_df["churn_probability"] = (display_df["churn_probability"] * 100).round(2).astype(str) + "%"
                    display_df["ltv_predicted"]     = "$" + display_df["ltv_predicted"].round(2).astype(str)
                    display_df["revenue_at_risk"]   = "$" + display_df["revenue_at_risk"].round(2).astype(str)
                    st.dataframe(display_df, use_container_width=True, height=400)

                    # Download
                    csv_out = rows_df.to_csv(index=False)
                    st.download_button("⬇️ Download Results CSV", csv_out,
                                       file_name="batch_predictions.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Customer Lookup
# ═══════════════════════════════════════════════════════════════════════════
elif "Customer Lookup" in page:
    st.markdown("## 🔎 Customer Lookup")
    st.markdown("Find a stored customer record by their ID from the database.")
    st.markdown("---")

    col_in, col_btn = st.columns([3, 1])
    customer_id = col_in.text_input("Customer ID", placeholder="e.g. 7590-VHVEG", label_visibility="collapsed")
    lookup_btn  = col_btn.button("🔍 Lookup", use_container_width=True)

    if lookup_btn and customer_id.strip():
        with st.spinner("Fetching customer record..."):
            data = api_get(f"/customers/{customer_id.strip()}")

        if data:
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown(f"### Customer `{data['customer_id']}`")

            c1, c2, c3, c4 = st.columns(4)

            ltv_pred = data.get("ltv_predicted", 0) or 0
            ltv_seg  = data.get("ltv_segment", "—")
            churn_status = data.get("churn_status", "—")
            churn_color  = "#ef4444" if churn_status == "Yes" else "#22c55e"

            with c1:
                st.markdown(f"""
                <div style="text-align:center">
                  <div class="pred-value" style="color:{churn_color}">{'⚠️' if churn_status=='Yes' else '✅'}</div>
                  <div class="pred-label">Churn Status</div>
                  <div style="margin-top:8px;font-weight:600;color:{churn_color}">{churn_status}</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                ltv_color = LTV_COLORS.get(ltv_seg, "#64748b")
                st.markdown(f"""
                <div style="text-align:center">
                  <div class="pred-value" style="color:{ltv_color}">${ltv_pred:,.0f}</div>
                  <div class="pred-label">Predicted LTV</div>
                  <div style="margin-top:8px">{ltv_badge(ltv_seg)}</div>
                </div>""", unsafe_allow_html=True)

            with c3:
                tenure = data.get("tenure", 0) or 0
                st.markdown(f"""
                <div style="text-align:center">
                  <div class="pred-value" style="color:#60a5fa">{tenure}</div>
                  <div class="pred-label">Tenure (months)</div>
                </div>""", unsafe_allow_html=True)

            with c4:
                monthly = data.get("monthly_charges", 0) or 0
                st.markdown(f"""
                <div style="text-align:center">
                  <div class="pred-value" style="color:#a78bfa">${monthly:.2f}</div>
                  <div class="pred-label">Monthly Charges</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Full Record")

            ltv_hist = data.get("ltv_historical", 0) or 0
            ltv_proj = data.get("ltv_projected", 0) or 0
            record_items = [
                ("Contract",           data.get("contract", "—")),
                ("Monthly Charges",    f"${monthly:.2f}"),
                ("Historical LTV",     f"${ltv_hist:,.2f}"),
                ("Projected LTV",      f"${ltv_proj:,.2f}"),
                ("Predicted LTV",      f"${ltv_pred:,.2f}"),
                ("LTV Segment",        ltv_seg),
                ("Churn Status",       churn_status),
            ]
            cols = st.columns(2)
            for i, (label, val) in enumerate(record_items):
                with cols[i % 2]:
                    lc, rc = st.columns([1, 1])
                    lc.markdown(f"<span style='color:#94a3b8;font-size:0.85rem'>{label}</span>", unsafe_allow_html=True)
                    rc.markdown(f"<span style='color:#f1f5f9;font-weight:600;font-size:0.85rem'>{val}</span>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    elif lookup_btn:
        st.warning("Please enter a Customer ID.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════
elif "Model Performance" in page:
    st.markdown("## 📊 Model Performance")
    st.markdown("Evaluation metrics for all trained churn classification and LTV regression models.")
    st.markdown("---")

    metrics = api_get("/metrics/models")

    if metrics:
        tab_churn, tab_ltv = st.tabs(["🎯 Churn Classification", "💰 LTV Regression"])

        # ── Churn metrics ──────────────────────────────────────────────
        with tab_churn:
            churn_data = metrics.get("churn_model_metrics", [])
            if isinstance(churn_data, list) and churn_data:
                df_c = pd.DataFrame(churn_data)
                st.dataframe(df_c, use_container_width=True)

                # Best model highlight
                if "f1_score" in df_c.columns or "f1" in df_c.columns:
                    f1_col = "f1_score" if "f1_score" in df_c.columns else "f1"
                    best_row = df_c.loc[df_c[f1_col].idxmax()]
                    st.success(f"🏆 Best Model: **{best_row.get('model_name', best_row.get('model',''))}** with F1 = {best_row[f1_col]:.4f}")

                # Bar comparison
                metric_cols = [c for c in df_c.columns if c not in ("model_name", "model", "id")]
                model_col   = "model_name" if "model_name" in df_c.columns else "model"
                if model_col in df_c.columns and metric_cols:
                    fig_bar = px.bar(
                        df_c.melt(id_vars=model_col, value_vars=metric_cols,
                                  var_name="Metric", value_name="Score"),
                        x="Metric", y="Score", color=model_col,
                        barmode="group", text_auto=".3f",
                        color_discrete_sequence=["#3b82f6", "#a78bfa", "#22c55e"],
                    )
                    fig_bar.update_traces(textposition="outside", marker_line_width=0)
                    fig_bar = dark_fig(fig_bar, 360)
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info(str(churn_data))

        # ── LTV metrics ────────────────────────────────────────────────
        with tab_ltv:
            ltv_data = metrics.get("ltv_model_metrics", [])
            if isinstance(ltv_data, list) and ltv_data:
                df_l = pd.DataFrame(ltv_data)
                st.dataframe(df_l, use_container_width=True)

                # R² bar chart
                r2_col = next((c for c in df_l.columns if "r2" in c.lower()), None)
                model_col = "model_name" if "model_name" in df_l.columns else ("model" if "model" in df_l.columns else None)
                if r2_col and model_col:
                    best_row = df_l.loc[df_l[r2_col].idxmax()]
                    st.success(f"🏆 Best LTV Model: **{best_row.get(model_col, '')}** with R² = {best_row[r2_col]:.4f}")

                    metric_cols = [c for c in df_l.columns if c not in (model_col, "id")]
                    fig_ltv_bar = px.bar(
                        df_l.melt(id_vars=model_col, value_vars=metric_cols,
                                  var_name="Metric", value_name="Score"),
                        x="Metric", y="Score", color=model_col,
                        barmode="group", text_auto=".3f",
                        color_discrete_sequence=["#3b82f6", "#a78bfa", "#22c55e"],
                    )
                    fig_ltv_bar.update_traces(textposition="outside", marker_line_width=0)
                    fig_ltv_bar = dark_fig(fig_ltv_bar, 360)
                    st.plotly_chart(fig_ltv_bar, use_container_width=True)
            else:
                st.info(str(ltv_data))
    else:
        st.warning("Could not load model metrics from API.")
