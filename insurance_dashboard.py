"""
Insurance Analytics Dashboard — Streamlit App
Run with: streamlit run insurance_dashboard.py
Requirements: pip install streamlit pandas numpy scikit-learn xgboost mlxtend plotly seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Analytics Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #252840 100%);
    border: 1px solid #2e3158; border-radius: 12px;
    padding: 18px 20px; text-align: center; margin-bottom: 10px;
  }
  .metric-value { font-size: 1.9rem; font-weight: 700; color: #7c9ef8; font-family: 'JetBrains Mono', monospace; }
  .metric-label { font-size: 0.75rem; color: #8b9cc8; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }
  .section-header {
    font-size: 1.3rem; font-weight: 700; color: #e8eaf6;
    border-left: 4px solid #7c9ef8; padding-left: 12px; margin: 24px 0 14px 0;
  }
  .bias-card {
    background: #151829; border: 1px solid #2a2f52; border-radius: 10px;
    padding: 18px 20px; margin-bottom: 14px;
  }
  .bias-title { font-size: 1rem; font-weight: 700; color: #f472b6; margin-bottom: 6px; }
  .bias-body  { font-size: 0.88rem; color: #9aa5c8; line-height: 1.7; }
  .severity-high   { color: #f87171; font-weight: 700; }
  .severity-medium { color: #fbbf24; font-weight: 700; }
  .severity-low    { color: #4ade80; font-weight: 700; }
  .claim-high   { background:#7f1d1d22; border:1px solid #dc2626; color:#f87171; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
  .claim-medium { background:#78350f22; border:1px solid #d97706; color:#fbbf24; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
  .claim-low    { background:#14532d22; border:1px solid #16a34a; color:#4ade80;  padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:700; }
  div[data-testid="stSidebar"] { background: #10121f; border-right: 1px solid #1e2235; }
  h1 { color: #e8eaf6 !important; }
  h2, h3 { color: #c4cbe8 !important; }
  p, li { color: #9aa5c8 !important; }
  .stSelectbox label, .stSlider label, .stMultiSelect label { color: #9aa5c8 !important; }
  code { font-family: 'JetBrains Mono', monospace; background: #1e2235; color: #7c9ef8; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

PALETTE    = ["#7c9ef8","#4ade80","#f472b6","#fb923c","#a78bfa","#34d399","#fbbf24","#60a5fa"]
PLOTLY_BG  = "#0d0f1a"
GRID_COLOR = "#1e2235"

def dark_fig(fig):
    fig.update_layout(
        plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_BG,
        font=dict(color="#9aa5c8", family="DM Sans"),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(t=45, b=40, l=40, r=20),
        legend=dict(bgcolor="#151829", bordercolor="#2a2f52", borderwidth=1),
    )
    return fig

# ── Load & enrich data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(path="InsuranceLR.csv"):
    df = pd.read_csv(path)
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    # Derived features
    df["smoker_bin"] = (df["smoker"] == "yes").astype(int)
    df["sex_bin"]    = (df["sex"] == "male").astype(int)
    df["bmi_category"] = pd.cut(df["bmi"],
        bins=[0,18.5,25,30,np.inf],
        labels=["Underweight","Normal","Overweight","Obese"])
    df["age_group"] = pd.cut(df["age"],
        bins=[17,25,35,45,55,65],
        labels=["18-25","26-35","36-45","46-55","56-64"])

    # ── Claim tiers ──────────────────────────────────────────────────────────
    # Charges split into 3 natural tiers based on distribution:
    # Low  (<$5,000)  → baseline / minimal claim
    # Mid  ($5k-$16k) → moderate claim
    # High (>$16k)    → significant / major claim
    # The ~$16k boundary aligns with the 75th percentile and separates
    # the upper cluster (driven heavily by smokers) from the middle band.
    df["claim_tier"] = pd.cut(
        df["charges"],
        bins=[0, 5000, 16000, np.inf],
        labels=["Low (<$5k)", "Moderate ($5k–$16k)", "High (>$16k)"]
    )
    df["claimed"] = df["charges"] > 5000  # True = claim beyond baseline premium

    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
    df = pd.concat([df, region_dummies], axis=1)
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Insurance Hub")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        if "index" in df_raw.columns:
            df_raw = df_raw.drop(columns=["index"])
        df_raw["smoker_bin"] = (df_raw["smoker"] == "yes").astype(int)
        df_raw["sex_bin"]    = (df_raw["sex"] == "male").astype(int)
        df_raw["bmi_category"] = pd.cut(df_raw["bmi"], bins=[0,18.5,25,30,np.inf],
            labels=["Underweight","Normal","Overweight","Obese"])
        df_raw["age_group"] = pd.cut(df_raw["age"], bins=[17,25,35,45,55,65],
            labels=["18-25","26-35","36-45","46-55","56-64"])
        df_raw["claim_tier"] = pd.cut(df_raw["charges"], bins=[0,5000,16000,np.inf],
            labels=["Low (<$5k)","Moderate ($5k–$16k)","High (>$16k)"])
        df_raw["claimed"] = df_raw["charges"] > 5000
        rd = pd.get_dummies(df_raw["region"], prefix="region", drop_first=True)
        df = pd.concat([df_raw, rd], axis=1)
        st.success("Custom dataset loaded!")
    else:
        df = load_data()

    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview",
        "📋 Claims Analysis",
        "🔍 Exploratory Analysis",
        "⚖️ Bias & Fairness",
        "🤖 ML — Prediction",
        "🧩 K-Means Clustering",
        "🛒 Apriori Rules",
        "⚡ XGBoost Analysis",
    ])
    st.markdown("---")
    st.markdown("### Filters")
    sel_regions = st.multiselect("Region", df["region"].unique().tolist(), default=df["region"].unique().tolist())
    sel_smoker  = st.multiselect("Smoker", ["yes","no"], default=["yes","no"])
    age_range   = st.slider("Age range", int(df["age"].min()), int(df["age"].max()), (18,64))
    df_f = df[
        df["region"].isin(sel_regions) &
        df["smoker"].isin(sel_smoker) &
        df["age"].between(*age_range)
    ]
    st.caption(f"Showing **{len(df_f):,}** records")

CLAIM_COLORS = {
    "Low (<$5k)":           PALETTE[1],
    "Moderate ($5k–$16k)":  PALETTE[6],
    "High (>$16k)":         PALETTE[3],
}

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Insurance Analytics Hub")
    st.markdown("Comprehensive analytics across claims, demographics, bias, and predictive modelling.")

    high   = df_f[df_f["claim_tier"] == "High (>$16k)"]
    mid    = df_f[df_f["claim_tier"] == "Moderate ($5k–$16k)"]
    low    = df_f[df_f["claim_tier"] == "Low (<$5k)"]

    cols = st.columns(6)
    metrics = [
        ("Total Records",    f"{len(df_f):,}"),
        ("High Claims",      f"{len(high):,}"),
        ("Moderate Claims",  f"{len(mid):,}"),
        ("Low/No Claims",    f"{len(low):,}"),
        ("Avg Charge",       f"${df_f['charges'].mean():,.0f}"),
        ("Smoker Rate",      f"{df_f['smoker_bin'].mean()*100:.1f}%"),
    ]
    for col, (label, val) in zip(cols, metrics):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Claim Tier Breakdown</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        tier_cnt = df_f["claim_tier"].value_counts().reset_index()
        tier_cnt.columns = ["Tier","Count"]
        fig = px.pie(tier_cnt, names="Tier", values="Count",
                     color="Tier", color_discrete_map=CLAIM_COLORS,
                     title="Distribution of Claim Tiers", hole=0.45)
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df_f, x="charges", color="claim_tier", nbins=80,
                           color_discrete_map=CLAIM_COLORS, barmode="overlay",
                           title="Charge Distribution by Claim Tier", opacity=0.8)
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Key Drivers</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df_f, x="smoker", y="charges", color="claim_tier",
                     color_discrete_map=CLAIM_COLORS,
                     title="Charges by Smoking Status & Claim Tier")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df_f, x="age", y="charges", color="claim_tier",
                         color_discrete_map=CLAIM_COLORS, size="bmi", opacity=0.6,
                         title="Age vs Charges (size = BMI)")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Correlation
    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["age","bmi","children","charges","smoker_bin","sex_bin"]
    corr = df_f[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="Blues", zmin=-1, zmax=1,
        text=np.round(corr.values,2), texttemplate="%{text}", showscale=True
    ))
    fig.update_layout(title="Correlation Matrix", plot_bgcolor=PLOTLY_BG,
                      paper_bgcolor=PLOTLY_BG, font=dict(color="#9aa5c8"), margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — CLAIMS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📋 Claims Analysis":
    st.markdown("# 📋 Claims Analysis")
    st.markdown("""
> **How claim tiers are defined** — Insurance charges in this dataset represent annual medical costs billed.
> There is no explicit "claimed / not claimed" flag, so claim status is inferred from charge level:
> - 🟢 **Low (<$5,000)** — likely baseline premium only, no significant claim filed  
> - 🟡 **Moderate ($5k–$16k)** — moderate medical claim (aligns with 25th–75th percentile)  
> - 🔴 **High (>$16k)** — major claim, above 75th percentile; upper cluster driven by smokers + age
    """)

    high = df_f[df_f["claim_tier"]=="High (>$16k)"]
    mid  = df_f[df_f["claim_tier"]=="Moderate ($5k–$16k)"]
    low  = df_f[df_f["claim_tier"]=="Low (<$5k)"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 High Claims", "🟡 Moderate Claims", "🟢 Low Claims", "📊 Comparisons"
    ])

    def tier_profile(tier_df, tier_name, color):
        cols = st.columns(4)
        m = [
            ("Count",        f"{len(tier_df):,}"),
            ("Avg Charge",   f"${tier_df['charges'].mean():,.0f}"),
            ("Smoker Rate",  f"{tier_df['smoker_bin'].mean()*100:.1f}%"),
            ("Avg Age",      f"{tier_df['age'].mean():.1f} yrs"),
        ]
        for col, (lbl, val) in zip(cols, m):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:{color}">{val}</div>
              <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            rc = tier_df["region"].value_counts().reset_index()
            fig = px.bar(rc, x="region", y="count", color="region",
                         color_discrete_sequence=PALETTE, title=f"{tier_name} — Claims by Region")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(tier_df, x="age", color="sex",
                               color_discrete_map={"male":PALETTE[0],"female":PALETTE[2]},
                               barmode="overlay", nbins=25,
                               title=f"{tier_name} — Age Distribution by Sex")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            bmi_cnt = tier_df["bmi_category"].value_counts().reset_index()
            fig = px.pie(bmi_cnt, names="bmi_category", values="count",
                         color_discrete_sequence=PALETTE, hole=0.4,
                         title=f"{tier_name} — BMI Category Mix")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            sm = tier_df["smoker"].value_counts().reset_index()
            fig = px.pie(sm, names="smoker", values="count", hole=0.4,
                         color_discrete_map={"yes":PALETTE[3],"no":PALETTE[0]},
                         title=f"{tier_name} — Smoker Split")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 Raw records"):
            st.dataframe(tier_df[["age","sex","bmi","children","smoker","region","charges","claim_tier"]]
                         .sort_values("charges", ascending=False).reset_index(drop=True))

    with tab1:
        tier_profile(high, "High Claims", PALETTE[3])
    with tab2:
        tier_profile(mid,  "Moderate Claims", PALETTE[6])
    with tab3:
        tier_profile(low,  "Low Claims", PALETTE[1])

    with tab4:
        st.markdown('<div class="section-header">Side-by-Side Comparisons</div>', unsafe_allow_html=True)

        # Smoker rate per tier
        smoker_rates = df_f.groupby("claim_tier", observed=True)["smoker_bin"].mean().reset_index()
        smoker_rates.columns = ["Tier","Smoker Rate"]
        smoker_rates["Smoker Rate"] *= 100
        fig = px.bar(smoker_rates, x="Tier", y="Smoker Rate", color="Tier",
                     color_discrete_map=CLAIM_COLORS, title="Smoker Rate by Claim Tier (%)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            avg_age = df_f.groupby("claim_tier", observed=True)["age"].mean().reset_index()
            fig = px.bar(avg_age, x="claim_tier", y="age", color="claim_tier",
                         color_discrete_map=CLAIM_COLORS, title="Average Age by Claim Tier")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            avg_bmi = df_f.groupby("claim_tier", observed=True)["bmi"].mean().reset_index()
            fig = px.bar(avg_bmi, x="claim_tier", y="bmi", color="claim_tier",
                         color_discrete_map=CLAIM_COLORS, title="Average BMI by Claim Tier")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # Stacked region bar
        region_tier = df_f.groupby(["region","claim_tier"], observed=True).size().reset_index(name="count")
        fig = px.bar(region_tier, x="region", y="count", color="claim_tier",
                     color_discrete_map=CLAIM_COLORS, barmode="stack",
                     title="Claim Tier Volume by Region")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # Heatmap: age group × region → avg charges
        pivot = df_f.groupby(["age_group","region"], observed=True)["charges"].mean().unstack()
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
            colorscale="Blues", text=np.round(pivot.values,0),
            texttemplate="$%{text:,.0f}", showscale=True
        ))
        fig.update_layout(title="Avg Charges Heatmap: Age Group × Region",
                          plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_BG,
                          font=dict(color="#9aa5c8"))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.markdown("# 🔍 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Demographics","BMI Analysis","Charges Deep Dive","Pairplot"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            sex_cnt = df_f["sex"].value_counts().reset_index()
            fig = px.pie(sex_cnt, names="sex", values="count",
                         color_discrete_sequence=PALETTE[:2], title="Sex Distribution")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            rc = df_f["region"].value_counts().reset_index()
            fig = px.bar(rc, x="region", y="count", color="region",
                         color_discrete_sequence=PALETTE, title="Records per Region")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_f, x="age", color="sex",
                               color_discrete_map={"male":PALETTE[0],"female":PALETTE[2]},
                               barmode="overlay", nbins=30, title="Age Distribution by Sex")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            cc = df_f["children"].value_counts().reset_index().sort_values("children")
            fig = px.bar(cc, x="children", y="count",
                         color_discrete_sequence=[PALETTE[1]], title="Number of Children")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_f, x="bmi", color="bmi_category",
                               color_discrete_sequence=PALETTE, nbins=50, title="BMI Distribution")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            avg_bmi = df_f.groupby(["age_group","smoker"], observed=True)["bmi"].mean().reset_index()
            fig = px.bar(avg_bmi, x="age_group", y="bmi", color="smoker", barmode="group",
                         color_discrete_map={"yes":PALETTE[3],"no":PALETTE[0]},
                         title="Avg BMI by Age Group & Smoker Status")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        fig = px.box(df_f, x="bmi_category", y="charges", color="smoker",
                     color_discrete_map={"yes":PALETTE[3],"no":PALETTE[0]},
                     title="Charges by BMI Category & Smoking Status")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.violin(df_f, x="region", y="charges", color="region",
                            color_discrete_sequence=PALETTE, box=True,
                            title="Charge Distribution per Region")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            pivot = df_f.groupby(["age_group","region"], observed=True)["charges"].mean().reset_index()
            fig = px.line(pivot, x="age_group", y="charges", color="region",
                          color_discrete_sequence=PALETTE, markers=True,
                          title="Avg Charges by Age Group & Region")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.scatter_matrix(df_f, dimensions=["age","bmi","children","charges"],
                                color="claim_tier", color_discrete_map=CLAIM_COLORS,
                                title="Pairplot — coloured by Claim Tier")
        fig.update_traces(marker=dict(size=3, opacity=0.6))
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — BIAS & FAIRNESS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Bias & Fairness":
    st.markdown("# ⚖️ Bias & Fairness Analysis")
    st.markdown("""
This page examines **statistical, representational, and algorithmic biases** present in the dataset
and what they mean for any models trained on it. Understanding bias is critical before deploying
insurance analytics in production.
    """)

    # ── 1. Data-level bias summary cards ──────────────────────────────────────
    st.markdown('<div class="section-header">1. Dataset-Level Biases</div>', unsafe_allow_html=True)

    biases = [
        {
            "title": "🚬 Smoker Dominance Bias",
            "severity": "HIGH",
            "body": (
                f"Smokers ({df['smoker_bin'].mean()*100:.1f}% of dataset) account for a "
                f"disproportionate share of high claims. Their mean charge (${df[df.smoker=='yes']['charges'].mean():,.0f}) "
                f"is {df[df.smoker=='yes']['charges'].mean()/df[df.smoker=='no']['charges'].mean():.1f}× "
                f"that of non-smokers (${df[df.smoker=='no']['charges'].mean():,.0f}). "
                "Any predictive model will learn to over-weight smoking status, potentially creating "
                "a discriminatory feedback loop where ex-smokers or incorrectly labelled individuals "
                "receive unfairly high premiums."
            )
        },
        {
            "title": "👨‍👩‍ Sex / Gender Bias",
            "severity": "MEDIUM",
            "body": (
                f"The dataset is {df[df.sex=='male'].shape[0]/len(df)*100:.1f}% male and "
                f"{df[df.sex=='female'].shape[0]/len(df)*100:.1f}% female — near balanced. "
                f"However, male mean charges (${df[df.sex=='male']['charges'].mean():,.0f}) vs "
                f"female (${df[df.sex=='female']['charges'].mean():,.0f}) show a modest but real gap. "
                "Including sex as a model feature can constitute gender-based price discrimination, "
                "which is illegal in many jurisdictions (e.g. EU Gender Goods Directive). "
                "Models may also learn proxy features correlated with sex (e.g. certain regions, BMI ranges)."
            )
        },
        {
            "title": "🗺️ Geographic / Regional Bias",
            "severity": "MEDIUM",
            "body": (
                f"Region distribution ranges from {df['region'].value_counts().min()} to "
                f"{df['region'].value_counts().max()} records. "
                "Southeast has the highest average BMI and charges, which could cause the model to "
                "penalise residents of that region disproportionately. Geographic proxies often correlate "
                "with race and socioeconomic status, introducing indirect racial bias even when race "
                "is not an explicit feature."
            )
        },
        {
            "title": "🎂 Age Bias",
            "severity": "MEDIUM",
            "body": (
                f"Age spans 18–64 with mean {df['age'].mean():.1f} years. Charges increase steeply "
                "with age, especially after 45. Models trained on this data will systematically predict "
                "higher costs for older individuals, which can conflict with age discrimination laws "
                "(e.g. Age Discrimination Act, GDPR profiling restrictions). The young cohort (18-25) "
                "is under-represented in the high-claim tier, which may lead to underpricing risk for that group."
            )
        },
        {
            "title": "⚖️ Class Imbalance Bias",
            "severity": "MEDIUM",
            "body": (
                f"Claim tiers are imbalanced: Low={len(df[df['claim_tier']=='Low (<$5k)'])}, "
                f"Moderate={len(df[df['claim_tier']=='Moderate ($5k–$16k)'])}, "
                f"High={len(df[df['claim_tier']=='High (>$16k)'])}. "
                "ML classifiers trained without resampling will be biased toward the majority class, "
                "under-predicting high claims — the most costly and important category. "
                "SMOTE, class weights, or stratified sampling should be applied."
            )
        },
        {
            "title": "👶 Children / Family Bias",
            "severity": "LOW",
            "body": (
                f"{(df['children']==0).mean()*100:.1f}% of policyholders have no children. "
                "The relationship between number of children and charges is weakly positive but "
                "can create unfair premium differentials for large families without clinical justification. "
                "Single parents or multi-child households may be disproportionately penalised."
            )
        },
        {
            "title": "📊 Confirmation / Labelling Bias",
            "severity": "HIGH",
            "body": (
                "The 'charges' target variable was likely set by insurers who already used actuarial "
                "models containing historical biases. Training ML models on this target risks perpetuating "
                "and amplifying existing unfair pricing — a form of historical bias laundering. "
                "The model learns to reproduce discriminatory patterns dressed up as 'objective' data."
            )
        },
        {
            "title": "🔢 Sample Size / Representation Bias",
            "severity": "LOW",
            "body": (
                f"The dataset contains only {len(df):,} records, all from a single US insurance context. "
                "It lacks racial/ethnic, income, disability, and urban/rural features. "
                "Conclusions drawn from this dataset should not be generalised globally or used to "
                "justify pricing in diverse populations without significant additional data collection."
            )
        },
    ]

    sev_color = {"HIGH": PALETTE[3], "MEDIUM": PALETTE[6], "LOW": PALETTE[1]}

    for b in biases:
        sev = b["severity"]
        st.markdown(f"""
        <div class="bias-card">
          <div class="bias-title">{b['title']}
            <span style="background:{sev_color[sev]}22;color:{sev_color[sev]};
              border:1px solid {sev_color[sev]};padding:2px 10px;border-radius:20px;
              font-size:0.72rem;font-weight:700;letter-spacing:1px;margin-left:10px">
              {sev}
            </span>
          </div>
          <div class="bias-body">{b['body']}</div>
        </div>""", unsafe_allow_html=True)

    # ── 2. Statistical plots ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">2. Bias Visualisations</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Charges by sex, split by smoker
        fig = px.box(df_f, x="sex", y="charges", color="smoker",
                     color_discrete_map={"yes":PALETTE[3],"no":PALETTE[0]},
                     title="Charge Disparity: Sex × Smoking Status")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Region × claim tier proportion
        rt = df_f.groupby(["region","claim_tier"], observed=True).size().reset_index(name="n")
        rt["pct"] = rt.groupby("region")["n"].transform(lambda x: x/x.sum()*100)
        fig = px.bar(rt, x="region", y="pct", color="claim_tier",
                     color_discrete_map=CLAIM_COLORS, barmode="stack",
                     title="Claim Tier % by Region (representation bias)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Age group × mean charges by sex
        ag = df_f.groupby(["age_group","sex"], observed=True)["charges"].mean().reset_index()
        fig = px.line(ag, x="age_group", y="charges", color="sex",
                      color_discrete_map={"male":PALETTE[0],"female":PALETTE[2]},
                      markers=True, title="Mean Charges by Age Group & Sex (age + gender bias)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Class imbalance bar
        ti = df_f["claim_tier"].value_counts().reset_index()
        ti.columns = ["Tier","Count"]
        fig = px.bar(ti, x="Tier", y="Count", color="Tier",
                     color_discrete_map=CLAIM_COLORS,
                     title="Class Imbalance — Claim Tier Counts")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    # Smoker disparity deep dive
    st.markdown('<div class="section-header">3. Smoker Disparity Deep Dive</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        smk = df_f.groupby(["age_group","smoker"], observed=True)["charges"].mean().reset_index()
        fig = px.bar(smk, x="age_group", y="charges", color="smoker", barmode="group",
                     color_discrete_map={"yes":PALETTE[3],"no":PALETTE[0]},
                     title="Mean Charges: Smoker vs Non-Smoker by Age Group")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        smk2 = df_f.groupby(["region","smoker"], observed=True)["charges"].mean().reset_index()
        fig = px.bar(smk2, x="region", y="charges", color="smoker", barmode="group",
                     color_discrete_map={"yes":PALETTE[3],"no":PALETTE[0]},
                     title="Mean Charges: Smoker vs Non-Smoker by Region")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    # ── 3. Mitigation recommendations ─────────────────────────────────────────
    st.markdown('<div class="section-header">4. Bias Mitigation Recommendations</div>', unsafe_allow_html=True)
    recs = [
        ("Pre-processing", "Remove or carefully audit `sex` and `region` as direct model features. Use fairness-aware feature selection. Apply SMOTE or class weights to address claim tier imbalance."),
        ("In-processing", "Use fairness constraints during training (e.g. adversarial debiasing, reweighing). Apply regularisation to penalise feature interactions that encode protected attributes."),
        ("Post-processing", "Audit model outputs for disparate impact across sex, age group, and region. Apply equalised odds calibration to ensure similar error rates across demographic groups."),
        ("Governance", "Never use this model as the sole basis for pricing decisions. Implement a human-in-the-loop review for edge cases. Conduct periodic bias audits as new data arrives."),
        ("Data collection", "Collect additional features (medical history, lifestyle beyond smoking) to reduce reliance on demographic proxies. Expand dataset diversity across geographies and demographics."),
    ]
    for title, body in recs:
        st.markdown(f"""
        <div class="bias-card">
          <div class="bias-title" style="color:#7c9ef8">✅ {title}</div>
          <div class="bias-body">{body}</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — ML PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML — Prediction":
    from sklearn.linear_model    import LinearRegression, Ridge, Lasso
    from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error

    st.markdown("# 🤖 Machine Learning — Charge Prediction")

    FEAT_COLS = ["age","bmi","children","smoker_bin","sex_bin",
                 "region_northwest","region_southeast","region_southwest"]
    X = df_f[[c for c in FEAT_COLS if c in df_f.columns]].copy()
    y = df_f["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s  = sc.transform(X_test)

    @st.cache_data
    def train_models(hash_val, Xtr, ytr, Xte, yte):
        models = {
            "Linear Reg":    LinearRegression(),
            "Ridge":         Ridge(alpha=10),
            "Lasso":         Lasso(alpha=10),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Grad Boost":    GradientBoostingRegressor(n_estimators=200, random_state=42),
        }
        results = {}
        for name, m in models.items():
            m.fit(Xtr, ytr)
            preds = m.predict(Xte)
            results[name] = {"model":m, "preds":preds,
                             "R2":r2_score(yte,preds),
                             "RMSE":np.sqrt(mean_squared_error(yte,preds)),
                             "MAE":mean_absolute_error(yte,preds)}
        return results

    import hashlib
    h = hashlib.md5(X_train_s.tobytes()).hexdigest()
    results = train_models(h, X_train_s, y_train, X_test_s, y_test)

    comp = pd.DataFrame({k: {"R²":v["R2"],"RMSE":v["RMSE"],"MAE":v["MAE"]} for k,v in results.items()}).T.reset_index()
    comp.columns = ["Model","R²","RMSE","MAE"]

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comp, x="Model", y="R²", color="Model",
                     color_discrete_sequence=PALETTE, title="R² Score (higher = better)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(comp, x="Model", y="RMSE", color="Model",
                     color_discrete_sequence=PALETTE, title="RMSE (lower = better)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp.set_index("Model").round(2))

    sel_model = st.selectbox("Select model for diagnostics", list(results.keys()), index=3)
    preds = results[sel_model]["preds"]
    residuals = y_test.values - preds

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values, y=preds, mode="markers",
                                 marker=dict(color=PALETTE[0], opacity=0.6, size=5), name="Predictions"))
        mn, mx = y_test.min(), y_test.max()
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                 line=dict(color=PALETTE[3], dash="dash"), name="Perfect fit"))
        fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual", yaxis_title="Predicted")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(x=residuals, nbins=60, color_discrete_sequence=[PALETTE[2]],
                           title="Residual Distribution")
        fig.add_vline(x=0, line_dash="dash", line_color=PALETTE[3])
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    if hasattr(results[sel_model]["model"], "feature_importances_"):
        fi = pd.DataFrame({"Feature":X.columns,
                           "Importance":results[sel_model]["model"].feature_importances_}).sort_values("Importance")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[PALETTE[1]], title="Feature Importances")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🎯 Live Prediction</div>', unsafe_allow_html=True)
    co1,co2,co3,co4 = st.columns(4)
    p_age      = co1.slider("Age", 18, 65, 35)
    p_bmi      = co2.slider("BMI", 15.0, 55.0, 30.0)
    p_children = co3.slider("Children", 0, 5, 1)
    p_smoker   = co4.selectbox("Smoker", ["No","Yes"])
    co5, co6   = st.columns(2)
    p_sex      = co5.selectbox("Sex", ["Male","Female"])
    p_region   = co6.selectbox("Region", ["northeast","northwest","southeast","southwest"])

    inp = pd.DataFrame([{
        "age":p_age,"bmi":p_bmi,"children":p_children,
        "smoker_bin":1 if p_smoker=="Yes" else 0,
        "sex_bin":1 if p_sex=="Male" else 0,
        "region_northwest":1 if p_region=="northwest" else 0,
        "region_southeast":1 if p_region=="southeast" else 0,
        "region_southwest":1 if p_region=="southwest" else 0,
    }])
    inp_sc   = sc.transform(inp[[c for c in FEAT_COLS if c in inp.columns]])
    pred_val = results[sel_model]["model"].predict(inp_sc)[0]
    tier_lbl = "High (>$16k)" if pred_val>16000 else ("Moderate ($5k–$16k)" if pred_val>5000 else "Low (<$5k)")
    tier_col = sev_color = {"High (>$16k)":PALETTE[3],"Moderate ($5k–$16k)":PALETTE[6],"Low (<$5k)":PALETTE[1]}[tier_lbl]
    st.markdown(f"""
    <div class="bias-card" style="text-align:center">
      <div style="font-size:0.8rem;color:#8b9cc8;text-transform:uppercase;letter-spacing:2px">
        Predicted Annual Charge ({sel_model})
      </div>
      <div style="font-size:2.8rem;font-weight:700;color:#4ade80;font-family:'JetBrains Mono',monospace">
        ${pred_val:,.2f}
      </div>
      <div style="margin-top:8px">
        <span style="background:{tier_col}22;color:{tier_col};border:1px solid {tier_col};
          padding:4px 14px;border-radius:20px;font-size:0.85rem;font-weight:700">
          {tier_lbl}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — K-MEANS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🧩 K-Means Clustering":
    from sklearn.cluster       import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics       import silhouette_score

    st.markdown("# 🧩 K-Means Clustering")

    feat_options = ["age","bmi","children","charges","smoker_bin","sex_bin"]
    sel_feats = st.multiselect("Features", feat_options, default=["age","bmi","charges","smoker_bin"])
    k = st.slider("Number of Clusters (K)", 2, 10, 4)

    Xc = df_f[sel_feats].dropna()
    sc = StandardScaler()
    Xcs = sc.fit_transform(Xc)

    import hashlib
    h = hashlib.md5(Xcs.tobytes()).hexdigest()

    @st.cache_data
    def run_kmeans(hv, Xcs_arr, k_val):
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        lbl = km.fit_predict(Xcs_arr)
        sil = silhouette_score(Xcs_arr, lbl)
        return lbl, km.inertia_, sil

    @st.cache_data
    def elbow(hv, Xcs_arr):
        inertias, sils = [], []
        for ki in range(2,11):
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            lbl = km.fit_predict(Xcs_arr)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(Xcs_arr, lbl))
        return inertias, sils

    labels, inertia, sil = run_kmeans(h, Xcs, k)
    inertias, sils = elbow(h, Xcs)
    Xc = Xc.copy(); Xc["Cluster"] = labels.astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,11)), y=inertias, mode="lines+markers",
                                 marker=dict(color=PALETTE[0],size=8), line=dict(color=PALETTE[0])))
        fig.add_vline(x=k, line_dash="dash", line_color=PALETTE[3])
        fig.update_layout(title="Elbow Curve", xaxis_title="K", yaxis_title="Inertia")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,11)), y=sils, mode="lines+markers",
                                 marker=dict(color=PALETTE[1],size=8), line=dict(color=PALETTE[1])))
        fig.add_vline(x=k, line_dash="dash", line_color=PALETTE[3])
        fig.update_layout(title="Silhouette Score", xaxis_title="K", yaxis_title="Score")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(Xcs)
    pca_df = pd.DataFrame(pca_coords, columns=["PC1","PC2"])
    pca_df["Cluster"] = labels.astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         color_discrete_sequence=PALETTE,
                         title=f"PCA 2D — {k} Clusters (Silhouette: {sil:.3f})", opacity=0.7)
        fig.update_traces(marker=dict(size=5))
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        profile = Xc.groupby("Cluster")[sel_feats].mean().round(2).reset_index()
        fig = px.bar(profile.melt(id_vars="Cluster"), x="variable", y="value",
                     color="Cluster", barmode="group",
                     color_discrete_sequence=PALETTE, title="Cluster Profiles (mean values)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cluster Summary**")
    st.dataframe(Xc.groupby("Cluster")[sel_feats].describe().round(2))


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 7 — APRIORI
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Apriori Rules":
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        st.error("Install mlxtend: `pip install mlxtend`"); st.stop()

    st.markdown("# 🛒 Apriori Association Rules")

    df_ap = df_f.copy()
    df_ap["age_cat"]     = pd.cut(df_ap["age"],   bins=[17,30,45,65], labels=["Young","Mid","Senior"])
    df_ap["bmi_cat"]     = pd.cut(df_ap["bmi"],   bins=[0,25,30,55],  labels=["Normal","Overweight","Obese"])
    df_ap["charge_cat"]  = pd.cut(df_ap["charges"],bins=[0,5000,16000,np.inf],
                                  labels=["LowCharge","ModCharge","HighCharge"])
    df_ap["children_cat"]= df_ap["children"].apply(
        lambda x: "NoChild" if x==0 else ("1Child" if x==1 else "2+Children"))
    ohe = pd.get_dummies(df_ap[["age_cat","bmi_cat","charge_cat","smoker","sex","region","children_cat"]].astype(str))

    col1, col2, col3 = st.columns(3)
    min_sup  = col1.slider("Min Support",    0.05, 0.5, 0.15, 0.01)
    min_conf = col2.slider("Min Confidence", 0.3,  0.9, 0.5,  0.05)
    min_lift = col3.slider("Min Lift",        1.0,  5.0, 1.2,  0.1)

    import hashlib
    h = hashlib.md5(ohe.values.tobytes()).hexdigest()

    @st.cache_data
    def run_apriori(hv, ohe_bool, ms, mc, ml):
        freq = apriori(ohe_bool.astype(bool), min_support=ms, use_colnames=True, max_len=4)
        if freq.empty: return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq, metric="confidence", min_threshold=mc)
        rules = rules[rules["lift"] >= ml]
        return freq, rules

    freq_items, rules = run_apriori(h, ohe, min_sup, min_conf, min_lift)

    if rules.empty:
        st.warning("No rules found — try lowering thresholds.")
    else:
        st.success(f"**{len(rules)}** rules found from **{len(freq_items)}** frequent itemsets.")
        col1, col2 = st.columns(2)
        with col1:
            tr = rules.nlargest(20,"lift")[["antecedents","consequents","support","confidence","lift"]].copy()
            tr["antecedents"] = tr["antecedents"].apply(lambda x: ", ".join(list(x)))
            tr["consequents"] = tr["consequents"].apply(lambda x: ", ".join(list(x)))
            st.markdown("**Top 20 Rules by Lift**")
            st.dataframe(tr.round(4), use_container_width=True)
        with col2:
            fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                             color_continuous_scale="Blues",
                             title="Support vs Confidence (size/color = Lift)")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        fi2 = freq_items.copy()
        fi2["itemsets"] = fi2["itemsets"].apply(lambda x: ", ".join(list(x)))
        fi2 = fi2.sort_values("support", ascending=False).head(30)
        fig = px.bar(fi2, x="support", y="itemsets", orientation="h",
                     color_discrete_sequence=[PALETTE[0]], title="Top 30 Frequent Itemsets")
        fig.update_layout(height=600, yaxis=dict(tickfont=dict(size=10)))
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 8 — XGBOOST
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚡ XGBoost Analysis":
    try:
        import xgboost as xgb
    except ImportError:
        st.error("Install xgboost: `pip install xgboost`"); st.stop()

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error

    st.markdown("# ⚡ XGBoost Analysis")

    FEAT_COLS = ["age","bmi","children","smoker_bin","sex_bin",
                 "region_northwest","region_southeast","region_southwest"]
    X = df_f[[c for c in FEAT_COLS if c in df_f.columns]].copy()
    y = df_f["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    col1, col2, col3 = st.columns(3)
    n_est = col1.slider("n_estimators",  50, 500, 200, 50)
    m_dep = col2.slider("max_depth",      2,  10,   5)
    lr    = col3.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)

    import hashlib
    h = hashlib.md5(X_train.values.tobytes()).hexdigest()

    @st.cache_data
    def run_xgb(hv, Xtr, ytr, Xte, yte, ne, md, lrv):
        m = xgb.XGBRegressor(n_estimators=ne, max_depth=md, learning_rate=lrv,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(Xtr, ytr, eval_set=[(Xtr,ytr),(Xte,yte)], verbose=False)
        preds = m.predict(Xte)
        evals = m.evals_result()
        cv    = cross_val_score(m, Xtr, ytr, cv=5, scoring="r2")
        return m, preds, evals, cv

    model, preds, evals, cv_scores = run_xgb(h, X_train, y_train, X_test, y_test, n_est, m_dep, lr)

    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val) in zip([c1,c2,c3,c4],[("R²",f"{r2:.4f}"),("RMSE",f"${rmse:,.0f}"),
                                              ("MAE",f"${mae:,.0f}"),("CV R²",f"{cv_scores.mean():.4f}")]):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)

    train_rmse = np.sqrt(evals["validation_0"]["rmse"])
    test_rmse  = np.sqrt(evals["validation_1"]["rmse"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_rmse, name="Train RMSE", line=dict(color=PALETTE[0])))
    fig.add_trace(go.Scatter(y=test_rmse,  name="Test RMSE",  line=dict(color=PALETTE[3])))
    fig.update_layout(title="RMSE over Boosting Rounds", xaxis_title="Round", yaxis_title="RMSE")
    dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    fi = pd.DataFrame({"Feature":X.columns,"Importance":model.feature_importances_}).sort_values("Importance")
    with col1:
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[PALETTE[1]], title="Feature Importance (Gain)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        residuals = y_test.values - preds
        fig = px.scatter(x=preds, y=residuals, color_discrete_sequence=[PALETTE[2]],
                         title="Residuals vs Fitted",
                         labels={"x":"Fitted","y":"Residuals"})
        fig.add_hline(y=0, line_dash="dash", line_color=PALETTE[3])
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(5)], y=cv_scores, marker_color=PALETTE[:5]))
    fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color=PALETTE[3],
                  annotation_text=f"Mean = {cv_scores.mean():.4f}")
    fig.update_layout(title="5-Fold Cross Validation R²", yaxis_title="R²")
    dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Per-Sample Feature Contributions</div>', unsafe_allow_html=True)
    sample_idx = st.slider("Select test sample", 0, len(X_test)-1, 0)
    base_pred = model.predict(X_test)[sample_idx]
    sample = X_test.iloc[[sample_idx]]
    contributions = {}
    for feat in X.columns:
        mod = sample.copy(); mod[feat] = X_train[feat].mean()
        contributions[feat] = base_pred - model.predict(mod)[0]
    cdf = pd.DataFrame.from_dict(contributions, orient="index", columns=["Contribution"]).sort_values("Contribution")
    fig = px.bar(cdf, x="Contribution", y=cdf.index, orientation="h",
                 color="Contribution", color_continuous_scale=["#f87171","#1e2235","#4ade80"],
                 title=f"Feature Contributions — Sample #{sample_idx} (Predicted: ${base_pred:,.0f})")
    dark_fig(fig); st.plotly_chart(fig, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5080;font-size:0.8rem;'>Insurance Analytics Hub • Streamlit Dashboard</p>",
    unsafe_allow_html=True
)
