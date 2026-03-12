"""
Insurance Claims Analytics Dashboard — Streamlit App
Dataset: Indian Life Insurance Claims (Approved Death Claim vs Repudiate Death)
Run with: streamlit run insurance_dashboard.py
Requirements: pip install streamlit pandas numpy scikit-learn xgboost mlxtend plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claims Analytics",
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
  .metric-value { font-size: 1.9rem; font-weight: 700; color: #7c9ef8;
    font-family: 'JetBrains Mono', monospace; }
  .metric-label { font-size: 0.72rem; color: #8b9cc8; text-transform: uppercase;
    letter-spacing: 2px; margin-top: 4px; }
  .section-header {
    font-size: 1.25rem; font-weight: 700; color: #e8eaf6;
    border-left: 4px solid #7c9ef8; padding-left: 12px; margin: 24px 0 14px 0;
  }
  .info-card {
    background: #151829; border: 1px solid #2a2f52; border-radius: 10px;
    padding: 18px 20px; margin-bottom: 14px;
  }
  .info-title { font-size: 1rem; font-weight: 700; color: #f472b6; margin-bottom: 6px; }
  .info-body  { font-size: 0.87rem; color: #9aa5c8; line-height: 1.75; }
  div[data-testid="stSidebar"] { background: #10121f; border-right: 1px solid #1e2235; }
  div[data-testid="stSidebar"] h2 { color: #e8eaf6 !important; font-size: 1.4rem !important; }
  div[data-testid="stSidebar"] p  { color: #8b9cc8 !important; font-size: 0.8rem !important; }
  h1 { color: #e8eaf6 !important; }
  h2, h3 { color: #c4cbe8 !important; }
  p, li { color: #9aa5c8 !important; }
  .stSelectbox label, .stSlider label, .stMultiSelect label, .stRadio label { color: #9aa5c8 !important; }
  code { font-family: 'JetBrains Mono', monospace; background: #1e2235;
    color: #7c9ef8; padding: 2px 6px; border-radius: 4px; }
  /* Radio option styling */
  div[data-testid="stSidebar"] .stRadio > label { font-size: 0.78rem !important; color: #8b9cc8 !important;
    text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

PALETTE    = ["#7c9ef8","#4ade80","#f472b6","#fb923c","#a78bfa","#34d399","#fbbf24","#60a5fa","#f87171","#38bdf8"]
PLOTLY_BG  = "#0d0f1a"
GRID_COLOR = "#1e2235"
STATUS_MAP = {"Approved Death Claim": "#4ade80", "Repudiate Death": "#f87171"}

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

# ── Load & clean data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data(path="Insurance.csv"):
    df = pd.read_csv(path)
    df["SUM_ASSURED"]      = df["SUM_ASSURED"].astype(str).str.replace(",","").astype(float)
    df["PI_ANNUAL_INCOME"] = df["PI_ANNUAL_INCOME"].astype(str).str.replace(",","").astype(float)
    df["CLAIMED"]          = (df["POLICY_STATUS"] == "Approved Death Claim").astype(int)
    df["AGE_GROUP"]        = pd.cut(df["PI_AGE"],
        bins=[0,20,30,40,50,60,70,85],
        labels=["<20","20-30","31-40","41-50","51-60","61-70","71+"])
    df["SA_TIER"] = pd.cut(df["SUM_ASSURED"],
        bins=[0,100000,300000,600000,1500000,np.inf],
        labels=["<1L","1L-3L","3L-6L","6L-15L",">15L"])
    df["INCOME_TIER"] = pd.cut(df["PI_ANNUAL_INCOME"],
        bins=[-1,0,100000,300000,600000,np.inf],
        labels=["Not Disclosed","<1L","1L-3L","3L-6L",">6L"])
    top_reasons = df["REASON_FOR_CLAIM"].value_counts().head(10).index.tolist()
    df["REASON_GROUPED"] = df["REASON_FOR_CLAIM"].apply(
        lambda x: x if x in top_reasons else ("Not Stated" if pd.isna(x) else "Other"))
    df["GENDER_BIN"]  = (df["PI_GENDER"] == "M").astype(int)
    df["EARLY_BIN"]   = (df["EARLY_NON"] == "EARLY").astype(int)
    df["MEDICAL_BIN"] = (df["MEDICAL_NONMED"] == "MEDICAL").astype(int)
    pm_dummies = pd.get_dummies(df["PAYMENT_MODE"], prefix="PM", drop_first=True)
    df = pd.concat([df, pm_dummies], axis=1)
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Insurance Claims")
    st.markdown("Indian Life Insurance Analytics")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        import tempfile
        raw = uploaded.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(raw); tmp_path = tmp.name
        df = load_data(tmp_path)
        st.success("Dataset loaded!")
    else:
        df = load_data()

    st.markdown("---")
    st.markdown("**Select Analysis Module**")
    page = st.radio("", [
        "📊 Executive Overview",
        "🎯 Classification Analysis",
        "🔮 Clustering Analysis",
        "📈 Regression Analysis",
        "🔗 Association Rules",
        "🗺️ Geographic Analysis",
        "⚖️ Bias & Fairness Analysis",
        "🔍 Deep Drill-Down Analysis",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Filters**")
    sel_status = st.multiselect("Claim Status",
        df["POLICY_STATUS"].unique().tolist(),
        default=df["POLICY_STATUS"].unique().tolist())
    sel_gender = st.multiselect("Gender",
        df["PI_GENDER"].unique().tolist(),
        default=df["PI_GENDER"].unique().tolist())
    age_range  = st.slider("Age range",
        int(df["PI_AGE"].min()), int(df["PI_AGE"].max()), (3, 82))

    df_f = df[
        df["POLICY_STATUS"].isin(sel_status) &
        df["PI_GENDER"].isin(sel_gender) &
        df["PI_AGE"].between(*age_range)
    ]
    st.caption(f"**{len(df_f):,}** records selected")

approved   = df_f[df_f["POLICY_STATUS"] == "Approved Death Claim"]
repudiated = df_f[df_f["POLICY_STATUS"] == "Repudiate Death"]


# ════════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown("# 📊 Executive Overview")
    st.markdown("High-level summary of all insurance claims — approved vs repudiated.")

    approval_rate = len(approved) / len(df_f) * 100 if len(df_f) else 0

    cols = st.columns(6)
    for col, (label, val) in zip(cols, [
        ("Total Policies",  f"{len(df_f):,}"),
        ("Approved",        f"{len(approved):,}"),
        ("Repudiated",      f"{len(repudiated):,}"),
        ("Approval Rate",   f"{approval_rate:.1f}%"),
        ("Avg Sum Assured", f"₹{df_f['SUM_ASSURED'].mean():,.0f}"),
        ("Avg Age",         f"{df_f['PI_AGE'].mean():.1f} yrs"),
    ]):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Claim Status Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        cnt = df_f["POLICY_STATUS"].value_counts().reset_index()
        fig = px.pie(cnt, names="POLICY_STATUS", values="count", hole=0.52,
                     color="POLICY_STATUS", color_discrete_map=STATUS_MAP,
                     title="Approved vs Repudiated")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_zones = df_f["ZONE"].value_counts().head(10).index
        zs = df_f[df_f["ZONE"].isin(top_zones)].groupby(["ZONE","POLICY_STATUS"]).size().reset_index(name="n")
        fig = px.bar(zs, x="ZONE", y="n", color="POLICY_STATUS",
                     color_discrete_map=STATUS_MAP, barmode="stack",
                     title="Claims Volume by Zone (Top 10)")
        fig.update_xaxes(tickangle=35)
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Demographics at a Glance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(df_f, x="PI_AGE", color="POLICY_STATUS",
                           color_discrete_map=STATUS_MAP, barmode="overlay",
                           nbins=40, opacity=0.8, title="Age Distribution")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df_f, x="POLICY_STATUS", y="SUM_ASSURED", color="POLICY_STATUS",
                     color_discrete_map=STATUS_MAP, log_y=True,
                     title="Sum Assured by Status")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col3:
        g_s = df_f.groupby(["PI_GENDER","POLICY_STATUS"]).size().reset_index(name="n")
        g_s["pct"] = g_s.groupby("PI_GENDER")["n"].transform(lambda x: x/x.sum()*100)
        fig = px.bar(g_s, x="PI_GENDER", y="pct", color="POLICY_STATUS",
                     color_discrete_map=STATUS_MAP, barmode="stack",
                     title="Approval Rate by Gender (%)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Cause of Death — Top 10</div>', unsafe_allow_html=True)
    rc = df_f["REASON_FOR_CLAIM"].value_counts().head(10).reset_index()
    rc.columns = ["Cause","Count"]
    fig = px.bar(rc, x="Count", y="Cause", orientation="h",
                 color="Count", color_continuous_scale="Blues",
                 title="Most Common Causes of Death")
    dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Payment Mode & Policy Type Mix</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        pm = df_f["PAYMENT_MODE"].value_counts().reset_index()
        fig = px.pie(pm, names="PAYMENT_MODE", values="count", hole=0.45,
                     color_discrete_sequence=PALETTE, title="Payment Mode")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        en = df_f["EARLY_NON"].value_counts().reset_index()
        fig = px.pie(en, names="EARLY_NON", values="count", hole=0.45,
                     color_discrete_sequence=[PALETTE[1],PALETTE[3]], title="Early vs Non-Early")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col3:
        med = df_f["MEDICAL_NONMED"].value_counts().reset_index()
        fig = px.pie(med, names="MEDICAL_NONMED", values="count", hole=0.45,
                     color_discrete_sequence=[PALETTE[4],PALETTE[5]], title="Medical vs Non-Medical")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — CLASSIFICATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Classification Analysis":
    from sklearn.linear_model    import LogisticRegression
    from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import (confusion_matrix, roc_auc_score,
                                         roc_curve, accuracy_score,
                                         precision_score, recall_score, f1_score)
    import hashlib

    st.markdown("# 🎯 Classification Analysis")
    st.markdown("Binary classification: predict whether a claim will be **Approved** or **Repudiated**.")

    FEAT_COLS = ["PI_AGE","SUM_ASSURED","GENDER_BIN","EARLY_BIN","MEDICAL_BIN",
                 "PM_Half-Yly","PM_Monthly","PM_Quarterly","PM_Single"]
    X = df_f[[c for c in FEAT_COLS if c in df_f.columns]].fillna(0)
    y = df_f["CLAIMED"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    h = hashlib.md5(Xtr.tobytes()).hexdigest()

    @st.cache_data
    def train_classifiers(hv, Xtr, ytr, Xte, yte):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        }
        out = {}
        for name, m in models.items():
            m.fit(Xtr, ytr)
            preds = m.predict(Xte)
            proba = m.predict_proba(Xte)[:,1]
            out[name] = {
                "model": m, "preds": preds, "proba": proba,
                "accuracy":  accuracy_score(yte, preds),
                "auc":       roc_auc_score(yte, proba),
                "precision": precision_score(yte, preds),
                "recall":    recall_score(yte, preds),
                "f1":        f1_score(yte, preds),
                "cm":        confusion_matrix(yte, preds),
            }
        return out

    results = train_classifiers(h, Xtr, y_train, Xte, y_test)
    comp = pd.DataFrame({k: {
        "Accuracy": v["accuracy"], "AUC": v["auc"],
        "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"]
    } for k,v in results.items()}).T.reset_index()
    comp.columns = ["Model","Accuracy","AUC","Precision","Recall","F1"]

    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        metrics_melt = comp.melt(id_vars="Model", value_vars=["Accuracy","AUC","Precision","Recall","F1"])
        fig = px.bar(metrics_melt, x="variable", y="value", color="Model",
                     barmode="group", color_discrete_sequence=PALETTE,
                     title="All Metrics — Side by Side")
        fig.update_yaxes(range=[0,1])
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(comp, x="Model", y="AUC", color="Model",
                     color_discrete_sequence=PALETTE, title="ROC-AUC Score")
        fig.update_yaxes(range=[0,1])
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comp.set_index("Model").round(4), use_container_width=True)

    sel = st.selectbox("Select model for detailed diagnostics",
                       list(results.keys()), index=1)
    res = results[sel]
    fpr, tpr, _ = roc_curve(y_test, res["proba"])

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"AUC = {res['auc']:.3f}",
                                 line=dict(color=PALETTE[0], width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 line=dict(dash="dash", color=PALETTE[3]), name="Random"))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        cm = res["cm"]
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred: Repudiated","Pred: Approved"],
            y=["True: Repudiated","True: Approved"],
            colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False
        ))
        fig.update_layout(title="Confusion Matrix",
                          plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_BG,
                          font=dict(color="#9aa5c8"))
        st.plotly_chart(fig, use_container_width=True)

    if hasattr(res["model"], "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": res["model"].feature_importances_
        }).sort_values("Importance")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[PALETTE[1]], title="Feature Importances")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">🎯 Live Claim Prediction</div>', unsafe_allow_html=True)
    co1, co2, co3 = st.columns(3)
    p_age    = co1.slider("Age", 3, 82, 50)
    p_sa     = co2.number_input("Sum Assured (₹)", min_value=0, value=300000, step=50000)
    p_gender = co3.selectbox("Gender", ["M","F"])
    co4, co5, co6 = st.columns(3)
    p_early  = co4.selectbox("Policy Type", ["NON EARLY","EARLY"])
    p_med    = co5.selectbox("Medical Exam", ["NON MEDICAL","MEDICAL"])
    p_pm     = co6.selectbox("Payment Mode", ["Annual","Half-Yly","Monthly","Quarterly","Single"])

    inp = pd.DataFrame([{
        "PI_AGE": p_age, "SUM_ASSURED": p_sa,
        "GENDER_BIN":    1 if p_gender=="M"       else 0,
        "EARLY_BIN":     1 if p_early=="EARLY"    else 0,
        "MEDICAL_BIN":   1 if p_med=="MEDICAL"    else 0,
        "PM_Half-Yly":   1 if p_pm=="Half-Yly"    else 0,
        "PM_Monthly":    1 if p_pm=="Monthly"      else 0,
        "PM_Quarterly":  1 if p_pm=="Quarterly"    else 0,
        "PM_Single":     1 if p_pm=="Single"       else 0,
    }])
    inp_s = sc.transform(inp[[c for c in FEAT_COLS if c in inp.columns]])
    prob  = res["model"].predict_proba(inp_s)[0]
    pred  = res["model"].predict(inp_s)[0]
    label = "✅ Likely Approved" if pred == 1 else "❌ Likely Repudiated"
    clr   = "#4ade80" if pred == 1 else "#f87171"
    st.markdown(f"""
    <div class="info-card" style="text-align:center">
      <div style="font-size:0.8rem;color:#8b9cc8;text-transform:uppercase;letter-spacing:2px">
        Prediction ({sel})
      </div>
      <div style="font-size:2.2rem;font-weight:700;color:{clr};margin-top:8px">{label}</div>
      <div style="margin-top:10px;color:#9aa5c8">
        Approved probability: <span style="color:#4ade80;font-family:'JetBrains Mono'">{prob[1]*100:.1f}%</span>
        &nbsp;|&nbsp;
        Repudiated: <span style="color:#f87171;font-family:'JetBrains Mono'">{prob[0]*100:.1f}%</span>
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — CLUSTERING ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Clustering Analysis":
    from sklearn.cluster       import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics       import silhouette_score
    import hashlib

    st.markdown("# 🔮 Clustering Analysis")
    st.markdown("K-Means segmentation — group policyholders into distinct risk and demographic profiles.")

    feat_options = ["PI_AGE","SUM_ASSURED","PI_ANNUAL_INCOME",
                    "GENDER_BIN","EARLY_BIN","MEDICAL_BIN","CLAIMED"]
    sel_feats = st.multiselect("Features for clustering", feat_options,
                               default=["PI_AGE","SUM_ASSURED","EARLY_BIN","CLAIMED"])
    k = st.slider("Number of Clusters (K)", 2, 10, 4)

    Xc  = df_f[sel_feats].fillna(0)
    sc  = StandardScaler()
    Xcs = sc.fit_transform(Xc)
    h   = hashlib.md5(Xcs.tobytes()).hexdigest()

    @st.cache_data
    def run_kmeans(hv, arr, k_val):
        km  = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        lbl = km.fit_predict(arr)
        sil = silhouette_score(arr, lbl)
        return lbl, km.inertia_, sil

    @st.cache_data
    def elbow_curve(hv, arr):
        inertias, sils = [], []
        for ki in range(2, 11):
            km  = KMeans(n_clusters=ki, random_state=42, n_init=10)
            lbl = km.fit_predict(arr)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(arr, lbl))
        return inertias, sils

    labels, inertia, sil = run_kmeans(h, Xcs, k)
    inertias, sils = elbow_curve(h, Xcs)
    Xc = Xc.copy()
    Xc["Cluster"] = labels.astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,11)), y=inertias, mode="lines+markers",
                                 marker=dict(color=PALETTE[0], size=8),
                                 line=dict(color=PALETTE[0])))
        fig.add_vline(x=k, line_dash="dash", line_color=PALETTE[3],
                      annotation_text=f"K={k}")
        fig.update_layout(title="Elbow Curve", xaxis_title="K", yaxis_title="Inertia")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,11)), y=sils, mode="lines+markers",
                                 marker=dict(color=PALETTE[1], size=8),
                                 line=dict(color=PALETTE[1])))
        fig.add_vline(x=k, line_dash="dash", line_color=PALETTE[3],
                      annotation_text=f"K={k}")
        fig.update_layout(title="Silhouette Score", xaxis_title="K", yaxis_title="Score")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    pca = PCA(n_components=2)
    pca_df = pd.DataFrame(pca.fit_transform(Xcs), columns=["PC1","PC2"])
    pca_df["Cluster"] = labels.astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         color_discrete_sequence=PALETTE, opacity=0.7,
                         title=f"PCA 2D Projection — {k} Clusters  |  Silhouette: {sil:.3f}")
        fig.update_traces(marker=dict(size=5))
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        profile = Xc.groupby("Cluster")[sel_feats].mean().round(2).reset_index()
        fig = px.bar(profile.melt(id_vars="Cluster"), x="variable", y="value",
                     color="Cluster", barmode="group",
                     color_discrete_sequence=PALETTE,
                     title="Cluster Profiles — Mean Feature Values")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    # Cluster composition by claim status
    Xc_full = df_f.copy()
    Xc_full["Cluster"] = labels.astype(str)
    cs = Xc_full.groupby(["Cluster","POLICY_STATUS"]).size().reset_index(name="n")
    cs["pct"] = cs.groupby("Cluster")["n"].transform(lambda x: x/x.sum()*100)
    fig = px.bar(cs, x="Cluster", y="pct", color="POLICY_STATUS",
                 color_discrete_map=STATUS_MAP, barmode="stack",
                 title="Claim Status % within Each Cluster")
    dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cluster Summary Statistics**")
    st.dataframe(Xc.groupby("Cluster")[sel_feats].describe().round(2), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — REGRESSION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Regression Analysis":
    from sklearn.linear_model    import LinearRegression, Ridge, Lasso
    from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error
    import hashlib

    st.markdown("# 📈 Regression Analysis")
    st.markdown("Predict **Sum Assured** (policy value) based on policyholder and policy attributes.")

    FEAT_COLS = ["PI_AGE","PI_ANNUAL_INCOME","GENDER_BIN","EARLY_BIN",
                 "MEDICAL_BIN","CLAIMED","PM_Half-Yly","PM_Monthly","PM_Quarterly","PM_Single"]
    X = df_f[[c for c in FEAT_COLS if c in df_f.columns]].fillna(0)
    y = df_f["SUM_ASSURED"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    h = hashlib.md5(Xtr.tobytes()).hexdigest()

    @st.cache_data
    def train_regressors(hv, Xtr, ytr, Xte, yte):
        models = {
            "Linear Regression":  LinearRegression(),
            "Ridge":              Ridge(alpha=100),
            "Lasso":              Lasso(alpha=100),
            "Random Forest":      RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, random_state=42),
        }
        out = {}
        for name, m in models.items():
            m.fit(Xtr, ytr)
            preds = m.predict(Xte)
            out[name] = {
                "model": m, "preds": preds,
                "R2":   r2_score(yte, preds),
                "RMSE": np.sqrt(mean_squared_error(yte, preds)),
                "MAE":  mean_absolute_error(yte, preds),
            }
        return out

    results = train_regressors(h, Xtr, y_train, Xte, y_test)
    comp = pd.DataFrame({k: {"R²": v["R2"], "RMSE": v["RMSE"], "MAE": v["MAE"]}
                         for k,v in results.items()}).T.reset_index()
    comp.columns = ["Model","R²","RMSE","MAE"]

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comp, x="Model", y="R²", color="Model",
                     color_discrete_sequence=PALETTE, title="R² Score (higher = better)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(comp, x="Model", y="RMSE", color="Model",
                     color_discrete_sequence=PALETTE, title="RMSE — ₹ (lower = better)")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp.set_index("Model").round(2), use_container_width=True)

    sel = st.selectbox("Select model for diagnostics", list(results.keys()), index=3)
    res = results[sel]
    preds     = res["preds"]
    residuals = y_test.values - preds

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values, y=preds, mode="markers",
                                 marker=dict(color=PALETTE[0], opacity=0.5, size=5),
                                 name="Predictions"))
        mn, mx = float(y_test.min()), float(y_test.max())
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                 line=dict(color=PALETTE[3], dash="dash"), name="Perfect fit"))
        fig.update_layout(title="Predicted vs Actual Sum Assured",
                          xaxis_title="Actual", yaxis_title="Predicted")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(x=residuals, nbins=60,
                           color_discrete_sequence=[PALETTE[2]],
                           title="Residual Distribution")
        fig.add_vline(x=0, line_dash="dash", line_color=PALETTE[3])
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    if hasattr(res["model"], "feature_importances_"):
        fi = pd.DataFrame({
            "Feature":    X.columns,
            "Importance": res["model"].feature_importances_
        }).sort_values("Importance")
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[PALETTE[1]], title="Feature Importances")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    # Cross-validation
    cv_scores = cross_val_score(res["model"], Xtr, y_train, cv=5, scoring="r2")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(5)],
                         y=cv_scores, marker_color=PALETTE[:5]))
    fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color=PALETTE[3],
                  annotation_text=f"Mean R² = {cv_scores.mean():.4f}")
    fig.update_layout(title="5-Fold Cross Validation R²", yaxis_title="R²")
    dark_fig(fig); st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        st.error("Install mlxtend: `pip install mlxtend`"); st.stop()
    import hashlib

    st.markdown("# 🔗 Association Rules (Apriori)")
    st.markdown("Discover co-occurrence patterns among policyholder attributes and claim outcomes.")

    df_ap = df_f.copy()
    df_ap["age_cat"]    = pd.cut(df_ap["PI_AGE"],
        bins=[0,35,50,65,85], labels=["Young (<35)","Mid (35-50)","Senior (50-65)","Elder (65+)"])
    df_ap["sa_cat"]     = df_ap["SA_TIER"].astype(str)
    df_ap["status_cat"] = df_ap["POLICY_STATUS"].apply(
        lambda x: "Approved" if "Approved" in x else "Repudiated")
    df_ap["reason_cat"] = df_ap["REASON_GROUPED"].fillna("Not Stated")

    ohe = pd.get_dummies(
        df_ap[["age_cat","sa_cat","PAYMENT_MODE","EARLY_NON",
               "MEDICAL_NONMED","PI_GENDER","status_cat","reason_cat"]].astype(str)
    )

    col1, col2, col3 = st.columns(3)
    min_sup  = col1.slider("Min Support",    0.05, 0.5, 0.12, 0.01)
    min_conf = col2.slider("Min Confidence", 0.3,  0.9, 0.5,  0.05)
    min_lift = col3.slider("Min Lift",        1.0,  5.0, 1.2,  0.1)

    h = hashlib.md5(ohe.values.tobytes()).hexdigest()

    @st.cache_data
    def run_apriori(hv, ohe_bool, ms, mc, ml):
        freq = apriori(ohe_bool.astype(bool), min_support=ms, use_colnames=True, max_len=4)
        if freq.empty:
            return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq, metric="confidence", min_threshold=mc)
        rules = rules[rules["lift"] >= ml]
        return freq, rules

    freq_items, rules = run_apriori(h, ohe, min_sup, min_conf, min_lift)

    if rules.empty:
        st.warning("No rules found with current thresholds — try lowering Support or Confidence.")
    else:
        st.success(f"Found **{len(rules)}** association rules from **{len(freq_items)}** frequent itemsets.")

        col1, col2 = st.columns(2)
        with col1:
            tr = rules.nlargest(20,"lift")[["antecedents","consequents",
                                           "support","confidence","lift"]].copy()
            tr["antecedents"] = tr["antecedents"].apply(lambda x: ", ".join(list(x)))
            tr["consequents"] = tr["consequents"].apply(lambda x: ", ".join(list(x)))
            st.markdown("**Top 20 Rules by Lift**")
            st.dataframe(tr.round(4), use_container_width=True)
        with col2:
            fig = px.scatter(rules, x="support", y="confidence",
                             size="lift", color="lift",
                             color_continuous_scale="Blues",
                             title="Support vs Confidence (size & colour = Lift)")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        fi2 = freq_items.copy()
        fi2["itemsets"] = fi2["itemsets"].apply(lambda x: ", ".join(list(x)))
        fi2 = fi2.sort_values("support", ascending=False).head(30)
        fig = px.bar(fi2, x="support", y="itemsets", orientation="h",
                     color_discrete_sequence=[PALETTE[0]],
                     title="Top 30 Frequent Itemsets by Support")
        fig.update_layout(height=640, yaxis=dict(tickfont=dict(size=9)))
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        # Rules leading to Repudiation
        st.markdown('<div class="section-header">Rules Leading to Repudiation</div>', unsafe_allow_html=True)
        repud_rules = rules[rules["consequents"].apply(lambda x: "Repudiated" in str(x))]
        if not repud_rules.empty:
            rr = repud_rules.nlargest(15,"lift")[["antecedents","consequents","support","confidence","lift"]].copy()
            rr["antecedents"] = rr["antecedents"].apply(lambda x: ", ".join(list(x)))
            rr["consequents"] = rr["consequents"].apply(lambda x: ", ".join(list(x)))
            st.dataframe(rr.round(4), use_container_width=True)
        else:
            st.info("No repudiation-targeted rules at current thresholds.")


# ════════════════════════════════════════════════════════════════════════════════
#  MODULE 6 — GEOGRAPHIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Geographic Analysis":
    st.markdown("# 🗺️ Geographic Analysis")
    st.markdown("Claims distribution, approval rates, and sum assured patterns across Indian states and zones.")

    tab1, tab2, tab3 = st.tabs(["By State","By Zone","Heatmaps"])

    with tab1:
        state_data = df_f.groupby(["PI_STATE","POLICY_STATUS"]).size().reset_index(name="Count")
        state_pivot = state_data.pivot(index="PI_STATE", columns="POLICY_STATUS",
                                       values="Count").fillna(0).reset_index()
        state_pivot.columns.name = None
        if "Approved Death Claim" not in state_pivot.columns:
            state_pivot["Approved Death Claim"] = 0
        if "Repudiate Death" not in state_pivot.columns:
            state_pivot["Repudiate Death"] = 0
        state_pivot["Total"] = state_pivot["Approved Death Claim"] + state_pivot["Repudiate Death"]
        state_pivot["Approval Rate %"] = (
            state_pivot["Approved Death Claim"] / state_pivot["Total"] * 100).round(1)
        state_pivot = state_pivot.sort_values("Total", ascending=False)

        fig = px.bar(state_pivot.head(20), x="PI_STATE",
                     y=["Approved Death Claim","Repudiate Death"],
                     color_discrete_map=STATUS_MAP, barmode="stack",
                     title="Top 20 States — Claim Volume & Status")
        fig.update_xaxes(tickangle=40)
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(state_pivot.sort_values("Approval Rate %"),
                         x="Approval Rate %", y="PI_STATE", orientation="h",
                         color="Approval Rate %", color_continuous_scale="RdYlGn",
                         title="Approval Rate % by State")
            fig.update_layout(height=560)
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            avg_sa = df_f.groupby("PI_STATE")["SUM_ASSURED"].mean().reset_index()
            avg_sa = avg_sa.sort_values("SUM_ASSURED", ascending=False).head(20)
            fig = px.bar(avg_sa, x="SUM_ASSURED", y="PI_STATE", orientation="h",
                         color_discrete_sequence=[PALETTE[0]],
                         title="Avg Sum Assured by State (Top 20)")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    with tab2:
        zone_data = df_f.groupby(["ZONE","POLICY_STATUS"]).size().reset_index(name="n")
        fig = px.bar(zone_data, x="ZONE", y="n", color="POLICY_STATUS",
                     color_discrete_map=STATUS_MAP, barmode="stack",
                     title="All Zones — Claim Volume & Status")
        fig.update_xaxes(tickangle=40)
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        zone_r = df_f.groupby("ZONE").apply(
            lambda x: (x["POLICY_STATUS"]=="Approved Death Claim").mean()*100
        ).reset_index(name="Approval Rate %").sort_values("Approval Rate %")
        fig = px.bar(zone_r, x="Approval Rate %", y="ZONE", orientation="h",
                     color="Approval Rate %", color_continuous_scale="RdYlGn",
                     title="Claim Approval Rate % by Zone")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            z_early = df_f.groupby(["ZONE","EARLY_NON"]).size().reset_index(name="n")
            fig = px.bar(z_early, x="ZONE", y="n", color="EARLY_NON",
                         color_discrete_sequence=[PALETTE[1],PALETTE[3]],
                         barmode="group", title="Early vs Non-Early by Zone")
            fig.update_xaxes(tickangle=40)
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            z_sa = df_f.groupby("ZONE")["SUM_ASSURED"].mean().reset_index()
            z_sa = z_sa.sort_values("SUM_ASSURED", ascending=False)
            fig = px.bar(z_sa, x="ZONE", y="SUM_ASSURED",
                         color_discrete_sequence=[PALETTE[0]],
                         title="Avg Sum Assured by Zone")
            fig.update_xaxes(tickangle=40)
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Age group × state — avg sum assured
        pivot = df_f.groupby(["AGE_GROUP","PI_STATE"], observed=True)["SUM_ASSURED"].mean().unstack().fillna(0)
        top_states = df_f["PI_STATE"].value_counts().head(12).index
        pivot = pivot[pivot.columns.intersection(top_states)]
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.astype(str).tolist(),
            colorscale="Blues",
            text=np.round(pivot.values/1000,0),
            texttemplate="₹%{text}k",
            showscale=True
        ))
        fig.update_layout(
            title="Avg Sum Assured (₹) — Age Group × State (Top 12 States)",
            plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_BG,
            font=dict(color="#9aa5c8"), xaxis=dict(tickangle=35)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Approval rate heatmap — zone × payment mode
        piv2 = df_f.groupby(["ZONE","PAYMENT_MODE"]).apply(
            lambda x: (x["POLICY_STATUS"]=="Approved Death Claim").mean()*100
        ).unstack().fillna(0)
        fig = go.Figure(go.Heatmap(
            z=piv2.values,
            x=piv2.columns.tolist(),
            y=piv2.index.tolist(),
            colorscale="RdYlGn",
            text=np.round(piv2.values,1),
            texttemplate="%{text}%",
            showscale=True, zmin=0, zmax=100
        ))
        fig.update_layout(
            title="Approval Rate % — Zone × Payment Mode",
            plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_BG,
            font=dict(color="#9aa5c8")
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  MODULE 7 — BIAS & FAIRNESS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Bias & Fairness Analysis":
    st.markdown("# ⚖️ Bias & Fairness Analysis")
    st.markdown("Structural and statistical biases in this dataset and their implications for ML models.")

    sev_color = {"HIGH": "#f87171", "MEDIUM": "#fbbf24", "LOW": "#4ade80"}
    m_pct       = (df_f["PI_GENDER"]=="M").mean()*100
    m_appr_rate = (df_f[df_f["PI_GENDER"]=="M"]["POLICY_STATUS"]=="Approved Death Claim").mean()*100
    f_appr_rate = (df_f[df_f["PI_GENDER"]=="F"]["POLICY_STATUS"]=="Approved Death Claim").mean()*100

    biases = [
        {"title":"👨 Severe Gender Imbalance","severity":"HIGH","body":(
            f"Dataset is {m_pct:.1f}% male and {100-m_pct:.1f}% female — a 4.3:1 ratio. "
            f"Male approval rate: {m_appr_rate:.1f}% vs female: {f_appr_rate:.1f}%. "
            "Models will have far more male examples, making female predictions unreliable. "
            "This reflects societal insurance uptake patterns but will be amplified by ML, "
            "disadvantaging female policyholders in automated decisions.")},
        {"title":"🕐 Early Policy Repudiation Bias","severity":"HIGH","body":(
            f"'Early' policies ({(df_f['EARLY_NON']=='EARLY').mean()*100:.1f}% of data) are claims "
            "filed within 2-3 years of issuance — a known fraud indicator. Insurers disproportionately "
            "repudiate these. ML models learn EARLY = higher repudiation risk, potentially penalising "
            f"legitimate early claimants. Early repudiation rate: "
            f"{(repudiated['EARLY_NON']=='EARLY').mean()*100:.1f}% vs "
            f"approved: {(approved['EARLY_NON']=='EARLY').mean()*100:.1f}%.")},
        {"title":"🗺️ Geographic & Zone Bias","severity":"HIGH","body":(
            f"AGENCY zone alone accounts for {(df_f['ZONE']=='AGENCY').mean()*100:.1f}% of records. "
            "Approval rates vary dramatically by zone and state. Models encode geographic privilege — "
            "policyholders in under-represented regions receive less accurate predictions, and those in "
            "zones with historically lower approval rates will be unfairly penalised.")},
        {"title":"💀 Cause of Death Missing Data Bias","severity":"HIGH","body":(
            f"{df_f['REASON_FOR_CLAIM'].isna().sum():,} records "
            f"({df_f['REASON_FOR_CLAIM'].isna().mean()*100:.1f}%) have no stated cause of death. "
            "This is non-random — missing cause correlates with approved natural deaths, while "
            "repudiated claims tend to have stated causes. Imputing or dropping these will "
            "systematically distort the model.")},
        {"title":"🔄 Historical / Insurer Labelling Bias","severity":"HIGH","body":(
            "POLICY_STATUS was determined by insurance investigators — humans with their own biases, "
            "inconsistent standards, and commercial incentives. Repudiation decisions may reflect "
            "fraudulent denials of legitimate claims. Training ML on this risks automating and scaling "
            "discriminatory denial patterns dressed as objective data science.")},
        {"title":"💰 Sum Assured Paradox","severity":"MEDIUM","body":(
            f"Repudiated claims have higher avg sum assured "
            f"(₹{repudiated['SUM_ASSURED'].mean():,.0f}) than approved "
            f"(₹{approved['SUM_ASSURED'].mean():,.0f}). Larger policies attract more scrutiny. "
            "An ML model may learn to flag high-value policies as suspicious, biasing against "
            "legitimate high-value coverage holders.")},
        {"title":"💼 Occupation Representation Bias","severity":"MEDIUM","body":(
            f"{df_f['PI_OCCUPATION'].isna().sum():,} records have no occupation. "
            f"Top 3 occupations account for "
            f"{df_f['PI_OCCUPATION'].value_counts().head(3).sum()/len(df_f)*100:.1f}% of records. "
            "Models over-generalise for rare occupations, and occupation proxies for income and "
            "class — introducing socioeconomic discrimination.")},
        {"title":"📊 Class Imbalance","severity":"MEDIUM","body":(
            f"Approved ({len(approved):,}) outnumber repudiated ({len(repudiated):,}) by "
            f"{len(approved)/max(len(repudiated),1):.1f}:1. An unweighted classifier defaults to "
            "predicting approval (~68% accuracy) while failing on repudiated cases — the "
            "business-critical outcome. SMOTE, class weights, or threshold tuning is essential.")},
        {"title":"📋 Non-Medical Policy Dominance","severity":"MEDIUM","body":(
            f"Non-medical policies: {(df_f['MEDICAL_NONMED']=='NON MEDICAL').mean()*100:.1f}% of data. "
            "Medical policies are heavily under-represented. The model will have poor calibration "
            "for medical policy claims and may conflate medical status with claim risk.")},
        {"title":"💵 Annual Income Zero-Inflation","severity":"LOW","body":(
            f"{(df_f['PI_ANNUAL_INCOME']==0).mean()*100:.1f}% of records show ₹0 annual income — "
            "not genuine zero but 'not disclosed' (retired, homemakers, etc.). Treating as numeric "
            "creates a false signal. Models may incorrectly link zero income to claim probability.")},
    ]

    for b in biases:
        sc2 = sev_color[b["severity"]]
        st.markdown(f"""
        <div class="info-card">
          <div class="info-title">{b['title']}
            <span style="background:{sc2}22;color:{sc2};border:1px solid {sc2};
              padding:2px 10px;border-radius:20px;font-size:0.72rem;font-weight:700;
              letter-spacing:1px;margin-left:10px">{b['severity']}</span>
          </div>
          <div class="info-body">{b['body']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Bias Visualisations</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        g_s = df_f.groupby(["PI_GENDER","POLICY_STATUS"]).size().reset_index(name="n")
        g_s["pct"] = g_s.groupby("PI_GENDER")["n"].transform(lambda x: x/x.sum()*100)
        fig = px.bar(g_s, x="PI_GENDER", y="pct", color="POLICY_STATUS",
                     color_discrete_map=STATUS_MAP, barmode="stack",
                     title="Approval Rate % by Gender")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        en_s = df_f.groupby(["EARLY_NON","POLICY_STATUS"]).size().reset_index(name="n")
        en_s["pct"] = en_s.groupby("EARLY_NON")["n"].transform(lambda x: x/x.sum()*100)
        fig = px.bar(en_s, x="EARLY_NON", y="pct", color="POLICY_STATUS",
                     color_discrete_map=STATUS_MAP, barmode="stack",
                     title="Approval Rate % — Early vs Non-Early")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        state_r = df_f.groupby("PI_STATE").apply(
            lambda x: (x["POLICY_STATUS"]=="Approved Death Claim").mean()*100
        ).reset_index(name="Approval Rate %").sort_values("Approval Rate %")
        fig = px.bar(state_r, x="Approval Rate %", y="PI_STATE", orientation="h",
                     color="Approval Rate %", color_continuous_scale="RdYlGn",
                     title="Approval Rate % by State (geographic bias)")
        fig.update_layout(height=580)
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
    with col2:
        ti = df_f["POLICY_STATUS"].value_counts().reset_index()
        ti.columns = ["Status","Count"]
        fig = px.bar(ti, x="Status", y="Count", color="Status",
                     color_discrete_map=STATUS_MAP,
                     title=f"Class Imbalance — {len(approved):,} vs {len(repudiated):,}")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Mitigation Recommendations</div>', unsafe_allow_html=True)
    for title, body in [
        ("Pre-processing", "Apply SMOTE or class weights for the 2.1:1 imbalance. Treat PI_ANNUAL_INCOME=0 as a separate 'Not Disclosed' category, not numeric zero. Group rare ZONE/STATE values. Flag missing REASON_FOR_CLAIM explicitly rather than dropping."),
        ("Feature engineering", "Create interaction features: EARLY × MEDICAL, AGE × SA_TIER. Group rare occupations (Agricultural, Professional, Business, Service, Other). Derive a policy-age-at-claim feature."),
        ("In-processing", "Apply fairness constraints (demographic parity / equalised odds) on PI_GENDER and PI_STATE. Use adversarial debiasing to reduce geographic proxy reliance."),
        ("Post-processing", "Audit for disparate impact: approval rates by gender, state, and age group should not deviate by more than 20% (80% rule). Apply threshold calibration per demographic group."),
        ("Governance", "Never deploy as sole decision-maker. Flag low-confidence predictions for human review. Conduct quarterly bias audits. Document feature drivers per repudiation decision for IRDAI auditability."),
    ]:
        st.markdown(f"""
        <div class="info-card">
          <div class="info-title" style="color:#7c9ef8">✅ {title}</div>
          <div class="info-body">{body}</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  MODULE 8 — DEEP DRILL-DOWN ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Deep Drill-Down Analysis":
    try:
        import xgboost as xgb
    except ImportError:
        st.error("Install xgboost: `pip install xgboost`"); st.stop()
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics         import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
    import hashlib

    st.markdown("# 🔍 Deep Drill-Down Analysis")

    tab1, tab2, tab3 = st.tabs([
        "⚡ XGBoost Model", "💀 Cause of Death", "📋 Claims Deep Dive"
    ])

    # ── Tab 1: XGBoost ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### ⚡ XGBoost Claim Repudiation Predictor")

        FEAT_COLS = ["PI_AGE","SUM_ASSURED","GENDER_BIN","EARLY_BIN","MEDICAL_BIN",
                     "PM_Half-Yly","PM_Monthly","PM_Quarterly","PM_Single"]
        X = df_f[[c for c in FEAT_COLS if c in df_f.columns]].fillna(0)
        y = df_f["CLAIMED"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        col1, col2, col3 = st.columns(3)
        n_est = col1.slider("n_estimators",  50, 500, 200, 50)
        m_dep = col2.slider("max_depth",      2,  10,   4)
        lr    = col3.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)

        h = hashlib.md5(X_train.values.tobytes()).hexdigest()

        @st.cache_data
        def run_xgb(hv, Xtr, ytr, Xte, yte, ne, md, lrv):
            spw = (ytr==0).sum() / (ytr==1).sum()
            m   = xgb.XGBClassifier(
                n_estimators=ne, max_depth=md, learning_rate=lrv,
                scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, eval_metric="logloss")
            m.fit(Xtr, ytr, eval_set=[(Xtr,ytr),(Xte,yte)], verbose=False)
            preds = m.predict(Xte)
            proba = m.predict_proba(Xte)[:,1]
            evals = m.evals_result()
            cv    = cross_val_score(m, Xtr, ytr, cv=5, scoring="roc_auc")
            return m, preds, proba, evals, cv

        model, preds, proba, evals, cv_scores = run_xgb(
            h, X_train, y_train, X_test, y_test, n_est, m_dep, lr)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        for col, (lbl, val) in zip(st.columns(4), [
            ("Accuracy", f"{acc*100:.1f}%"),
            ("ROC-AUC",  f"{auc:.4f}"),
            ("CV AUC",   f"{cv_scores.mean():.4f}"),
            ("CV Std",   f"±{cv_scores.std():.4f}"),
        ]):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=evals["validation_0"]["logloss"], name="Train Loss",
                                 line=dict(color=PALETTE[0])))
        fig.add_trace(go.Scatter(y=evals["validation_1"]["logloss"], name="Test Loss",
                                 line=dict(color=PALETTE[3])))
        fig.update_layout(title="Log Loss over Boosting Rounds",
                          xaxis_title="Round", yaxis_title="Log Loss")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fi = pd.DataFrame({
                "Feature":    X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance")
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[PALETTE[1]], title="Feature Importance")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            fpr, tpr, _ = roc_curve(y_test, proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc:.3f}",
                                     line=dict(color=PALETTE[0], width=2)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                                     line=dict(dash="dash", color=PALETTE[3]), name="Random"))
            fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(5)],
                             y=cv_scores, marker_color=PALETTE[:5]))
        fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color=PALETTE[3],
                      annotation_text=f"Mean = {cv_scores.mean():.4f}")
        fig.update_layout(title="5-Fold CV AUC", yaxis_title="AUC")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Per-Sample Feature Contributions**")
        sample_idx = st.slider("Select test sample", 0, len(X_test)-1, 0)
        base_p = model.predict_proba(X_test)[sample_idx][1]
        sample = X_test.iloc[[sample_idx]]
        contribs = {}
        for feat in X.columns:
            mod = sample.copy(); mod[feat] = X_train[feat].mean()
            contribs[feat] = base_p - model.predict_proba(mod)[0][1]
        cdf = pd.DataFrame.from_dict(contribs, orient="index",
                                     columns=["Contribution"]).sort_values("Contribution")
        pred_lbl = "✅ Approved" if model.predict(X_test)[sample_idx]==1 else "❌ Repudiated"
        fig = px.bar(cdf, x="Contribution", y=cdf.index, orientation="h",
                     color="Contribution",
                     color_continuous_scale=["#f87171","#1e2235","#4ade80"],
                     title=f"Sample #{sample_idx} — {pred_lbl} | P(Approved)={base_p*100:.1f}%")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Cause of Death ───────────────────────────────────────────────────
    with tab2:
        st.markdown("### 💀 Cause of Death Deep Dive")
        df_r = df_f[df_f["REASON_FOR_CLAIM"].notna()].copy()

        col1, col2 = st.columns(2)
        with col1:
            rc = df_r["REASON_FOR_CLAIM"].value_counts().reset_index()
            rc.columns = ["Cause","Count"]
            fig = px.bar(rc.head(15), x="Count", y="Cause", orientation="h",
                         color="Count", color_continuous_scale="Blues",
                         title="Top 15 Causes of Death")
            fig.update_layout(height=480)
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.treemap(rc, path=["Cause"], values="Count",
                             color="Count", color_continuous_scale="Blues",
                             title="Cause of Death — Treemap")
            fig.update_layout(paper_bgcolor=PLOTLY_BG, font=dict(color="#e8eaf6"))
            st.plotly_chart(fig, use_container_width=True)

        rs    = df_r.groupby(["REASON_FOR_CLAIM","POLICY_STATUS"]).size().reset_index(name="n")
        top15 = df_r["REASON_FOR_CLAIM"].value_counts().head(15).index
        rs15  = rs[rs["REASON_FOR_CLAIM"].isin(top15)]
        fig = px.bar(rs15, x="n", y="REASON_FOR_CLAIM", color="POLICY_STATUS",
                     color_discrete_map=STATUS_MAP, barmode="stack", orientation="h",
                     title="Claim Status by Cause (Top 15)")
        fig.update_layout(height=500)
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        appr = df_r.groupby("REASON_FOR_CLAIM").apply(
            lambda x: (x["POLICY_STATUS"]=="Approved Death Claim").mean()*100
        ).reset_index(name="Approval Rate %")
        appr = appr[appr["REASON_FOR_CLAIM"].isin(top15)].sort_values("Approval Rate %")
        fig = px.bar(appr, x="Approval Rate %", y="REASON_FOR_CLAIM", orientation="h",
                     color="Approval Rate %", color_continuous_scale="RdYlGn",
                     title="Approval Rate % by Cause of Death")
        dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Select a specific cause for detailed breakdown**")
        selected_cause = st.selectbox("Cause of Death",
            df_r["REASON_FOR_CLAIM"].value_counts().head(20).index.tolist())
        sub = df_r[df_r["REASON_FOR_CLAIM"] == selected_cause]

        for col, (lbl, val, clr) in zip(st.columns(4), [
            ("Total",      f"{len(sub):,}",                                          "#7c9ef8"),
            ("Approved",   f"{(sub['POLICY_STATUS']=='Approved Death Claim').sum():,}", "#4ade80"),
            ("Repudiated", f"{(sub['POLICY_STATUS']=='Repudiate Death').sum():,}",    "#f87171"),
            ("Avg Age",    f"{sub['PI_AGE'].mean():.1f}",                             "#fb923c"),
        ]):
            col.markdown(f"""<div class="metric-card">
              <div class="metric-value" style="color:{clr}">{val}</div>
              <div class="metric-label">{lbl}</div></div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(sub, x="PI_AGE", color="POLICY_STATUS",
                               color_discrete_map=STATUS_MAP, nbins=20,
                               title=f"Age Distribution — {selected_cause}")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(sub, x="POLICY_STATUS", y="SUM_ASSURED",
                         color="POLICY_STATUS", color_discrete_map=STATUS_MAP,
                         log_y=True, title=f"Sum Assured — {selected_cause}")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Claims deep dive ─────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📋 Claims Deep Dive — Approved vs Repudiated")

        col1, col2 = st.columns(2)
        with col1:
            age_s = df_f.groupby(["AGE_GROUP","POLICY_STATUS"], observed=True).size().reset_index(name="n")
            fig = px.bar(age_s, x="AGE_GROUP", y="n", color="POLICY_STATUS",
                         color_discrete_map=STATUS_MAP, barmode="group",
                         title="Claims by Age Group")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            pm_r = df_f.groupby("PAYMENT_MODE").apply(
                lambda x: (x["POLICY_STATUS"]=="Approved Death Claim").mean()*100
            ).reset_index(name="Approval Rate %")
            fig = px.bar(pm_r, x="PAYMENT_MODE", y="Approval Rate %",
                         color_discrete_sequence=[PALETTE[0]],
                         title="Approval Rate % by Payment Mode")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            asg = df_f.groupby(["PI_GENDER","POLICY_STATUS"])["SUM_ASSURED"].mean().reset_index()
            fig = px.bar(asg, x="PI_GENDER", y="SUM_ASSURED", color="POLICY_STATUS",
                         color_discrete_map=STATUS_MAP, barmode="group",
                         title="Avg Sum Assured: Gender × Status")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)
        with col2:
            occ_r = df_f.groupby("PI_OCCUPATION").apply(
                lambda x: (x["POLICY_STATUS"]=="Approved Death Claim").mean()*100
            ).reset_index(name="Approval Rate %")
            occ_r = occ_r[occ_r.index.isin(
                df_f["PI_OCCUPATION"].value_counts().head(12).index
            ) | occ_r["PI_OCCUPATION"].isin(
                df_f["PI_OCCUPATION"].value_counts().head(12).index
            )].sort_values("Approval Rate %")
            fig = px.bar(occ_r.head(12), x="Approval Rate %", y="PI_OCCUPATION",
                         orientation="h", color="Approval Rate %",
                         color_continuous_scale="RdYlGn",
                         title="Approval Rate % by Occupation (Top 12)")
            dark_fig(fig); st.plotly_chart(fig, use_container_width=True)

        with st.expander("📄 View Approved Claims"):
            st.dataframe(approved[["POLICY_NO","PI_GENDER","PI_AGE","PI_STATE","ZONE",
                                    "SUM_ASSURED","PAYMENT_MODE","EARLY_NON",
                                    "REASON_FOR_CLAIM","POLICY_STATUS"]].reset_index(drop=True))
        with st.expander("📄 View Repudiated Claims"):
            st.dataframe(repudiated[["POLICY_NO","PI_GENDER","PI_AGE","PI_STATE","ZONE",
                                      "SUM_ASSURED","PAYMENT_MODE","EARLY_NON",
                                      "REASON_FOR_CLAIM","POLICY_STATUS"]].reset_index(drop=True))


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5080;font-size:0.8rem;'>"
    "Insurance Claims Analytics Dashboard • Streamlit</p>",
    unsafe_allow_html=True
)
