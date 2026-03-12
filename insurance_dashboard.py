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
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Analytics Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
  
  .main { background: #0d0f1a; }
  
  .metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #252840 100%);
    border: 1px solid #2e3158;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 10px;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #7c9ef8; font-family: 'JetBrains Mono', monospace; }
  .metric-label { font-size: 0.8rem; color: #8b9cc8; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }
  
  .section-header {
    font-size: 1.4rem; font-weight: 700; color: #e8eaf6;
    border-left: 4px solid #7c9ef8; padding-left: 12px;
    margin: 24px 0 16px 0;
  }
  
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 1px;
  }
  .badge-blue  { background: #1e3a8a22; color: #7c9ef8; border: 1px solid #3b5fd8; }
  .badge-green { background: #14532d22; color: #4ade80; border: 1px solid #16a34a; }
  .badge-red   { background: #7f1d1d22; color: #f87171; border: 1px solid #dc2626; }
  
  div[data-testid="stSidebar"] { background: #10121f; border-right: 1px solid #1e2235; }
  div[data-testid="stSidebar"] .stRadio label { color: #c4cbe8 !important; }
  
  h1 { color: #e8eaf6 !important; }
  h2, h3 { color: #c4cbe8 !important; }
  p, li { color: #9aa5c8 !important; }
  
  .stSelectbox label, .stSlider label, .stMultiSelect label { color: #9aa5c8 !important; }
  
  .highlight-box {
    background: #151829; border: 1px solid #2a2f52; border-radius: 8px;
    padding: 16px; margin: 8px 0;
  }
  
  code { font-family: 'JetBrains Mono', monospace; background: #1e2235; color: #7c9ef8; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE   = ["#7c9ef8","#4ade80","#f472b6","#fb923c","#a78bfa","#34d399","#fbbf24","#60a5fa"]
PLOTLY_BG = "#0d0f1a"
PLOTLY_PAPER = "#0d0f1a"
GRID_COLOR = "#1e2235"

def dark_fig(fig):
    fig.update_layout(
        plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_PAPER,
        font=dict(color="#9aa5c8", family="Space Grotesk"),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(t=40, b=40, l=40, r=20),
        legend=dict(bgcolor="#151829", bordercolor="#2a2f52", borderwidth=1),
    )
    return fig

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path="InsuranceLR.csv"):
    df = pd.read_csv(path)
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    df["smoker_bin"]   = (df["smoker"] == "yes").astype(int)
    df["sex_bin"]      = (df["sex"] == "male").astype(int)
    df["bmi_category"] = pd.cut(df["bmi"],
        bins=[0,18.5,25,30,np.inf],
        labels=["Underweight","Normal","Overweight","Obese"])
    df["age_group"] = pd.cut(df["age"],
        bins=[17,25,35,45,55,65],
        labels=["18-25","26-35","36-45","46-55","56-64"])
    region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
    df = pd.concat([df, region_dummies], axis=1)
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Insurance Hub")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV", type="csv", help="Upload your insurance dataset")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "index" in df.columns:
            df = df.drop(columns=["index"])
        df["smoker_bin"]   = (df["smoker"] == "yes").astype(int)
        df["sex_bin"]      = (df["sex"] == "male").astype(int)
        df["bmi_category"] = pd.cut(df["bmi"],
            bins=[0,18.5,25,30,np.inf],
            labels=["Underweight","Normal","Overweight","Obese"])
        df["age_group"] = pd.cut(df["age"],
            bins=[17,25,35,45,55,65],
            labels=["18-25","26-35","36-45","46-55","56-64"])
        region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
        df = pd.concat([df, region_dummies], axis=1)
        st.success("Custom dataset loaded!")
    else:
        df = load_data()

    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview",
        "🔍 Exploratory Analysis",
        "🤖 ML — Prediction",
        "🧩 K-Means Clustering",
        "🛒 Apriori Rules",
        "⚡ XGBoost Analysis",
    ])
    st.markdown("---")
    st.markdown("### Filters")
    sel_regions = st.multiselect("Region", options=df["region"].unique().tolist(), default=df["region"].unique().tolist())
    sel_smoker  = st.multiselect("Smoker",  options=["yes","no"], default=["yes","no"])
    age_range   = st.slider("Age range", int(df["age"].min()), int(df["age"].max()), (18, 64))
    df_f = df[
        df["region"].isin(sel_regions) &
        df["smoker"].isin(sel_smoker) &
        df["age"].between(*age_range)
    ]
    st.caption(f"Showing **{len(df_f):,}** records")

# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Insurance Analytics Hub")
    st.markdown("**Comprehensive data analytics for insurance charge prediction and profiling.**")

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("Total Records", f"{len(df_f):,}"),
        ("Avg Charges", f"${df_f['charges'].mean():,.0f}"),
        ("Avg Age", f"{df_f['age'].mean():.1f} yrs"),
        ("Avg BMI", f"{df_f['bmi'].mean():.1f}"),
        ("Smoker %", f"{df_f['smoker_bin'].mean()*100:.1f}%"),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4,c5], metrics):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Charges Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df_f, x="charges", nbins=60, color_discrete_sequence=[PALETTE[0]],
                           title="Distribution of Insurance Charges")
        fig.update_traces(marker_line_color="#0d0f1a", marker_line_width=0.5)
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df_f, x="smoker", y="charges", color="smoker",
                     color_discrete_map={"yes": PALETTE[3], "no": PALETTE[0]},
                     title="Charges by Smoking Status")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Feature Relationships</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(df_f, x="age", y="charges", color="smoker", size="bmi",
                         color_discrete_map={"yes": PALETTE[3], "no": PALETTE[0]},
                         hover_data=["region","children"], title="Age vs Charges (size=BMI)")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_region = df_f.groupby("region")["charges"].mean().reset_index()
        fig = px.bar(avg_region, x="region", y="charges", color="region",
                     color_discrete_sequence=PALETTE, title="Average Charges by Region")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ["age","bmi","children","charges","smoker_bin","sex_bin"]
    corr = df_f[num_cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="Blues", zmin=-1, zmax=1,
        text=np.round(corr.values,2), texttemplate="%{text}",
        showscale=True
    ))
    fig.update_layout(title="Correlation Matrix", plot_bgcolor=PLOTLY_BG, paper_bgcolor=PLOTLY_PAPER,
                      font=dict(color="#9aa5c8"), margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORATORY ANALYSIS
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
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            region_cnt = df_f["region"].value_counts().reset_index()
            fig = px.bar(region_cnt, x="region", y="count", color="region",
                         color_discrete_sequence=PALETTE, title="Records per Region")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_f, x="age", color="sex",
                               color_discrete_map={"male": PALETTE[0], "female": PALETTE[2]},
                               barmode="overlay", nbins=30, title="Age Distribution by Sex")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            children_cnt = df_f["children"].value_counts().reset_index().sort_values("children")
            fig = px.bar(children_cnt, x="children", y="count",
                         color_discrete_sequence=[PALETTE[1]], title="Number of Children")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_f, x="bmi", color="bmi_category",
                               color_discrete_sequence=PALETTE, nbins=50, title="BMI Distribution")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            avg_bmi = df_f.groupby(["age_group","smoker"])["bmi"].mean().reset_index()
            fig = px.bar(avg_bmi, x="age_group", y="bmi", color="smoker", barmode="group",
                         color_discrete_map={"yes": PALETTE[3], "no": PALETTE[0]},
                         title="Avg BMI by Age Group & Smoker Status")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df_f, x="bmi_category", y="charges", color="smoker",
                     color_discrete_map={"yes": PALETTE[3], "no": PALETTE[0]},
                     title="Charges by BMI Category & Smoking Status")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.violin(df_f, x="region", y="charges", color="region",
                            color_discrete_sequence=PALETTE, box=True, title="Charge Distribution per Region")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            pivot = df_f.groupby(["age_group","region"])["charges"].mean().reset_index()
            fig = px.line(pivot, x="age_group", y="charges", color="region",
                          color_discrete_sequence=PALETTE, markers=True,
                          title="Avg Charges by Age Group & Region")
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Top / Bottom 10 %
        col1, col2 = st.columns(2)
        top10  = df_f.nlargest(int(len(df_f)*0.1), "charges")
        bot10  = df_f.nsmallest(int(len(df_f)*0.1), "charges")
        with col1:
            st.markdown("**Top 10% Earner Profile (highest charges)**")
            st.dataframe(top10.describe()[["age","bmi","charges"]].round(2))
        with col2:
            st.markdown("**Bottom 10% Profile (lowest charges)**")
            st.dataframe(bot10.describe()[["age","bmi","charges"]].round(2))

    with tab4:
        st.markdown("**Scatter Matrix — numeric features coloured by smoker status**")
        fig = px.scatter_matrix(df_f, dimensions=["age","bmi","children","charges"],
                                color="smoker",
                                color_discrete_map={"yes": PALETTE[3], "no": PALETTE[0]},
                                title="Pairplot")
        fig.update_traces(marker=dict(size=3, opacity=0.6))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ML PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML — Prediction":
    from sklearn.linear_model     import LinearRegression, Ridge, Lasso
    from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection  import train_test_split, cross_val_score
    from sklearn.preprocessing    import StandardScaler
    from sklearn.metrics          import mean_squared_error, r2_score, mean_absolute_error

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
    def train_models(Xtr, ytr, Xte, yte):
        models = {
            "Linear Reg":   LinearRegression(),
            "Ridge":        Ridge(alpha=10),
            "Lasso":        Lasso(alpha=10),
            "Random Forest":RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boost":GradientBoostingRegressor(n_estimators=200, random_state=42),
        }
        results = {}
        for name, m in models.items():
            m.fit(Xtr, ytr)
            preds = m.predict(Xte)
            results[name] = {
                "model": m,
                "preds": preds,
                "R2":    r2_score(yte, preds),
                "RMSE":  np.sqrt(mean_squared_error(yte, preds)),
                "MAE":   mean_absolute_error(yte, preds),
            }
        return results

    results = train_models(X_train_s, y_train, X_test_s, y_test)

    # Model comparison bar chart
    comp = pd.DataFrame({k: {"R²": v["R2"], "RMSE": v["RMSE"], "MAE": v["MAE"]} for k,v in results.items()}).T.reset_index()
    comp.columns = ["Model","R²","RMSE","MAE"]

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comp, x="Model", y="R²", color="Model",
                     color_discrete_sequence=PALETTE, title="R² Score (higher = better)")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(comp, x="Model", y="RMSE", color="Model",
                     color_discrete_sequence=PALETTE, title="RMSE (lower = better)")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comp.set_index("Model").round(2))

    # Residuals
    st.markdown('<div class="section-header">Predicted vs Actual & Residuals</div>', unsafe_allow_html=True)
    sel_model = st.selectbox("Select model", list(results.keys()), index=3)
    preds = results[sel_model]["preds"]
    residuals = y_test.values - preds

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test.values, y=preds, mode="markers",
                                 marker=dict(color=PALETTE[0], opacity=0.6, size=5),
                                 name="Predictions"))
        mn, mx = y_test.min(), y_test.max()
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                 line=dict(color=PALETTE[3], dash="dash"), name="Perfect fit"))
        fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual", yaxis_title="Predicted")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(x=residuals, nbins=60, color_discrete_sequence=[PALETTE[2]],
                           title="Residual Distribution")
        fig.add_vline(x=0, line_dash="dash", line_color=PALETTE[3])
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance (if tree model)
    if hasattr(results[sel_model]["model"], "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": results[sel_model]["model"].feature_importances_
        }).sort_values("Importance", ascending=True)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=[PALETTE[1]], title="Feature Importances")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Interactive prediction
    st.markdown('<div class="section-header">🎯 Live Prediction</div>', unsafe_allow_html=True)
    co1, co2, co3, co4 = st.columns(4)
    p_age      = co1.slider("Age", 18, 65, 35)
    p_bmi      = co2.slider("BMI", 15.0, 55.0, 30.0)
    p_children = co3.slider("Children", 0, 5, 1)
    p_smoker   = co4.selectbox("Smoker", ["No","Yes"])
    co5, co6, co7 = st.columns(3)
    p_sex    = co5.selectbox("Sex", ["Male","Female"])
    p_region = co6.selectbox("Region", ["northeast","northwest","southeast","southwest"])

    inp = pd.DataFrame([{
        "age": p_age, "bmi": p_bmi, "children": p_children,
        "smoker_bin": 1 if p_smoker=="Yes" else 0,
        "sex_bin": 1 if p_sex=="Male" else 0,
        "region_northwest": 1 if p_region=="northwest" else 0,
        "region_southeast": 1 if p_region=="southeast" else 0,
        "region_southwest": 1 if p_region=="southwest" else 0,
    }])
    inp_sc = sc.transform(inp[[c for c in FEAT_COLS if c in inp.columns]])
    pred_val = results[sel_model]["model"].predict(inp_sc)[0]
    st.markdown(f"""
    <div class="highlight-box" style="text-align:center">
      <div style="font-size:0.8rem;color:#8b9cc8;text-transform:uppercase;letter-spacing:2px">
        Predicted Annual Insurance Charge ({sel_model})
      </div>
      <div style="font-size:2.8rem;font-weight:700;color:#4ade80;font-family:'JetBrains Mono',monospace">
        ${pred_val:,.2f}
      </div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — K-MEANS CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🧩 K-Means Clustering":
    from sklearn.cluster        import KMeans
    from sklearn.preprocessing  import StandardScaler
    from sklearn.decomposition  import PCA
    from sklearn.metrics        import silhouette_score

    st.markdown("# 🧩 K-Means Clustering")
    st.markdown("Segment policyholders into distinct risk/demographic profiles.")

    feat_options = ["age","bmi","children","charges","smoker_bin","sex_bin"]
    sel_feats = st.multiselect("Features for clustering", feat_options,
                               default=["age","bmi","charges","smoker_bin"])
    k = st.slider("Number of Clusters (K)", 2, 10, 4)

    Xc = df_f[sel_feats].dropna()
    sc = StandardScaler()
    Xcs = sc.fit_transform(Xc)

    @st.cache_data
    def run_kmeans(data_hash, Xcs_arr, k_val):
        km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
        labels = km.fit_predict(Xcs_arr)
        sil = silhouette_score(Xcs_arr, labels)
        return labels, km.inertia_, sil

    import hashlib
    data_hash = hashlib.md5(Xcs.tobytes()).hexdigest()
    labels, inertia, sil_score = run_kmeans(data_hash, Xcs, k)

    Xc = Xc.copy()
    Xc["Cluster"] = labels.astype(str)

    # Elbow
    st.markdown('<div class="section-header">Elbow Curve & Silhouette Score</div>', unsafe_allow_html=True)
    @st.cache_data
    def elbow(hash_val, Xcs_arr):
        inertias, sils = [], []
        for ki in range(2,11):
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            lbl = km.fit_predict(Xcs_arr)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(Xcs_arr, lbl))
        return inertias, sils

    inertias, sils = elbow(data_hash, Xcs)
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,11)), y=inertias, mode="lines+markers",
                                 marker=dict(color=PALETTE[0], size=8),
                                 line=dict(color=PALETTE[0])))
        fig.add_vline(x=k, line_dash="dash", line_color=PALETTE[3])
        fig.update_layout(title="Elbow Curve", xaxis_title="K", yaxis_title="Inertia")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,11)), y=sils, mode="lines+markers",
                                 marker=dict(color=PALETTE[1], size=8),
                                 line=dict(color=PALETTE[1])))
        fig.add_vline(x=k, line_dash="dash", line_color=PALETTE[3])
        fig.update_layout(title="Silhouette Score", xaxis_title="K", yaxis_title="Score")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f'<div class="section-header">Cluster Visualisation (K={k} | Silhouette: {sil_score:.3f})</div>', unsafe_allow_html=True)

    # PCA 2D
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(Xcs)
    pca_df = pd.DataFrame(pca_coords, columns=["PC1","PC2"])
    pca_df["Cluster"] = labels.astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         color_discrete_sequence=PALETTE,
                         title=f"PCA Projection — {k} Clusters",
                         opacity=0.7)
        fig.update_traces(marker=dict(size=5))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        profile = Xc.groupby("Cluster")[sel_feats].mean().round(2).reset_index()
        fig = px.bar(profile.melt(id_vars="Cluster"), x="variable", y="value",
                     color="Cluster", barmode="group",
                     color_discrete_sequence=PALETTE, title="Cluster Profiles (mean values)")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Cluster Summary Statistics**")
    st.dataframe(Xc.groupby("Cluster")[sel_feats].describe().round(2))


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — APRIORI ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Apriori Rules":
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing    import TransactionEncoder
    except ImportError:
        st.error("Please install mlxtend: `pip install mlxtend`")
        st.stop()

    st.markdown("# 🛒 Apriori Association Rules")
    st.markdown("Discover co-occurrence patterns among policyholder attributes.")

    # Discretise continuous features into bins
    df_ap = df_f.copy()
    df_ap["age_cat"]     = pd.cut(df_ap["age"],   bins=[17,30,45,65], labels=["Young","Mid","Senior"])
    df_ap["bmi_cat"]     = pd.cut(df_ap["bmi"],   bins=[0,25,30,55],  labels=["Normal","Overweight","Obese"])
    df_ap["charge_cat"]  = pd.cut(df_ap["charges"],
                                  bins=[0,5000,15000,70000], labels=["Low","Medium","High"])
    df_ap["children_cat"]= df_ap["children"].apply(lambda x: "NoChild" if x==0 else ("1Child" if x==1 else "2+Children"))

    bin_cols = ["age_cat","bmi_cat","charge_cat","smoker","sex","region","children_cat"]
    # one-hot encode
    ohe_df = pd.get_dummies(df_ap[bin_cols].astype(str))
    ohe_bool = ohe_df.astype(bool)

    col1, col2, col3 = st.columns(3)
    min_support    = col1.slider("Min Support",    0.05, 0.5, 0.15, 0.01)
    min_confidence = col2.slider("Min Confidence", 0.3,  0.9, 0.5,  0.05)
    min_lift       = col3.slider("Min Lift",        1.0,  5.0, 1.2,  0.1)

    @st.cache_data
    def run_apriori(hash_val, ohe_bool, min_sup, min_conf, min_lift_val):
        freq = apriori(ohe_bool, min_support=min_sup, use_colnames=True, max_len=4)
        if freq.empty:
            return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift_val]
        return freq, rules

    import hashlib
    h = hashlib.md5(ohe_bool.values.tobytes()).hexdigest()
    freq_items, rules = run_apriori(h, ohe_bool, min_support, min_confidence, min_lift)

    if rules.empty:
        st.warning("No rules found with current thresholds. Try lowering support/confidence.")
    else:
        st.success(f"Found **{len(rules)}** association rules from **{len(freq_items)}** frequent itemsets.")

        col1, col2 = st.columns(2)
        with col1:
            top_rules = rules.nlargest(20, "lift")[["antecedents","consequents","support","confidence","lift"]].copy()
            top_rules["antecedents"] = top_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
            top_rules["consequents"] = top_rules["consequents"].apply(lambda x: ", ".join(list(x)))
            st.markdown("**Top 20 Rules by Lift**")
            st.dataframe(top_rules.round(4), use_container_width=True)

        with col2:
            fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift",
                             color_continuous_scale="Blues",
                             title="Support vs Confidence (size & color = Lift)",
                             hover_data={"lift": True})
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Frequent Itemsets</div>', unsafe_allow_html=True)
        freq_show = freq_items.copy()
        freq_show["itemsets"] = freq_show["itemsets"].apply(lambda x: ", ".join(list(x)))
        freq_show = freq_show.sort_values("support", ascending=False).head(30)
        fig = px.bar(freq_show, x="support", y="itemsets", orientation="h",
                     color_discrete_sequence=[PALETTE[0]], title="Top 30 Frequent Itemsets")
        fig.update_layout(height=600, yaxis=dict(tickfont=dict(size=10)))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — XGBOOST
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚡ XGBoost Analysis":
    try:
        import xgboost as xgb
    except ImportError:
        st.error("Please install xgboost: `pip install xgboost`")
        st.stop()

    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error

    st.markdown("# ⚡ XGBoost Analysis")
    st.markdown("Gradient-boosted decision trees for high-accuracy charge prediction.")

    FEAT_COLS = ["age","bmi","children","smoker_bin","sex_bin",
                 "region_northwest","region_southeast","region_southwest"]
    X = df_f[[c for c in FEAT_COLS if c in df_f.columns]].copy()
    y = df_f["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    col1, col2, col3 = st.columns(3)
    n_estimators = col1.slider("n_estimators",  50, 500, 200, 50)
    max_depth     = col2.slider("max_depth",      2,  10,   5)
    learning_rate = col3.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)

    @st.cache_data
    def run_xgb(Xtr, ytr, Xte, yte, ne, md, lr):
        model = xgb.XGBRegressor(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
        model.fit(Xtr, ytr,
                  eval_set=[(Xtr, ytr),(Xte, yte)],
                  verbose=False)
        preds = model.predict(Xte)
        evals = model.evals_result()
        cv_scores = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2")
        return model, preds, evals, cv_scores

    model, preds, evals, cv_scores = run_xgb(
        X_train, y_train, X_test, y_test,
        n_estimators, max_depth, learning_rate
    )

    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)

    c1,c2,c3,c4 = st.columns(4)
    for col, label, val in zip([c1,c2,c3,c4],
        ["R² Score","RMSE","MAE","CV R² (mean)"],
        [f"{r2:.4f}", f"${rmse:,.0f}", f"${mae:,.0f}", f"{cv_scores.mean():.4f}"]):
        col.markdown(f"""<div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
    train_rmse = np.sqrt(evals["validation_0"]["rmse"])
    test_rmse  = np.sqrt(evals["validation_1"]["rmse"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_rmse, name="Train RMSE", line=dict(color=PALETTE[0])))
    fig.add_trace(go.Scatter(y=test_rmse,  name="Test RMSE",  line=dict(color=PALETTE[3])))
    fig.update_layout(title="RMSE over Boosting Rounds", xaxis_title="Round", yaxis_title="RMSE")
    dark_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    fi = pd.DataFrame({
        "Feature": X.columns,
        "Importance (gain)": model.feature_importances_
    }).sort_values("Importance (gain)", ascending=True)

    with col1:
        fig = px.bar(fi, x="Importance (gain)", y="Feature", orientation="h",
                     color_discrete_sequence=[PALETTE[1]], title="Feature Importance (Gain)")
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        residuals = y_test.values - preds
        fig = px.scatter(x=preds, y=residuals, color_discrete_sequence=[PALETTE[2]],
                         title="Residuals vs Fitted",
                         labels={"x":"Fitted Values","y":"Residuals"})
        fig.add_hline(y=0, line_dash="dash", line_color=PALETTE[3])
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Cross-validation box
    st.markdown('<div class="section-header">5-Fold Cross Validation R²</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(5)], y=cv_scores,
                         marker_color=PALETTE[:5]))
    fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color=PALETTE[3],
                  annotation_text=f"Mean = {cv_scores.mean():.4f}")
    fig.update_layout(title="Cross Validation R² per Fold", yaxis_title="R²")
    dark_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    # SHAP-style manual explanation (no shap library needed)
    st.markdown('<div class="section-header">Prediction Explained — Feature Contribution</div>', unsafe_allow_html=True)
    st.markdown("Approximate each feature's marginal contribution to a single prediction by ablation.")

    sample_idx = st.slider("Select test sample", 0, len(X_test)-1, 0)
    sample = X_test.iloc[[sample_idx]]
    base_pred = model.predict(X_test)[sample_idx]

    contributions = {}
    for feat in X.columns:
        modified = sample.copy()
        modified[feat] = X_train[feat].mean()
        contributions[feat] = base_pred - model.predict(modified)[0]

    contrib_df = pd.DataFrame.from_dict(contributions, orient="index", columns=["Contribution"]).sort_values("Contribution")
    fig = px.bar(contrib_df, x="Contribution", y=contrib_df.index, orientation="h",
                 color="Contribution", color_continuous_scale=["#f87171","#1e2235","#4ade80"],
                 title=f"Feature Contributions for Sample #{sample_idx} (Predicted: ${base_pred:,.0f})")
    dark_fig(fig)
    st.plotly_chart(fig, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5080;font-size:0.8rem;'>Insurance Analytics Hub • Built with Streamlit</p>",
    unsafe_allow_html=True
)
