import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Churnpredict.ai", page_icon="📉", layout="wide")
st.title("📉 Churnpredict.ai – Churn Probability Predictor")


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def compute_slope(x):
    """Compute linear slope for last 3 months."""
    if len(x) < 3:
        return 0.0
    y = x.values
    xx = np.arange(len(y))
    return np.polyfit(xx, y, 1)[0]


def build_soft_churn_labels(df):
    """
    Create:
      - soft_churn
      - future_soft_churn_2m
      - contract_end_churn
    """
    df = df.sort_values(["company_id", "vendor_id", "date"]).copy()

    # Contract-end churn
    rel_end = df.groupby(["company_id", "vendor_id"])["date"].max().reset_index()
    rel_end["contract_end_churn"] = 1
    df = df.merge(rel_end, on=["company_id", "vendor_id", "date"], how="left")
    df["contract_end_churn"] = df["contract_end_churn"].fillna(0)

    # 2-month rolling mean on lagged spend
    df["roll2_tmp"] = (
        df.groupby(["company_id", "vendor_id"])["monthly_spend"]
        .shift(1).rolling(2).mean()
    )
    df["pct_change_vs_roll2"] = (
        (df["monthly_spend"] - df["roll2_tmp"]) / df["roll2_tmp"]
    )
    df["drop_flag"] = df["pct_change_vs_roll2"] <= -0.40

    df["soft_churn_signal"] = (
        df.groupby(["company_id", "vendor_id"])["drop_flag"]
        .rolling(2).sum().reset_index(level=[0,1], drop=True)
    )

    df["soft_churn"] = (
        (df["soft_churn_signal"] >= 2) &
        (df["contract_end_churn"] == 0)
    ).astype(int)

    df["future_soft_churn_2m"] = (
        df.groupby(["company_id", "vendor_id"])["soft_churn"].shift(-2)
    )
    df["future_soft_churn_2m"] = df["future_soft_churn_2m"].fillna(0).astype(int)

    df["churn"] = df[["contract_end_churn", "soft_churn"]].max(axis=1)

    return df


def build_features(df):
    """Feature engineering for churn prediction."""
    df = df.sort_values(["company_id","vendor_id","date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    # Rolling stats
    df["roll1"] = df.groupby(["company_id","vendor_id"])["monthly_spend"].shift(1)
    df["roll2_mean"] = (
        df.groupby(["company_id","vendor_id"])["monthly_spend"]
        .shift(1).rolling(2).mean().reset_index(level=[0,1], drop=True)
    )
    df["roll3_mean"] = (
        df.groupby(["company_id","vendor_id"])["monthly_spend"]
        .shift(1).rolling(3).mean().reset_index(level=[0,1], drop=True)
    )
    df["roll3_std"] = (
        df.groupby(["company_id","vendor_id"])["monthly_spend"]
        .shift(1).rolling(3).std().reset_index(level=[0,1], drop=True)
    )

    # Growth
    df["mom_growth"] = (df["monthly_spend"] - df["roll1"]) / df["roll1"]
    df["qoq_growth"] = (df["monthly_spend"] - df["roll3_mean"]) / df["roll3_mean"]

    # Trend slope
    df["slope_3m"] = (
        df.groupby(["company_id","vendor_id"])["monthly_spend"]
        .shift(1).rolling(3).apply(compute_slope, raw=False)
    )

    # Volatility score
    df["volatility_score"] = df["roll3_std"] / (df["roll3_mean"] + 1)

    # Lifecycle
    df["contract_month"] = df.groupby(["company_id","vendor_id"]).cumcount()

    # Spend normalization
    size_map = {"SMB":1, "MidMarket":2, "Enterprise":3}
    if "company_size" in df.columns:
        df["size_norm_spend"] = df["monthly_spend"] / df["company_size"].map(size_map)
    else:
        df["size_norm_spend"] = df["monthly_spend"]

    df = df.replace([np.inf,-np.inf], np.nan).fillna(0)
    df = df.dropna(subset=["roll3_mean","roll3_std","slope_3m"]).copy()

    # One-hot encoding for categoricals
    categorical_columns = [
        c for c in ["company_region","company_size","vendor_contract_type","vendor_category"]
        if c in df.columns
    ]
    df = pd.get_dummies(df, columns=categorical_columns)

    return df


def train_model(features_df):
    """Fit logistic regression and return model + scaler + feature list."""
    y = features_df["future_soft_churn_2m"]
    drop_cols = [
        "company_id","vendor_id","date","monthly_spend","contract_end_churn",
        "soft_churn","churn","future_soft_churn_2m","roll2_tmp",
        "pct_change_vs_roll2","drop_flag","soft_churn_signal"
    ]
    feature_columns = [c for c in features_df.columns if c not in drop_cols]

    X = features_df[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y)

    auc = roc_auc_score(y, model.predict_proba(X_scaled)[:,1])
    return model, scaler, feature_columns, auc


def score(features_df, model, scaler, feature_columns):
    X = features_df[feature_columns]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:,1]

    scored = features_df.copy()
    scored["churn_prob"] = probs
    scored["churn_risk_pct"] = probs * 100
    return scored


def group_churn_curve(scored):
    df = scored.copy()
    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["churn_prob"]
        .mean().reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly["churn_risk_pct"] = monthly["churn_prob"] * 100
    return monthly


def vendor_curve(scored, vendor_id):
    df = scored[scored["vendor_id"] == vendor_id].copy()
    if df.empty:
        return pd.DataFrame(columns=["date","churn_risk_pct"])
    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["churn_prob"]
        .mean().reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly["churn_risk_pct"] = monthly["churn_prob"] * 100
    return monthly


# ------------------------------------------------------------
# SIDEBAR — FILE UPLOAD
# ------------------------------------------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df_raw = pd.read_csv(uploaded)
df_raw["date"] = pd.to_datetime(df_raw["date"])

st.subheader("Raw Data Preview")
st.dataframe(df_raw.head())


# ------------------------------------------------------------
# STEP 1 — BUILD CHURN LABELS
# ------------------------------------------------------------
st.subheader("Step 1 — Building Churn Labels")
df_labeled = build_soft_churn_labels(df_raw)
st.write(df_labeled["future_soft_churn_2m"].value_counts())


# ------------------------------------------------------------
# STEP 2 — FEATURE ENGINEERING
# ------------------------------------------------------------
st.subheader("Step 2 — Feature Engineering")
features_df = build_features(df_labeled)
st.write("Feature table shape:", features_df.shape)


# ------------------------------------------------------------
# STEP 3 — TRAIN MODEL
# ------------------------------------------------------------
st.subheader("Step 3 — Train Model")
model, scaler, feature_columns, auc = train_model(features_df)
st.success(f"Logistic Regression trained successfully. In-sample ROC-AUC = {auc:.3f}")


# ------------------------------------------------------------
# STEP 4 — SCORE PROBABILITIES
# ------------------------------------------------------------
st.subheader("Step 4 — Predict Churn Probabilities")
scored = score(features_df, model, scaler, feature_columns)


# ------------------------------------------------------------
# TABS: OVERVIEW | VENDORS | CUSTOMERS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Group Churn Curve", "🏷 Vendor Churn", "👥 Customer Risk"])


# -------------------- TAB 1 --------------------
with tab1:
    st.subheader("Group Churn Probability Curve (%)")

    group = group_churn_curve(scored)

    fig = px.line(
        group, x="date", y="churn_risk_pct", markers=True,
        title="Average 2-Month Churn Probability (All Customers)"
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------- TAB 2 --------------------
with tab2:
    st.subheader("Vendor-level Churn Probability Curve (%)")

    vendors = sorted(scored["vendor_id"].unique())
    v_id = st.selectbox("Select Vendor:", vendors)

    curve = vendor_curve(scored, v_id)
    fig2 = px.line(
        curve, x="date", y="churn_risk_pct", markers=True,
        title=f"Vendor {v_id} — Churn Probability (%)"
    )
    st.plotly_chart(fig2, use_container_width=True)


# -------------------- TAB 3 --------------------
with tab3:
    st.subheader("Customer-level Churn Risk")

    cust = (
        scored.groupby("company_id")["churn_prob"]
        .mean().reset_index()
    )
    cust["churn_risk_pct"] = cust["churn_prob"] * 100
    st.dataframe(cust.sort_values("churn_prob", ascending=False))
