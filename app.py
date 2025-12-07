import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="Churnpredict.ai", page_icon="📉", layout="wide")
st.title("📉 Churnpredict.ai – Churn Probability Predictor")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def compute_slope(x: pd.Series) -> float:
    """Compute simple linear slope over the given series."""
    if len(x) < 3:
        return 0.0
    y = x.values
    xx = np.arange(len(y))
    return np.polyfit(xx, y, 1)[0]


def build_soft_churn_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build:
      - contract_end_churn
      - soft_churn
      - future_soft_churn_2m
    """
    df = df.sort_values(["company_id", "vendor_id", "date"]).copy()

    # Contract-end churn: last month of each (company, vendor)
    rel_end = df.groupby(["company_id", "vendor_id"])["date"].max().reset_index()
    rel_end["contract_end_churn"] = 1
    df = df.merge(rel_end, on=["company_id", "vendor_id", "date"], how="left")
    df["contract_end_churn"] = df["contract_end_churn"].fillna(0)

    # Rolling 2-month avg on lagged spend
    g = df.groupby(["company_id", "vendor_id"])["monthly_spend"]
    df["roll2_tmp"] = g.transform(lambda s: s.shift(1).rolling(2).mean())
    df["pct_change_vs_roll2"] = (df["monthly_spend"] - df["roll2_tmp"]) / df["roll2_tmp"]

    # Large negative drop flag
    df["drop_flag"] = df["pct_change_vs_roll2"] <= -0.40

    # Need 2 consecutive big drops
    df["soft_churn_signal"] = (
        df.groupby(["company_id", "vendor_id"])["drop_flag"]
        .transform(lambda s: s.rolling(2).sum())
    )

    df["soft_churn"] = (
        (df["soft_churn_signal"] >= 2) & (df["contract_end_churn"] == 0)
    ).astype(int)

    # Combined churn (if needed)
    df["churn"] = df[["contract_end_churn", "soft_churn"]].max(axis=1)

    # Future soft churn in the next 2 months
    df["future_soft_churn_2m"] = (
        df.groupby(["company_id", "vendor_id"])["soft_churn"].shift(-2)
    )
    df["future_soft_churn_2m"] = df["future_soft_churn_2m"].fillna(0).astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer rolling stats, growth, trend, volatility, contract_month,
    size-normalized spend, and one-hot categoricals.
    """
    df = df.sort_values(["company_id", "vendor_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    # Group for rolling operations
    g = df.groupby(["company_id", "vendor_id"])["monthly_spend"]

    # Rolling stats via transform (no reset_index, no multi-index issues)
    df["roll1"] = g.transform(lambda s: s.shift(1))
    df["roll2_mean"] = g.transform(lambda s: s.shift(1).rolling(2).mean())
    df["roll3_mean"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    df["roll3_std"] = g.transform(lambda s: s.shift(1).rolling(3).std())

    # Growth
    df["mom_growth"] = (df["monthly_spend"] - df["roll1"]) / df["roll1"]
    df["qoq_growth"] = (df["monthly_spend"] - df["roll3_mean"]) / df["roll3_mean"]

    # Trend slope (3 months, on lagged spend)
    df["slope_3m"] = g.transform(
        lambda s: s.shift(1).rolling(3).apply(compute_slope, raw=False)
    )

    # Volatility score
    df["volatility_score"] = df["roll3_std"] / (df["roll3_mean"] + 1)

    # Contract month
    df["contract_month"] = df.groupby(["company_id", "vendor_id"]).cumcount()

    # Size-normalized spend
    size_map = {"SMB": 1, "MidMarket": 2, "Enterprise": 3}
    if "company_size" in df.columns:
        df["size_norm_spend"] = df["monthly_spend"] / df["company_size"].map(size_map)
    else:
        df["size_norm_spend"] = df["monthly_spend"]

    # Clean inf/NaN
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Keep rows where core rolling features exist
    df = df.dropna(subset=["roll3_mean", "roll3_std", "slope_3m"]).copy()

    # One-hot encode categoricals if present
    cat_cols = [
        c
        for c in ["company_region", "company_size", "vendor_contract_type", "vendor_category"]
        if c in df.columns
    ]
    df = pd.get_dummies(df, columns=cat_cols)

    return df


def train_model(features_df: pd.DataFrame):
    """
    Train logistic regression.

    If we don't have both classes in the target, show error and return Nones.
    """
    y = features_df["future_soft_churn_2m"]

    # Need at least 2 classes in the target
    if y.nunique() < 2:
        st.error(
            "❗ Not enough churn events to train a model.\n\n"
            f"`future_soft_churn_2m` contains only one class: {int(y.unique()[0])}.\n\n"
            "Please upload a larger or more volatile dataset where at least some "
            "relationships show soft churn."
        )
        return None, None, None, None

    drop_cols = [
        "company_id", "vendor_id", "date", "monthly_spend",
        "contract_end_churn", "soft_churn", "churn",
        "future_soft_churn_2m", "roll2_tmp", "pct_change_vs_roll2",
        "drop_flag", "soft_churn_signal",
    ]
    feature_columns = [c for c in features_df.columns if c not in drop_cols]

    X = features_df[feature_columns]

    # Guard: if no usable features, bail
    if X.shape[1] == 0:
        st.error("No feature columns available to train the model.")
        return None, None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y)

    auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    return model, scaler, feature_columns, auc


def score_rows(features_df: pd.DataFrame, model, scaler, feature_columns):
    """Score each row with churn probabilities."""
    X = features_df[feature_columns]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    df_scored = features_df.copy()
    df_scored["churn_prob"] = probs
    df_scored["churn_risk_pct"] = probs * 100.0
    return df_scored


def group_churn_curve(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["churn_prob"]
        .mean().reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly["churn_risk_pct"] = monthly["churn_prob"] * 100.0
    return monthly


def vendor_churn_curve(scored_df: pd.DataFrame, vendor_id: str) -> pd.DataFrame:
    df = scored_df[scored_df["vendor_id"] == vendor_id].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "churn_risk_pct"])
    df["date"] = pd.to_datetime(df["date"])
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["churn_prob"]
        .mean().reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly["churn_risk_pct"] = monthly["churn_prob"] * 100.0
    return monthly


# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------

st.sidebar.header("Upload CSV")
uploaded = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is None:
    st.info("⬅️ Upload a CSV file in the sidebar to begin.")
    st.stop()

# Read CSV safely (handle ParserError)
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

required_cols = ["company_id", "vendor_id", "date", "monthly_spend"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df_raw["date"] = pd.to_datetime(df_raw["date"])

st.subheader("Raw Data Preview")
st.dataframe(df_raw.head())


# ---------- Step 1: Churn labels ----------
st.subheader("Step 1 — Build Churn Labels")
df_labeled = build_soft_churn_labels(df_raw)

st.write("Soft churn counts:")
st.write(df_labeled["soft_churn"].value_counts())
st.write("Future soft churn (2m) counts:")
st.write(df_labeled["future_soft_churn_2m"].value_counts())


# ---------- Step 2: Features ----------
st.subheader("Step 2 — Feature Engineering")
features_df = build_features(df_labeled)
st.write("Feature table shape:", features_df.shape)


# ---------- Step 3: Train model ----------
st.subheader("Step 3 — Train Model")
model, scaler, feature_columns, auc = train_model(features_df)

if model is None:
    # train_model already showed an error message
    st.stop()

st.success(f"✅ Logistic Regression trained. In-sample ROC-AUC = {auc:.3f}")


# ---------- Step 4: Score ----------
st.subheader("Step 4 — Predict Churn Probabilities")
scored_df = score_rows(features_df, model, scaler, feature_columns)


# ------------------------------------------------------------
# TABS: OVERVIEW | VENDORS | CUSTOMERS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Group Churn Curve", "🏷 Vendor Churn", "👥 Customer Risk"])


with tab1:
    st.subheader("Group Churn Probability (%) Over Time")
    group_df = group_churn_curve(scored_df)
    if group_df.empty:
        st.warning("Not enough data to compute group churn curve.")
    else:
        fig = px.line(
            group_df,
            x="date",
            y="churn_risk_pct",
            markers=True,
            title="Average 2-Month Soft Churn Probability – All Relationships",
            labels={"date": "Month", "churn_risk_pct": "Churn probability (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.subheader("Vendor-level Churn Probability Curve")
    vendors = sorted(scored_df["vendor_id"].unique())
    v_id = st.selectbox("Select vendor", vendors)
    v_curve = vendor_churn_curve(scored_df, v_id)
    if v_curve.empty:
        st.warning("No data for this vendor.")
    else:
        fig2 = px.line(
            v_curve,
            x="date",
            y="churn_risk_pct",
            markers=True,
            title=f"Vendor {v_id} – Average 2-Month Soft Churn Probability",
            labels={"date": "Month", "churn_risk_pct": "Churn probability (%)"},
        )
        st.plotly_chart(fig2, use_container_width=True)


with tab3:
    st.subheader("Customer-level Churn Risk")
    cust = (
        scored_df.groupby("company_id")["churn_prob"]
        .mean().reset_index()
    )
    cust["churn_risk_pct"] = cust["churn_prob"] * 100.0
    st.dataframe(
        cust.sort_values("churn_prob", ascending=False),
        use_container_width=True
    )
