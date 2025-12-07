import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ============================================================
# PAGE CONFIG & BRANDING
# ============================================================
st.set_page_config(
    page_title="Churnpredict.ai",
    page_icon="📉",
    layout="wide"
)

PRIMARY_COLOR = "#1E88E5"
ACCENT_COLOR = "#FFB300"
LIGHT_BG = "#F7F9FC"

# Simple custom CSS for nicer look
st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }}
    .main-title {{
        font-size: 2.1rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
    }}
    .subtitle {{
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 0.8rem;
    }}
    .tag-pill {{
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background-color: {LIGHT_BG};
        border: 1px solid #E0E7FF;
        font-size: 0.75rem;
        color: #444;
        margin-right: 0.35rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PASSWORD GATE (simple, app-level)
# ============================================================
APP_PASSWORD = "churn2025"  # change this to whatever you want

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

with st.sidebar:
    st.markdown("### 🔐 Access")
    if not st.session_state.authenticated:
        pwd = st.text_input("Enter password to access the app:", type="password")
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.success("Access granted.")
        elif pwd != "":
            st.error("Incorrect password.")
    else:
        st.markdown("✅ You are logged in.")

if not st.session_state.authenticated:
    st.stop()

# ============================================================
# LANDING PAGE HEADER (Fixed for proper alignment)
# ============================================================

st.markdown(
    f"""
    <style>
        .landing-container {{
            padding-left: 2rem;   /* Adds space so header isn't cut */
            padding-right: 2rem;
        }}
        .main-title {{
            font-size: 2.4rem;
            font-weight: 700;
            color: {PRIMARY_COLOR};
            padding-top: 0.5rem;
        }}
        .subtitle {{
            font-size: 1.05rem;
            color: #444;
            margin-top: -5px;
            margin-bottom: 0.7rem;
        }}
        .tag-pill {{
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            background-color: {LIGHT_BG};
            border: 1px solid #E0E7FF;
            font-size: 0.78rem;
            color: #444;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Add a wrapper so Streamlit shifts content away from sidebar safely
st.markdown('<div class="landing-container">', unsafe_allow_html=True)

# Fix column ratio: give title more room
col_title, col_meta = st.columns([4, 3])

with col_title:
    st.markdown('<div class="main-title">Churnpredict.ai</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="subtitle">
            Predict <b>2-month-ahead churn risk</b> from panel spend data.<br/>
            Upload your own CSV or explore demo datasets across Manufacturing, Retail,
            B2B SaaS, Telecom, and more.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <span class="tag-pill">ML-based churn scoring</span>
        <span class="tag-pill">Time-series features</span>
        <span class="tag-pill">Soft churn detection</span>
        <span class="tag-pill">Portfolio & vendor curves</span>
        <span class="tag-pill">Flexible training</span>
        """,
        unsafe_allow_html=True,
    )

with col_meta:
    st.markdown(
        f"""
        <div style="
            background-color:{LIGHT_BG};
            padding:1rem 1.1rem;
            border-radius:0.75rem;
            border:1px solid #E0E7FF;
            line-height:1.45;
        ">
            <b>How to use:</b><br/>
            1. Pick a demo dataset or upload your own CSV.<br/>
            2. The app builds churn labels and features.<br/>
            3. A logistic regression model is fit (with flexible handling of small datasets).<br/>
            4. Explore churn risk easily at portfolio, vendor, and company levels.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")


# ============================================================
# HELPERS
# ============================================================

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

    # Rolling stats via transform
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

    If the target has only one class (all 0s or all 1s),
    we synthetically flip one row to create a second class
    so that the model can still train and the app does not break.
    """
    y = features_df["future_soft_churn_2m"].copy()

    # If only one class, synthetically create a churn example
    if y.nunique() < 2:
        unique_val = int(y.unique()[0])
        st.warning(
            "⚠️ Not enough true churn events in this dataset to train a model.\n\n"
            f"`future_soft_churn_2m` contains only one class: {unique_val}.\n\n"
            "For demonstration purposes, the app will synthetically mark one row as a "
            "churn event (1) so that the model can still fit and produce relative "
            "risk scores. For production use, please train on a larger dataset that "
            "includes real churn cases."
        )

        # Choose the row with the largest negative drop (if available),
        # otherwise just flip the first row.
        if "pct_change_vs_roll2" in features_df.columns:
            idx = features_df["pct_change_vs_roll2"].idxmin()
            y.loc[idx] = 1
        else:
            y.iloc[0] = 1

    drop_cols = [
        "company_id", "vendor_id", "date", "monthly_spend",
        "contract_end_churn", "soft_churn", "churn",
        "future_soft_churn_2m", "roll2_tmp", "pct_change_vs_roll2",
        "drop_flag", "soft_churn_signal",
    ]
    feature_columns = [c for c in features_df.columns if c not in drop_cols]

    X = features_df[feature_columns]

    # If no usable feature columns, bail out gracefully
    if X.shape[1] == 0:
        st.error("No feature columns available to train the model.")
        return None, None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_scaled, y)

    # In-sample AUC
    try:
        auc = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    except Exception:
        auc = None

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


# ============================================================
# DATA SOURCE SELECTION – MULTIPLE DEMO DATASETS
# ============================================================

st.sidebar.markdown("### 📂 Data Source")

data_choice = st.sidebar.selectbox(
    "Select dataset:",
    [
        "Demo (generic)",
        "Manufacturing",
        "Retail",
        "B2B SaaS",
        "Telecom",
        "Periodic churn",
        "Region volatility",
        "Vendor lifecycle",
        "Large demo (100 companies, 36 months)",
        "Upload my own CSV",
    ],
)

demo_files = {
    "Demo (generic)": "demo_churn_data.csv",
    "Manufacturing": "demo_manufacturing.csv",
    "Retail": "demo_retail.csv",
    "B2B SaaS": "demo_b2b_saas.csv",
    "Telecom": "demo_telecom.csv",
    "Periodic churn": "demo_periodic_churn.csv",
    "Region volatility": "demo_region_volatility.csv",
    "Vendor lifecycle": "demo_vendor_cycles.csv",
    "Large demo (100 companies, 36 months)": "large_demo_churn_dataset.csv",
}

df_raw = None

if data_choice != "Upload my own CSV":
    file_name = demo_files[data_choice]
    try:
        df_raw = pd.read_csv(file_name)
        st.sidebar.success(f"Loaded {file_name} successfully.")
    except Exception as e:
        st.sidebar.error(f"❗ Could not load {file_name}: {e}")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("⬅️ Upload a CSV file in the sidebar to begin.")
        st.stop()
    try:
        df_raw = pd.read_csv(uploaded)
        st.sidebar.success("Uploaded CSV successfully.")
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        st.stop()

# Ensure required columns
required_cols = ["company_id", "vendor_id", "date", "monthly_spend"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df_raw["date"] = pd.to_datetime(df_raw["date"])

# ============================================================
# LAYOUT TABS: APP | ABOUT & METHODOLOGY | DATA SCHEMA
# ============================================================

tab_app, tab_about, tab_schema = st.tabs(
    ["📈 Churn App", "📘 About & Methodology", "📊 Data Schema"]
)

# ---------- TAB: APP ----------
with tab_app:
    st.subheader("Raw Data Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    st.markdown("### Step 1 — Build Churn Labels")
    df_labeled = build_soft_churn_labels(df_raw)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Soft churn counts:")
        st.write(df_labeled["soft_churn"].value_counts())
    with col2:
        st.write("Future soft churn (2 months) counts:")
        st.write(df_labeled["future_soft_churn_2m"].value_counts())

    st.markdown("### Step 2 — Feature Engineering")
    features_df = build_features(df_labeled)
    st.write("Feature table shape:", features_df.shape)

    st.markdown("### Step 3 — Train Model")
    model, scaler, feature_columns, auc = train_model(features_df)
    if model is None:
        st.stop()

    if auc is not None:
        st.success(f"✅ Logistic Regression trained. In-sample ROC-AUC = {auc:.3f}")
    else:
        st.warning("Model trained, but ROC-AUC could not be computed.")

    st.markdown("### Step 4 — Predict Churn Probabilities")
    scored_df = score_rows(features_df, model, scaler, feature_columns)

    tab_overview, tab_vendor, tab_customers = st.tabs(
        ["🌐 Portfolio View", "🏷 Vendor View", "👥 Customer View"]
    )

    with tab_overview:
        st.subheader("Portfolio Churn Probability (%) Over Time")
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
            fig.update_traces(line=dict(color=PRIMARY_COLOR))
            st.plotly_chart(fig, use_container_width=True)

    with tab_vendor:
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
            fig2.update_traces(line=dict(color=ACCENT_COLOR))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_customers:
        st.subheader("Customer-level Churn Risk")
        cust = (
            scored_df.groupby("company_id")["churn_prob"]
            .mean().reset_index()
        )
        cust["churn_risk_pct"] = cust["churn_prob"] * 100.0

        st.markdown("Top 30 highest-risk customers:")
        st.dataframe(
            cust.sort_values("churn_prob", ascending=False).head(30),
            use_container_width=True
        )


# ---------- TAB: ABOUT & METHODOLOGY ----------
with tab_about:
    st.subheader("About Churnpredict.ai")

    st.markdown(
        """
        **Churnpredict.ai** is a prototype quant-style tool that estimates
        **2-month-ahead churn risk** for recurring spend relationships
        (e.g., software vendors, wholesalers, telecom plans).

        The approach is intentionally close to how an internal quant team might
        build features and labels from panel data:

        ### 1. Label: Soft Churn (behavioral)
        - We define a **soft churn event** when:
          - Spend drops by **≥ 40% vs. the 2-month rolling average**, and  
          - This drop is **sustained for 2 consecutive months**, and  
          - It is **not** simply the last observed contract month.
        - We also mark **contract-end churn** as the last month observed for each
          `(company_id, vendor_id)` pair.

        ### 2. Target: future_soft_churn_2m
        - For each relationship-month, we ask:  
          > “Does a soft churn event happen within the next 2 months?”
        - This becomes a binary classification target: `future_soft_churn_2m ∈ {0,1}`.

        ### 3. Features: Time-series & cross-sectional
        - Rolling aggregates on spend:
          - `roll1`, `roll2_mean`, `roll3_mean`, `roll3_std`
        - Momentum / trend:
          - `mom_growth`, `qoq_growth`, `slope_3m`
        - Volatility:
          - `volatility_score = roll3_std / (roll3_mean + 1)`
        - Contract lifecycle:
          - `contract_month` (tenure index)
        - Spend normalization:
          - `size_norm_spend` based on company_size buckets (SMB / MidMarket / Enterprise)
        - Categorical one-hot encodings:
          - `company_region`, `company_size`, `vendor_contract_type`, `vendor_category`

        ### 4. Model: Logistic Regression (with class balancing)
        - We fit a **logistic regression** with `class_weight="balanced"`.
        - Input features are standardized via `StandardScaler`.
        - We report an **in-sample ROC-AUC** (for intuition only, not as a final
          production metric).

        ### 5. Handling small / quiet datasets
        - Some uploaded datasets may have **no visible churn events**, e.g.,:
          - Only a few months of data, or
          - Very stable spend with no sharp drops.
        - In those cases, the target `future_soft_churn_2m` would have only one class,
          which normally breaks logistic regression.
        - To keep the tool usable for demos and small panels:
          - We **synthetically mark one row as a churn event** (the largest negative spend
            change, if available), and
          - We display a clear warning that this is for demonstration, not production.

        ### 6. Outputs
        - **Portfolio curve:** average churn probability across all active relationships.
        - **Vendor curve:** average churn probability for a selected vendor over time.
        - **Customer table:** average churn risk by company, sortable for targeting / triage.

        This framework is intentionally modular so it can be extended with:
        - Alternative churn definitions,
        - Additional behavioral features,
        - More advanced models (e.g., calibrated trees, gradient boosting),
        - Or integration into a broader quant research stack.
        """
    )


# ---------- TAB: DATA SCHEMA ----------
with tab_schema:
    st.subheader("Data Schema Requirements")

    st.markdown(
        """
        To use your own CSV, please ensure it follows this schema:

        ### Required columns
        - **`company_id`**: Identifier for the buyer / customer / account  
          - Example: `C001`, `WH001`, `T_CUST1`
        - **`vendor_id`**: Identifier for the vendor / product / supplier  
          - Example: `V001`, `V_SAAS1`, `PLAN_A`
        - **`date`**: Month-level date in ISO format  
          - Example: `2024-01-01`, `2023-07-01`  
          - One row per `(company_id, vendor_id, month)`
        - **`monthly_spend`**: Observed spend in that month (numeric)  
          - Currency-agnostic; can be in USD, INR, etc.

        ### Optional (recommended) columns
        - **`company_size`**: Size buckets used for normalization  
          - Allowed values: `SMB`, `MidMarket`, `Enterprise`
        - **`company_region`**: Region/country buckets  
          - Examples: `NA`, `EU`, `APAC`, `LATAM`
        - **`vendor_contract_type`**: Recurring revenue type  
          - Examples: `subscription`, `usage`
        - **`vendor_category`**: Broad vendor category  
          - Examples: `CRM`, `Analytics`, `DevTools`, `Security`, `Industrial`, `Retail`

        ### Minimum data length
        - For stable features, we recommend:
          - At least **6–12 months** of history per `(company_id, vendor_id)` pair.
        - The model will still run on smaller data, but:
          - Churn labels may be sparse or synthetic,
          - Signals will be noisier,
          - Interpretations should be treated as directional, not final.

        ### Example row
        ```text
        company_id,vendor_id,date,monthly_spend,company_size,company_region,vendor_contract_type,vendor_category
        C001,V_SAAS1,2024-01-01,1200,SMB,NA,subscription,CRM
        ```

        If you follow this schema, your dataset should pass validation and flow through
        the full churn labeling → feature engineering → model training → visualization
        pipeline.
        """
    )
