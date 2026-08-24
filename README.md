# 📄 ChurnPredict — Customer Churn Probability Research & Forecasting Model

**Author:** Meet Shah
**Live demo:** [churnpredict-ai.streamlit.app](https://churnpredict-ai-jzetwdmeabfrsfysdcr9ag.streamlit.app/) *(PW: `churn2025` — click "get this app back up" if it's asleep)*

> Part of ongoing personal development toward quantitative research and data science — applying statistical and forecasting methods to business, economic, and financial datasets.

---

## 1. Overview

**ChurnPredict** is a quantitative model that estimates **two-month-ahead churn probability** for customers based on historical spend patterns. Using multi-year software spend panel data (2022–2025), the project builds:

- A forward-looking **soft churn** definition
- A comprehensive **feature engineering** layer (~50 features)
- **Time-aware** model training and testing (no look-ahead bias)
- **Walk-forward validation** for robust out-of-sample evaluation
- **Calibrated** churn probability estimates
- Monthly **"Group Churn Probability %"** curves for investor/portfolio visibility

The result is a statistically sound, interpretable model for analyzing customer retention risk and long-term revenue stability.

---

## 2. Why This Matters

Existing directional signals (e.g., raw spend and % change tracking) capture *what* happened but not the full *behavioral dynamics* across a panel. Large and small customers influence spend differently, and declines often emerge gradually and unevenly across cohorts.

ChurnPredict incorporates a broader signal set — **spend volatility, trend slope, rolling averages, momentum, and contract lifecycle** — to assess risk consistently across all panelists. This produces a more representative, predictive view of spend stability for forecasting, retention analysis, and investor interpretation.

---

## 3. Dataset

A realistic multi-entity spend dataset is simulated (random sample generated) with the following characteristics:

| Attribute | Detail |
|---|---|
| Time horizon | Jan 2022 – Dec 2025 |
| Entities | 500 companies (by region, size) × 60 software vendors (by category, contract type) |
| Metric | Monthly software spend per company–vendor pair |
| Size | ~65,000 monthly observations |

Simulated dynamics include: subscription stability, usage-based spend volatility, contract start/end behavior, monthly seasonality, and size/region-based differences.

> ⚠️ **Note:** Current dataset is synthetic. Next step is applying this framework to real panelist spend datasets.

---

## 4. Churn Definition

Traditional churn (contract termination) is abrupt and poorly predictable from spend alone. Instead, this project defines a **soft churn** signal that is forward-looking and predictive.

- **Contract-end churn:** binary flag on the final month of a contract — typically abrupt, poor predictive value.
- **Soft churn (predictive):** flagged when spend drops **≥ 40%** vs. the 2-month rolling average, the drop **persists ≥ 2 consecutive months**, and it is **not** caused by contract end.
- **Modeling target:** `future_soft_churn_2m` — whether a soft churn event occurs within the next 2 months.

---

## 5. Feature Engineering

~50 engineered features across six categories:

- **Rolling statistics:** `roll1`, `roll2_mean`, `roll3_mean`, `roll3_std` (volatility), `volatility_score = roll3_std / (roll3_mean + 1)`
- **Growth indicators:** `mom_growth` (month-over-month), `qoq_growth` (vs. trailing 3-month mean)
- **Trend:** `slope_3m` (3-month trend slope)
- **Contract lifecycle:** `contract_month` (months since customer start)
- **Spend normalization:** `size_norm_spend` (normalized by company size)
- **Categorical:** one-hot encodings for region, company size, contract type, vendor category

These features let the model detect deteriorating spend patterns, volatility spikes, lifecycle transition points, and size/region-based churn behavior.

---

## 6. Modeling Approach

Two models are evaluated:

| Model | Strengths |
|---|---|
| **Logistic Regression** | Highly interpretable; coefficients convert to odds ratios; works well on monotonic churn signals |
| **Random Forest** | Captures nonlinear patterns; strong discriminative power (initially miscalibrated) |

**Probability calibration:** Random Forest outputs are calibrated using Platt scaling (`CalibratedClassifierCV(method="sigmoid")`) to improve probability truthfulness — verified via calibration curves and Brier scores.

---

## 7. Validation (Quant Backtesting)

Because churn is time-dependent, **temporal splits** are used instead of random splits:

- **Train:** 2022–2024
- **Test:** 2025
- **Walk-forward validation** across sequential folds to confirm stability over time and prevent look-ahead bias.

---

## 8. Results

### Time-Based Test Performance
| Model | ROC-AUC |
|---|---|
| Logistic Regression | ~0.60 |
| Random Forest (uncalibrated) | ~0.90 |
| Random Forest (calibrated) | ~0.90 |

### Calibration Performance
| Metric | Uncalibrated RF | Calibrated RF |
|---|---|---|
| Brier Score | 0.004857 | 0.004942 |

The calibrated model aligns more closely with the diagonal on calibration curves — meaning predicted probabilities better reflect true churn rates (uncalibrated RF underestimates risk).

---

## 9. Interpretation of Churn Drivers

**Logistic Regression (odds ratios):**
- `contract_month` (OR ≈ 1.28) — risk increases as customers approach renewal
- `volatility_score` (OR ≈ 1.17) — unstable spend precedes churn
- `roll1 / roll3_mean` (OR > 1) — weakening recent spend
- `slope_3m` (OR > 1) — negative trend raises risk
- `company_size_SMB` (OR > 1) — SMBs churn more readily

**Random Forest — top feature importances:**
`contract_month`, `mom_growth`, `qoq_growth`, `slope_3m`, `volatility_score`, `roll3_std`, `size_norm_spend`

These patterns align with empirical SaaS churn behavior.

---

## 10. Monthly Group Churn Probability (%) — Investor View

Monthly average churn probability is computed and plotted across all active customers, producing a **portfolio-level risk curve** that shows:

- How churn risk evolves over time
- Macro-level degradation or improvement
- Vendor cohort stability
- Sensitivity to seasonality/segment shifts

This answers: *"What is the expected churn risk next month, based on the spend behavior of the current customer base?"*

An interactive per-vendor chart is also generated (see notebook final cell) for product-signal use cases.

---

## 11. Key Takeaways

- Soft-churn definition produces a predictable, forward-looking target
- Feature engineering captures volatility, trend, momentum, and lifecycle signals
- Strong model performance: **AUC ≈ 0.90** (calibrated Random Forest)
- Calibration ensures probabilities reflect true risk
- Walk-forward validation confirms stability over time
- Group Churn % curve provides a macro-level churn barometer for investors
- Top churn predictors reflect realistic business patterns

---

## 12. Limitations & Future Enhancements

**Limitations**
- Data is currently synthetic (next step: apply to real panelist spend datasets)
- Lacks granular usage-level data
- Contract terms are simplified
- No exogenous signals (macro conditions, competition)

**Potential Enhancements**
- Add seasonality features
- Incorporate survival analysis (Cox Proportional Hazards)
- Use gradient-boosted models (LightGBM / XGBoost)
- Add ARIMA/SARIMA-based spend forecasting
- Cluster customers into behavioral cohorts
- Add anomaly detection for sudden spend spikes
- Explore vendor-switching dynamics: tracking spend shifting *from* one vendor *to* another within a similar category/industry (churn for one vendor, growth for another)

---

## 13. Conclusion

ChurnPredict establishes a robust, interpretable, and validated churn prediction framework built on engineered time-series features, soft-churn target construction, temporal backtesting, and calibrated probabilities. It provides investor-facing insights — monthly churn probability trends, cohort-level indicators, and churn driver identification — applicable to any business managing recurring revenue relationships.

---

## 14. Links & Resources

- 🌐 **Live demo (Streamlit):** [churnpredict-ai.streamlit.app](https://churnpredict-ai-jzetwdmeabfrsfysdcr9ag.streamlit.app/) — password: `churn2025`

---

## 15. Tech Stack

- Python (pandas, numpy, scikit-learn)
- Logistic Regression & Random Forest (`CalibratedClassifierCV`)
- Streamlit (app hosting/demo)
- Databricks (notebook environment)

---

## Disclaimer

This is an internal research project (YipitData). The dataset used here is **synthetic** and generated for methodology demonstration purposes only. It is not derived from or representative of real client/panelist data unless explicitly stated.
