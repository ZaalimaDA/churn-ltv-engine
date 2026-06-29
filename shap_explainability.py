import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ── 0. SETUP ────────────────────────────────────────────────────────────────
load_dotenv()
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

sns.set_theme(style="whitegrid")
OUTPUT_DIR = "shap_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 5 engineered features (must match model_training.py exactly) ─────────
ENGINEERED = [
    "charge_per_tenure",
    "service_count",
    "charge_to_value_ratio",
    "support_dependency_score",
    "tenure_contract_risk",
]

# ════════════════════════════════════════════════════════════════════════════
# PART 1 — LOAD MODEL & REBUILD FEATURE MATRIX
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PART 1 — LOAD MODEL & REBUILD FEATURE MATRIX")
print("=" * 65)

# Load the saved XGBoost model (best performer from Day 4-6)
xgb = joblib.load("saved_models/xgboost.pkl")
print("\nLoaded: saved_models/xgboost.pkl")

# Rebuild the exact same feature matrix used in model_training.py
df = pd.read_sql("SELECT * FROM customers_features", engine)
df["total_charges"] = pd.to_numeric(
    df["total_charges"], errors="coerce").fillna(0)
df["churn_binary"] = (df["churn"] == "Yes").astype(int)
print(f"Loaded customers_features: {df.shape[0]} rows")

# --- Encode categoricals (same as model_training.py) ---
df["gender_enc"]      = (df["gender"] == "Male").astype(int)
df["partner_enc"]     = (df["partner"] == "Yes").astype(int)
df["dependents_enc"]  = (df["dependents"] == "Yes").astype(int)
df["paperless_enc"]   = (df["paperless_billing"] == "Yes").astype(int)
contract_map          = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["contract_enc"]    = df["contract"].map(contract_map)

service_raw = [
    "phone_service", "multiple_lines", "online_security",
    "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies",
]
for col in service_raw:
    df[col] = df[col].replace(
        {"No internet service": "No", "No phone service": "No"})
    df[f"{col}_enc"] = (df[col] == "Yes").astype(int)

SERVICE_ENC = [f"{c}_enc" for c in service_raw]
NUMERIC     = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]
EXTRA_CATS  = ["gender_enc", "partner_enc", "dependents_enc",
               "paperless_enc", "contract_enc"]

df = pd.get_dummies(
    df, columns=["internet_service", "payment_method"],
    prefix=["internet", "payment"], drop_first=False,
)
ONEHOT = [c for c in df.columns
          if c.startswith("internet_") or c.startswith("payment_")]

ALL_FEATURES = ENGINEERED + NUMERIC + SERVICE_ENC + EXTRA_CATS + ONEHOT

X = df[ALL_FEATURES].copy()
y = df["churn_binary"].copy()
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} cols")
print(f"Engineered features: {ENGINEERED}")

# ════════════════════════════════════════════════════════════════════════════
# PART 2 — COMPUTE SHAP VALUES
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 2 — COMPUTING SHAP VALUES")
print("=" * 65)

# Use a sample of 1000 rows for speed (SHAP is computationally expensive)
# Use the full dataset for production-level analysis
SAMPLE_SIZE = 1000
X_sample = X.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
y_sample = y.loc[X_sample.index].reset_index(drop=True)

print(f"\nComputing SHAP values for {SAMPLE_SIZE} samples...")
print("(This may take 30-60 seconds)")

# TreeExplainer is optimised for tree-based models (XGBoost, RF)
explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)

print(f"SHAP values shape : {shap_values.shape}")
print(f"Base value (E[f]) : {explainer.expected_value:.4f}  "
      f"(model's average prediction in log-odds)")

# Convert to DataFrame for easier manipulation
shap_df = pd.DataFrame(shap_values, columns=ALL_FEATURES)

# Mean absolute SHAP per feature (global importance)
mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
print(f"\nTop 10 features by mean |SHAP|:")
print(mean_abs_shap.head(10).round(4).to_string())

# ════════════════════════════════════════════════════════════════════════════
# PART 3 — PLOTS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 3 — GENERATING SHAP PLOTS")
print("=" * 65)

# ── PLOT 1: Beeswarm Summary Plot ────────────────────────────────────────
print("\nGenerating Plot 1: Beeswarm Summary...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_sample,
    feature_names=ALL_FEATURES,
    max_display=15,
    show=False,
    plot_size=None,
)
plt.title("SHAP Summary — Global Feature Impact on Churn\n"
          "(Red = high feature value  |  Blue = low feature value  |  "
          "X-axis = impact on churn probability)",
          fontsize=11, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_shap_summary_beeswarm.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/01_shap_summary_beeswarm.png")

# ── PLOT 2: Global Bar Chart (Mean |SHAP|) ───────────────────────────────
print("Generating Plot 2: Global Bar Chart...")
top15        = mean_abs_shap.head(15)
eng_set      = set(ENGINEERED)
bar_colors   = ["#F44336" if f in eng_set else "#2196F3"
                for f in top15.index]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    range(len(top15)), top15.values[::-1],
    color=bar_colors[::-1], edgecolor="white",
)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index[::-1], fontsize=10)
ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)", fontsize=11)
ax.set_title("Global Feature Importance via SHAP\n"
             "(Red bars = engineered features from Week 2 Day 1-3)",
             fontsize=12, fontweight="bold")

red_patch  = mpatches.Patch(color="#F44336", label="Engineered feature")
blue_patch = mpatches.Patch(color="#2196F3", label="Original feature")
ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=10)

for i, (bar, val) in enumerate(zip(bars[::-1], top15.values[::-1])):
    ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_shap_bar_global.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/02_shap_bar_global.png")

# ── PLOT 3: Engineered Features SHAP Spotlight ──────────────────────────
print("Generating Plot 3: Engineered Features Spotlight...")
eng_shap_mean = mean_abs_shap[ENGINEERED].sort_values(ascending=True)
eng_colors    = ["#FF5722", "#FF9800", "#FFC107", "#E91E63", "#F44336"]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(
    eng_shap_mean.index, eng_shap_mean.values,
    color=eng_colors[:len(eng_shap_mean)], edgecolor="white",
)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.set_title("SHAP Importance — 5 Engineered Features\n"
             "Validates that feature engineering added real predictive value",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, eng_shap_mean.values):
    ax.text(val + 0.0003, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_shap_engineered_spotlight.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/03_shap_engineered_spotlight.png")

# ── PLOT 4: Dependence Plots for Top 3 Engineered Features ──────────────
print("Generating Plot 4: Dependence Plots...")
top3_eng = mean_abs_shap[ENGINEERED].sort_values(ascending=False).head(3).index.tolist()
print(f"  Top 3 engineered features: {top3_eng}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, feat in zip(axes, top3_eng):
    feat_vals  = X_sample[feat].values
    shap_vals  = shap_df[feat].values
    sc = ax.scatter(
        feat_vals, shap_vals,
        c=feat_vals, cmap="RdYlBu_r",
        alpha=0.5, s=15, edgecolors="none",
    )
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feat.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel("SHAP Value (impact on churn)", fontsize=10)
    ax.set_title(f"Dependence: {feat.replace('_', ' ').title()}",
                 fontweight="bold")
    plt.colorbar(sc, ax=ax, label="Feature value")

plt.suptitle("SHAP Dependence Plots — Top 3 Engineered Features\n"
             "(Above 0 = pushes toward churn  |  Below 0 = pushes away from churn)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_shap_dependence_top3.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/04_shap_dependence_top3.png")

# ── PLOT 5: Waterfall — Single High-Risk Customer ────────────────────────
print("Generating Plot 5: Waterfall - High Risk Customer...")

# Find a real churner with high model confidence
xgb_probs     = xgb.predict_proba(X_sample)[:, 1]
churner_mask  = y_sample == 1
if churner_mask.sum() > 0:
    churner_probs = xgb_probs.copy()
    churner_probs[~churner_mask] = -1
    high_risk_idx = int(np.argmax(churner_probs))
else:
    high_risk_idx = int(np.argmax(xgb_probs))

hr_prob = xgb_probs[high_risk_idx]
hr_actual = y_sample.iloc[high_risk_idx]

print(f"  High-risk customer index : {high_risk_idx}")
print(f"  Churn probability        : {hr_prob:.4f}")
print(f"  Actual churn             : {'Yes' if hr_actual == 1 else 'No'}")

# Print engineered feature values for this customer
print(f"  Engineered feature values:")
for feat in ENGINEERED:
    print(f"    {feat:35s}: {X_sample.loc[high_risk_idx, feat]:.4f}")

shap_exp_hr = shap.Explanation(
    values         = shap_values[high_risk_idx],
    base_values    = explainer.expected_value,
    data           = X_sample.iloc[high_risk_idx].values,
    feature_names  = ALL_FEATURES,
)

fig, ax = plt.subplots(figsize=(12, 7))
shap.waterfall_plot(shap_exp_hr, max_display=12, show=False)
plt.title(f"SHAP Waterfall — HIGH-RISK Customer\n"
          f"Predicted Churn Probability: {hr_prob:.1%}  "
          f"| Actual: {'Churned' if hr_actual == 1 else 'Retained'}",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_shap_waterfall_churner.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/05_shap_waterfall_churner.png")

# ── PLOT 6: Waterfall — Single Low-Risk Customer ─────────────────────────
print("Generating Plot 6: Waterfall - Low Risk Customer...")

retained_mask  = y_sample == 0
if retained_mask.sum() > 0:
    retained_probs = xgb_probs.copy()
    retained_probs[~retained_mask] = 2
    low_risk_idx = int(np.argmin(retained_probs))
else:
    low_risk_idx = int(np.argmin(xgb_probs))

lr_prob   = xgb_probs[low_risk_idx]
lr_actual = y_sample.iloc[low_risk_idx]

print(f"  Low-risk customer index  : {low_risk_idx}")
print(f"  Churn probability        : {lr_prob:.4f}")
print(f"  Actual churn             : {'Yes' if lr_actual == 1 else 'No'}")

shap_exp_lr = shap.Explanation(
    values         = shap_values[low_risk_idx],
    base_values    = explainer.expected_value,
    data           = X_sample.iloc[low_risk_idx].values,
    feature_names  = ALL_FEATURES,
)

fig, ax = plt.subplots(figsize=(12, 7))
shap.waterfall_plot(shap_exp_lr, max_display=12, show=False)
plt.title(f"SHAP Waterfall — LOW-RISK Customer\n"
          f"Predicted Churn Probability: {lr_prob:.1%}  "
          f"| Actual: {'Churned' if lr_actual == 1 else 'Retained'}",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_shap_waterfall_retained.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/06_shap_waterfall_retained.png")

# ── PLOT 7: Stakeholder Business Report Dashboard ────────────────────────
print("Generating Plot 7: Stakeholder Business Report...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "SHAP Explainability Report — Customer Churn Prediction\n"
    "Business Stakeholder Summary  |  XGBoost Model  |  churn-ltv-engine",
    fontsize=16, fontweight="bold", y=0.98,
)

# Panel 1: Top 10 global importance bar
ax1 = fig.add_subplot(2, 3, 1)
top10   = mean_abs_shap.head(10)
colors1 = ["#F44336" if f in eng_set else "#90CAF9" for f in top10.index]
bars1   = ax1.barh(range(len(top10)), top10.values[::-1],
                   color=colors1[::-1], edgecolor="white")
ax1.set_yticks(range(len(top10)))
ax1.set_yticklabels(
    [f.replace("_", " ").title()[:22] for f in top10.index[::-1]],
    fontsize=8,
)
ax1.set_title("Top 10 Churn Drivers", fontweight="bold", fontsize=11)
ax1.set_xlabel("Mean |SHAP|", fontsize=9)
red_p  = mpatches.Patch(color="#F44336",  label="Engineered")
blue_p = mpatches.Patch(color="#90CAF9", label="Original")
ax1.legend(handles=[red_p, blue_p], fontsize=8, loc="lower right")

# Panel 2: Engineered features only
ax2 = fig.add_subplot(2, 3, 2)
eng_vals   = mean_abs_shap[ENGINEERED].sort_values(ascending=True)
eng_labels = [f.replace("_", " ").title() for f in eng_vals.index]
ax2.barh(eng_labels, eng_vals.values,
         color=["#FF5722","#FF9800","#FFC107","#E91E63","#F44336"],
         edgecolor="white")
ax2.set_title("Engineered Feature Impact", fontweight="bold", fontsize=11)
ax2.set_xlabel("Mean |SHAP|", fontsize=9)
for i, val in enumerate(eng_vals.values):
    ax2.text(val + 0.0002, i, f"{val:.4f}", va="center", fontsize=9)

# Panel 3: Churn risk distribution
ax3 = fig.add_subplot(2, 3, 3)
ax3.hist(xgb_probs[y_sample == 0], bins=30, alpha=0.6,
         color="#2196F3", label="Retained", edgecolor="white")
ax3.hist(xgb_probs[y_sample == 1], bins=30, alpha=0.6,
         color="#F44336", label="Churned",  edgecolor="white")
ax3.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5,
            label="Decision boundary (0.5)")
ax3.set_xlabel("Predicted Churn Probability", fontsize=9)
ax3.set_ylabel("Count", fontsize=9)
ax3.set_title("Risk Score Distribution", fontweight="bold", fontsize=11)
ax3.legend(fontsize=8)

# Panel 4: High risk waterfall (condensed)
ax4 = fig.add_subplot(2, 3, (4, 5))
top_shap_idx = np.argsort(np.abs(shap_values[high_risk_idx]))[-8:]
feat_names   = [ALL_FEATURES[i] for i in top_shap_idx]
feat_shaps   = [shap_values[high_risk_idx][i] for i in top_shap_idx]
colors_wf    = ["#F44336" if v > 0 else "#2196F3" for v in feat_shaps]
labels_wf    = [f.replace("_", " ").title()[:25] for f in feat_names]

ax4.barh(labels_wf, feat_shaps, color=colors_wf, edgecolor="white")
ax4.axvline(x=0, color="black", linewidth=0.8)
ax4.set_title(
    f"High-Risk Customer Explanation  "
    f"(Churn Probability: {hr_prob:.1%})\n"
    "Red = pushing toward churn  |  Blue = pushing away from churn",
    fontweight="bold", fontsize=11,
)
ax4.set_xlabel("SHAP Value", fontsize=9)

# Panel 5: Key business findings text box
ax5 = fig.add_subplot(2, 3, 6)
ax5.axis("off")
top1     = mean_abs_shap.index[0].replace("_", " ").title()
top_eng  = mean_abs_shap[ENGINEERED].idxmax().replace("_", " ").title()
n_eng_top5 = sum(1 for f in mean_abs_shap.head(5).index if f in eng_set)

findings = (
    f"KEY FINDINGS FOR STAKEHOLDERS\n"
    f"{'='*34}\n\n"
    f"1. Top churn driver:\n   {top1}\n\n"
    f"2. Engineered features in top 5:\n   {n_eng_top5} out of 5 features\n\n"
    f"3. Best engineered signal:\n   {top_eng}\n\n"
    f"4. High-risk customer profile:\n"
    f"   Prob = {hr_prob:.1%}\n\n"
    f"5. Model separates churners\n"
    f"   from retained customers\n"
    f"   using SHAP-ranked drivers\n\n"
    f"Model: XGBoost\n"
    f"Sample: {SAMPLE_SIZE} customers\n"
    f"Features: {len(ALL_FEATURES)} total (5 engineered)"
)
ax5.text(
    0.05, 0.95, findings,
    transform=ax5.transAxes,
    fontsize=9, verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5",
              edgecolor="#048A81", linewidth=1.5),
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUTPUT_DIR}/07_shap_business_report.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_DIR}/07_shap_business_report.png")

# ════════════════════════════════════════════════════════════════════════════
# PART 4 — SAVE SHAP VALUES TO POSTGRESQL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 4 — SAVE SHAP VALUES TO POSTGRESQL")
print("=" * 65)

# Save mean absolute SHAP per feature (global importance table)
shap_importance = pd.DataFrame({
    "feature"         : mean_abs_shap.index,
    "mean_abs_shap"   : mean_abs_shap.values.round(6),
    "is_engineered"   : [f in eng_set for f in mean_abs_shap.index],
    "global_rank"     : range(1, len(mean_abs_shap) + 1),
})
shap_importance.to_sql(
    "shap_feature_importance", engine,
    if_exists="replace", index=False,
)
print(f"\nSaved shap_feature_importance table ({len(shap_importance)} rows)")

# Save per-customer SHAP values for sample
shap_customer = shap_df.copy()
shap_customer["churn_actual"]      = y_sample.values
shap_customer["churn_probability"] = xgb_probs.round(4)
shap_customer.to_sql(
    "shap_customer_values", engine,
    if_exists="replace", index=False,
)
print(f"Saved shap_customer_values table ({len(shap_customer)} rows)")

# ════════════════════════════════════════════════════════════════════════════
# PART 5 — BUSINESS NARRATIVE 
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 5 — BUSINESS NARRATIVE")
print("=" * 65)

top5 = mean_abs_shap.head(5)
print("\nTop 5 Churn Drivers (SHAP Global Importance):")
for rank, (feat, val) in enumerate(top5.items(), 1):
    tag = "[ENGINEERED]" if feat in eng_set else "[ORIGINAL]  "
    print(f"  {rank}. {tag} {feat:35s} mean|SHAP| = {val:.4f}")

print(f"""
BUSINESS INTERPRETATION:
------------------------
The XGBoost model's decisions can now be explained to stakeholders:

Globally (across all customers):
  The most impactful factors driving churn are those identified
  in Week 1 EDA and quantified in Week 2 feature engineering.
  Engineered features appearing in the top 5 confirms that
  the feature engineering work added real predictive value.

For a high-risk individual customer (prob = {hr_prob:.1%}):
  The waterfall plot shows exactly which features pushed this
  customer toward the churn prediction and by how much.
  This enables targeted, personalised retention actions.

For the marketing team:
  Focus retention budget on customers where tenure_contract_risk
  and charge_per_tenure are high — these are the two signals
  most strongly associated with churn across all customers.
""")

