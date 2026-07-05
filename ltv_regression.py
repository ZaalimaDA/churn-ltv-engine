import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
 
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
 
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBRegressor
 
# ── 0. SETUP ────────────────────────────────────────────────────────────────
load_dotenv()
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
sns.set_theme(style="whitegrid", palette="muted")
OUTPUT_DIR = "ltv_outputs"
MODELS_DIR = "saved_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# PART 1 — LOAD DATA & COMPUTE LTV TARGET
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PART 1 — LOAD DATA & COMPUTE LTV TARGET")
print("=" * 65)
 
df = pd.read_sql("SELECT * FROM customers_features", engine)
df["total_charges"] = pd.to_numeric(
    df["total_charges"], errors="coerce").fillna(0)
df["churn_binary"] = (df["churn"] == "Yes").astype(int)
print(f"\nLoaded customers_features : {df.shape[0]} rows, {df.shape[1]} cols")
 
# ── 1a. Historical LTV (what each customer has already paid) ──────────────
df["ltv_historical"] = df["total_charges"]
 
# ── 1b. Projected LTV target variable ─────────────────────────────────────
# For active (non-churned) customers:
#   Projected LTV = monthly_charges × expected_remaining_months
#
# Expected remaining months is estimated using a simple formula:
#   - If on Two year contract   → assume 24 more months minimum
#   - If on One year contract   → assume 12 more months minimum
#   - If on Month-to-month      → use survival estimate = avg_tenure × (1 - churn_rate)
#
# For all customers we create a blended target:
#   ltv_projected = historical_ltv + monthly_charges × projected_months
 
contract_months = {
    "Month-to-month": 6,    # conservative: ~50% churn within 6mo
    "One year"       : 12,
    "Two year"       : 24,
}
df["projected_months"] = df["contract"].map(contract_months).fillna(6)
 
# Blend: historical + projected
df["ltv_projected"] = (
    df["total_charges"] + df["monthly_charges"] * df["projected_months"]
)
 
# Cap at 99th percentile to reduce outlier influence
ltv_cap = df["ltv_projected"].quantile(0.99)
df["ltv_projected"] = df["ltv_projected"].clip(upper=ltv_cap)
 
print(f"\nLTV Target Statistics:")
print(f"  Historical LTV  — mean: ${df['ltv_historical'].mean():,.2f}  "
      f"  std: ${df['ltv_historical'].std():,.2f}  "
      f"  max: ${df['ltv_historical'].max():,.2f}")
print(f"  Projected LTV   — mean: ${df['ltv_projected'].mean():,.2f}  "
      f"  std: ${df['ltv_projected'].std():,.2f}  "
      f"  max: ${df['ltv_projected'].max():,.2f}")
 
# ── 1c. LTV Segments for business use ─────────────────────────────────────
df["ltv_segment"] = pd.qcut(
    df["ltv_projected"],
    q=4,
    labels=["Low", "Medium", "High", "Premium"],
)
print(f"\nLTV Segment Distribution:")
print(df["ltv_segment"].value_counts().sort_index())
 
# ════════════════════════════════════════════════════════════════════════════
# PART 2 — EDA ON LTV
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 2 — LTV EXPLORATORY ANALYSIS")
print("=" * 65)
 
# ── PLOT 1: LTV Distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
axes[0].hist(df["ltv_historical"], bins=50,
             color="#2196F3", edgecolor="white", alpha=0.8)
axes[0].set_title("Historical LTV Distribution", fontweight="bold")
axes[0].set_xlabel("Total Charges ($)")
axes[0].set_ylabel("Count")
 
axes[1].hist(df["ltv_projected"], bins=50,
             color="#4CAF50", edgecolor="white", alpha=0.8)
axes[1].set_title("Projected LTV Distribution", fontweight="bold")
axes[1].set_xlabel("Projected LTV ($)")
 
seg_counts = df["ltv_segment"].value_counts().sort_index()
colors_seg = ["#90CAF9", "#42A5F5", "#1E88E5", "#1565C0"]
axes[2].bar(seg_counts.index, seg_counts.values,
            color=colors_seg, edgecolor="white")
axes[2].set_title("Customers by LTV Segment", fontweight="bold")
axes[2].set_xlabel("LTV Segment")
axes[2].set_ylabel("Count")
for i, v in enumerate(seg_counts.values):
    axes[2].text(i, v + 20, str(v), ha="center", fontweight="bold")
 
plt.suptitle("LTV Distribution & Segmentation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_ltv_distribution.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/01_ltv_distribution.png")
 
# ── PLOT 2: LTV by Contract Type ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
sns.boxplot(data=df, x="contract", y="ltv_projected",
            palette=["#F44336", "#FF9800", "#4CAF50"], ax=axes[0])
axes[0].set_title("Projected LTV by Contract Type", fontweight="bold")
axes[0].set_xlabel("Contract Type")
axes[0].set_ylabel("Projected LTV ($)")
 
ltv_contract = df.groupby("contract")["ltv_projected"].mean().sort_values()
axes[1].bar(ltv_contract.index, ltv_contract.values,
            color=["#F44336", "#FF9800", "#4CAF50"], edgecolor="white")
axes[1].set_title("Mean Projected LTV by Contract Type", fontweight="bold")
axes[1].set_ylabel("Mean LTV ($)")
for i, (_, v) in enumerate(ltv_contract.items()):
    axes[1].text(i, v + 20, f"${v:,.0f}", ha="center", fontweight="bold")
 
plt.suptitle("LTV vs Contract Type", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_ltv_by_contract.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/02_ltv_by_contract.png")
 
# ── PLOT 3: LTV vs Churn Risk (Churn Segment) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
sns.boxplot(data=df, x="churn", y="ltv_projected",
            palette={"No": "#2196F3", "Yes": "#F44336"}, ax=axes[0])
axes[0].set_title("Projected LTV by Churn Status", fontweight="bold")
axes[0].set_xlabel("Churned?")
axes[0].set_ylabel("Projected LTV ($)")
 
# Scatter: LTV vs Tenure coloured by churn
sc = axes[1].scatter(
    df["tenure"], df["ltv_projected"],
    c=df["churn_binary"], cmap="RdBu_r",
    alpha=0.4, s=10, edgecolors="none",
)
plt.colorbar(sc, ax=axes[1], label="Churned (1=Yes)")
axes[1].set_title("LTV vs Tenure (coloured by churn)",
                  fontweight="bold")
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Projected LTV ($)")
 
plt.suptitle("LTV vs Churn Risk", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_ltv_vs_churn.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/03_ltv_vs_churn.png")
 
# ── PLOT 4: LTV Segment × Churn Heatmap ──────────────────────────────────
pivot = pd.crosstab(
    df["ltv_segment"], df["churn"],
    normalize="index",
) * 100
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "% of segment"})
ax.set_title("Churn Rate (%) by LTV Segment\n"
             "(High-value customers who churn = biggest revenue loss)",
             fontweight="bold")
ax.set_xlabel("Churned?")
ax.set_ylabel("LTV Segment")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_ltv_segment_churn_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/04_ltv_segment_churn_heatmap.png")

# ════════════════════════════════════════════════════════════════════════════
# PART 3 — BUILD FEATURE MATRIX FOR REGRESSION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 3 — FEATURE MATRIX FOR REGRESSION")
print("=" * 65)
 
# Engineered features (from Week 2 Day 1-3)
ENGINEERED = [
    "charge_per_tenure",
    "service_count",
    "charge_to_value_ratio",
    "support_dependency_score",
    "tenure_contract_risk",
]
 
# Original numeric features
NUMERIC = ["tenure", "monthly_charges", "senior_citizen"]
 
# Encode categoricals
df["contract_enc"]   = df["contract"].map(
    {"Month-to-month": 0, "One year": 1, "Two year": 2})
df["gender_enc"]     = (df["gender"] == "Male").astype(int)
df["partner_enc"]    = (df["partner"] == "Yes").astype(int)
df["dependents_enc"] = (df["dependents"] == "Yes").astype(int)
df["paperless_enc"]  = (df["paperless_billing"] == "Yes").astype(int)
 
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
EXTRA_CATS  = ["contract_enc", "gender_enc", "partner_enc",
               "dependents_enc", "paperless_enc"]
 
df = pd.get_dummies(
    df,
    columns=["internet_service", "payment_method"],
    prefix=["internet", "payment"],
    drop_first=False,
)
ONEHOT = [c for c in df.columns
          if c.startswith("internet_") or c.startswith("payment_")]
 
# Note: total_charges EXCLUDED from features — it is the base of our target
ALL_FEATURES = ENGINEERED + NUMERIC + SERVICE_ENC + EXTRA_CATS + ONEHOT
 
X = df[ALL_FEATURES].copy()
y = df["ltv_projected"].copy()
 
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)
 
print(f"\nFeature matrix : {X.shape[0]} rows × {X.shape[1]} features")
print(f"Target         : ltv_projected  "
      f"(mean=${y.mean():,.2f}, std=${y.std():,.2f})")
print(f"Null values    : {X.isnull().sum().sum()}")
 
# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain : {X_train.shape[0]} rows")
print(f"Test  : {X_test.shape[0]} rows")
 
# Scale for Linear Regression
scaler_ltv     = StandardScaler()
X_train_scaled = scaler_ltv.fit_transform(X_train)
X_test_scaled  = scaler_ltv.transform(X_test)

# ════════════════════════════════════════════════════════════════════════════
# PART 4 — REGRESSION MODEL TRAINING
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 4 — REGRESSION MODEL TRAINING")
print("=" * 65)
 
reg_results = {}
 
def evaluate_regressor(name, y_true, y_pred):
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
 
# ── MODEL 1: Linear Regression ────────────────────────────────────────────
print("\n--- Model 1: Linear Regression ---")
lr_reg = LinearRegression()
lr_reg.fit(X_train_scaled, y_train)
y_pred_lr = lr_reg.predict(X_test_scaled)
metrics_lr = evaluate_regressor("Linear Regression", y_test, y_pred_lr)
reg_results["Linear Regression"] = {
    "model": lr_reg, "y_pred": y_pred_lr, **metrics_lr}
print(f"  MAE  : ${metrics_lr['mae']:,.2f}")
print(f"  RMSE : ${metrics_lr['rmse']:,.2f}")
print(f"  R²   : {metrics_lr['r2']:.4f}")
print(f"  MAPE : {metrics_lr['mape']:.2f}%")
joblib.dump(lr_reg,      f"{MODELS_DIR}/ltv_linear_regression.pkl")
joblib.dump(scaler_ltv,  f"{MODELS_DIR}/ltv_scaler.pkl")
print(f"  Saved: {MODELS_DIR}/ltv_linear_regression.pkl")
 
# ── MODEL 2: Random Forest Regressor ─────────────────────────────────────
print("\n--- Model 2: Random Forest Regressor ---")
rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
metrics_rf = evaluate_regressor("Random Forest", y_test, y_pred_rf)
reg_results["Random Forest"] = {
    "model": rf_reg, "y_pred": y_pred_rf, **metrics_rf}
print(f"  MAE  : ${metrics_rf['mae']:,.2f}")
print(f"  RMSE : ${metrics_rf['rmse']:,.2f}")
print(f"  R²   : {metrics_rf['r2']:.4f}")
print(f"  MAPE : {metrics_rf['mape']:.2f}%")
joblib.dump(rf_reg, f"{MODELS_DIR}/ltv_random_forest.pkl")
print(f"  Saved: {MODELS_DIR}/ltv_random_forest.pkl")
 
# ── MODEL 3: XGBoost Regressor ────────────────────────────────────────────
print("\n--- Model 3: XGBoost Regressor ---")
xgb_reg = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,        # L1 regularisation
    reg_lambda=1.0,       # L2 regularisation
    random_state=42,
    verbosity=0,
)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)
metrics_xgb = evaluate_regressor("XGBoost", y_test, y_pred_xgb)
reg_results["XGBoost"] = {
    "model": xgb_reg, "y_pred": y_pred_xgb, **metrics_xgb}
print(f"  MAE  : ${metrics_xgb['mae']:,.2f}")
print(f"  RMSE : ${metrics_xgb['rmse']:,.2f}")
print(f"  R²   : {metrics_xgb['r2']:.4f}")
print(f"  MAPE : {metrics_xgb['mape']:.2f}%")
joblib.dump(xgb_reg, f"{MODELS_DIR}/ltv_xgboost.pkl")
print(f"  Saved: {MODELS_DIR}/ltv_xgboost.pkl")
 
# ── Cross-validation (R² score) ───────────────────────────────────────────
print("\n5-Fold Cross-Validation (R²):")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for name, v in reg_results.items():
    if name == "Linear Regression":
        cv_scores = cross_val_score(
            v["model"], X_train_scaled, y_train, cv=kf, scoring="r2")
    else:
        cv_scores = cross_val_score(
            v["model"], X_train, y_train, cv=kf, scoring="r2")
    print(f"  {name:<22}: mean R²={cv_scores.mean():.4f}  "
          f"std={cv_scores.std():.4f}")

# ════════════════════════════════════════════════════════════════════════════
# PART 5 — REGRESSION EVALUATION PLOTS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 5 — EVALUATION PLOTS")
print("=" * 65)

model_names = list(reg_results.keys())
COLORS = {
    "Linear Regression": "#2196F3",
    "Random Forest"    : "#4CAF50",
    "XGBoost"          : "#F44336",
}

# ── PLOT 5: Actual vs Predicted (all 3 models) ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, v) in zip(axes, reg_results.items()):
    ax.scatter(y_test, v["y_pred"],
               alpha=0.3, s=8, color=COLORS[name], edgecolors="none")
    lims = [min(y_test.min(), v["y_pred"].min()),
            max(y_test.max(), v["y_pred"].max())]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect prediction")
    ax.set_title(f"{name}\nR²={v['r2']:.4f}  MAE=${v['mae']:,.0f}",
                 fontweight="bold")
    ax.set_xlabel("Actual LTV ($)")
    ax.set_ylabel("Predicted LTV ($)")
    ax.legend(fontsize=8)

plt.suptitle("Actual vs Predicted LTV — All 3 Models",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_actual_vs_predicted.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/05_actual_vs_predicted.png")

# ── PLOT 6: Residuals Distribution ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, v) in zip(axes, reg_results.items()):
    residuals = y_test.values - v["y_pred"]
    ax.hist(residuals, bins=50, color=COLORS[name],
            edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(f"{name}\nResidual Distribution",
                 fontweight="bold")
    ax.set_xlabel("Residual = Actual - Predicted ($)")
    ax.set_ylabel("Count")

plt.suptitle("Residuals Distribution — All 3 Models",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_residuals.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/06_residuals.png")

# ── PLOT 7: Metrics Comparison ────────────────────────────────────────────
metric_keys   = ["mae", "rmse", "r2", "mape"]
metric_labels = ["MAE ($)", "RMSE ($)", "R² Score", "MAPE (%)"]
bar_colors    = [COLORS[m] for m in model_names]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
    vals = [reg_results[m][key] for m in model_names]
    bars = axes[i].bar(
        [m.replace(" ", "\n") for m in model_names],
        vals, color=bar_colors, edgecolor="white", width=0.5,
    )
    axes[i].set_title(label, fontweight="bold")
    axes[i].set_ylabel(label)
    for bar, val in zip(bars, vals):
        fmt = f"${val:,.0f}" if key in ["mae","rmse"] else f"{val:.3f}"
        axes[i].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() * 1.01,
            fmt, ha="center", fontweight="bold", fontsize=9,
        )

plt.suptitle("Regression Model Performance Comparison",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_regression_metrics.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Saved: {OUTPUT_DIR}/07_regression_metrics.png")

# ════════════════════════════════════════════════════════════════════════════
# PART 6 — GENERATE LTV PREDICTIONS FOR ALL CUSTOMERS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 6 — GENERATE LTV PREDICTIONS FOR ALL CUSTOMERS")
print("=" * 65)

# Use best model (XGBoost by default — highest R² expected)
best_name   = max(reg_results, key=lambda m: reg_results[m]["r2"])
best_model  = reg_results[best_name]["model"]
print(f"\nBest model by R² : {best_name}")

X_all       = df[ALL_FEATURES].copy()
bool_all    = X_all.select_dtypes(include="bool").columns
X_all[bool_all] = X_all[bool_all].astype(int)

df["ltv_predicted"] = best_model.predict(X_all).round(2)
df["ltv_predicted"] = df["ltv_predicted"].clip(lower=0)  # no negative LTV

print(f"\nLTV Predictions Summary:")
print(f"  Mean  : ${df['ltv_predicted'].mean():,.2f}")
print(f"  Median: ${df['ltv_predicted'].median():,.2f}")
print(f"  Min   : ${df['ltv_predicted'].min():,.2f}")
print(f"  Max   : ${df['ltv_predicted'].max():,.2f}")

print("\nMean Predicted LTV by Churn Status:")
print(df.groupby("churn")["ltv_predicted"].mean().round(2).to_string())

print("\nMean Predicted LTV by Contract Type:")
print(df.groupby("contract")["ltv_predicted"].mean().round(2).to_string())

 