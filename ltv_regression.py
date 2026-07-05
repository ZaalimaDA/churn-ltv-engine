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
