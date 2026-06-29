import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

# ── SETUP ────────────────────────────────────────────────────────────────────
load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

sns.set_theme(style="whitegrid", palette="muted")
PALETTE    = {"No": "#2196F3", "Yes": "#F44336"}
OUTPUT_DIR = "feature_engineering_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_sql("SELECT * FROM customers", engine)
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)
print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")

# Normalize service columns upfront
SERVICE_COLS = [
    "phone_service", "multiple_lines", "online_security",
    "online_backup",  "device_protection", "tech_support",
    "streaming_tv",   "streaming_movies"
]
for col in SERVICE_COLS:
    df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

# ── REUSABLE HELPERS ─────────────────────────────────────────────────────────

def print_feature(name, formula, col):
    """Print feature stats — min, max, mean by churn."""
    print(f"\nFormula  : {formula}")
    print(f"Min/Max  : {df[col].min():.2f} / {df[col].max():.2f}")
    print(f"Mean by churn:\n{df.groupby('churn')[col].mean().round(3)}")

def plot_kde_box(col, title, xlabel, filename):
    """KDE + boxplot side by side — used for continuous features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for val, color, label in [("No", "#2196F3", "Retained"), ("Yes", "#F44336", "Churned")]:
        df[df["churn"] == val][col].plot.kde(
            ax=axes[0], color=color, label=label, linewidth=2.5)
    axes[0].set_title(f"{col} Distribution by Churn")
    axes[0].set_xlabel(xlabel)
    axes[0].legend()
    sns.boxplot(data=df, x="churn", y=col, ax=axes[1], palette=PALETTE)
    axes[1].set_title(f"{col} Spread by Churn")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {OUTPUT_DIR}/{filename}")

def plot_bar_box(col, title, xlabel, filename):
    """Stacked bar + boxplot — used for count-based features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ct = df.groupby([col, "churn"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", stacked=True, ax=axes[0],
                color=["#2196F3", "#F44336"], edgecolor="white", rot=0)
    axes[0].set_title(f"Churn Rate by {col}")
    axes[0].set_xlabel(xlabel)
    axes[0].legend(title="Churn", labels=["No", "Yes"])
    sns.boxplot(data=df, x="churn", y=col, ax=axes[1], palette=PALETTE)
    axes[1].set_title(f"{col} Spread by Churn")
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {OUTPUT_DIR}/{filename}")

def section(n, name):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"FEATURE {n} — {name}")
    print("=" * 60)

# ── FEATURE 1: charge_per_tenure ─────────────────────────────────────────────
section(1, "charge_per_tenure")
df["charge_per_tenure"] = df["total_charges"] / (df["tenure"] + 1)
print_feature("charge_per_tenure", "total_charges / (tenure + 1)", "charge_per_tenure")
plot_kde_box("charge_per_tenure", "Feature 1: charge_per_tenure",
             "Charge per Tenure ($)", "01_charge_per_tenure.png")

# ── FEATURE 2: service_count ──────────────────────────────────────────────────
section(2, "service_count")
df["service_count"] = df[SERVICE_COLS].apply(lambda row: (row == "Yes").sum(), axis=1)
print_feature("service_count", "count of subscribed services (0-8)", "service_count")
plot_bar_box("service_count", "Feature 2: service_count",
             "Number of Services", "02_service_count.png")

# ── FEATURE 3: charge_to_value_ratio ─────────────────────────────────────────
section(3, "charge_to_value_ratio")
df["charge_to_value_ratio"] = (df["monthly_charges"] / (df["service_count"] + 1)).round(4)
print_feature("charge_to_value_ratio", "monthly_charges / (service_count + 1)", "charge_to_value_ratio")
plot_kde_box("charge_to_value_ratio", "Feature 3: charge_to_value_ratio",
             "$/service", "03_charge_to_value_ratio.png")

# ── FEATURE 4: support_dependency_score ──────────────────────────────────────
section(4, "support_dependency_score")
SUPPORT_COLS = ["online_security", "device_protection", "tech_support", "online_backup"]
df["support_dependency_score"] = df[SUPPORT_COLS].apply(lambda row: (row == "Yes").sum(), axis=1)
print_feature("support_dependency_score", "count of support/protection services (0-4)", "support_dependency_score")
plot_bar_box("support_dependency_score", "Feature 4: support_dependency_score",
             "Support Services (0-4)", "04_support_dependency_score.png")

# ── FEATURE 5: tenure_contract_risk ──────────────────────────────────────────
section(5, "tenure_contract_risk")
contract_map           = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["tenure_norm"]      = df["tenure"] / df["tenure"].max()
df["contract_norm"]    = df["contract"].map(contract_map) / 2
df["tenure_contract_risk"] = (1 - df["tenure_norm"]) * (1 - df["contract_norm"])
print_feature("tenure_contract_risk", "(1 - tenure_norm) * (1 - contract_norm)", "tenure_contract_risk")
plot_kde_box("tenure_contract_risk", "Feature 5: tenure_contract_risk",
             "Risk Score (0=low, 1=high)", "05_tenure_contract_risk.png")

# ── CORRELATION SUMMARY ───────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("FEATURE CORRELATION WITH CHURN")
print("=" * 60)

df["churn_binary"] = (df["churn"] == "Yes").astype(int)
ENGINEERED = [
    "charge_per_tenure", "service_count", "charge_to_value_ratio",
    "support_dependency_score", "tenure_contract_risk"
]

correlations = df[ENGINEERED + ["churn_binary"]].corr()["churn_binary"].drop("churn_binary")
print("\nPearson Correlation with Churn:")
print(correlations.sort_values(key=abs, ascending=False).round(4))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(df[ENGINEERED + ["churn_binary"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, ax=axes[0],
            vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
axes[0].set_title("Engineered Feature Correlation Matrix", fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)

colors = ["#F44336" if c > 0 else "#2196F3" for c in correlations]
axes[1].barh(correlations.index, correlations.values, color=colors, edgecolor="white")
axes[1].axvline(x=0, color="black", linewidth=0.8)
axes[1].set_title("Feature Correlation with Churn", fontweight="bold")
axes[1].set_xlabel("Pearson Correlation Coefficient")
for i, v in enumerate(correlations.values):
    axes[1].text(v + (0.005 if v >= 0 else -0.005), i, f"{v:.3f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=9)
plt.suptitle("Engineered Feature Correlations with Churn", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {OUTPUT_DIR}/06_feature_correlations.png")

# ── SAVE TO POSTGRESQL ────────────────────────────────────────────────────────
df_save = df.drop(columns=["tenure_norm", "contract_norm", "churn_binary"], errors="ignore")
df_save.to_sql("customers_features", engine, if_exists="replace", index=False)

with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM customers_features")).scalar()

print(f"\n✅ Saved to table: customers_features | Rows: {count} | Cols: {df_save.shape[1]}")