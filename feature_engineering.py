import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")
 
# 0. SETUP 
load_dotenv()
 
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
 
sns.set_theme(style="whitegrid", palette="muted")
OUTPUT_DIR = "feature_engineering_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
df = pd.read_sql("SELECT * FROM customers", engine)
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")
 
# Normalize service columns upfront
service_all = ["phone_service", "multiple_lines", "online_security",
               "online_backup", "device_protection", "tech_support",
               "streaming_tv", "streaming_movies"]
for col in service_all:
    df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

# Feature 1: charge_per_tenure
print("=" * 60)
print("FEATURE 1 — charge_per_tenure")
print("=" * 60)
 
df["charge_per_tenure"] = df["total_charges"] / (df["tenure"] + 1)
 
print(f"\nFormula  : total_charges / (tenure + 1)")
print(f"Min/Max  : {df['charge_per_tenure'].min():.2f} / {df['charge_per_tenure'].max():.2f}")
print(f"Mean by churn:\n{df.groupby('churn')['charge_per_tenure'].mean().round(2)}")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for val, color, label in [("No", "#2196F3", "Retained"), ("Yes", "#F44336", "Churned")]:
    df[df["churn"] == val]["charge_per_tenure"].plot.kde(ax=axes[0], color=color, label=label, linewidth=2.5)
axes[0].set_title("charge_per_tenure Distribution by Churn")
axes[0].set_xlabel("Charge per Tenure ($)")
axes[0].legend()
sns.boxplot(data=df, x="churn", y="charge_per_tenure", ax=axes[1], palette={"No": "#2196F3", "Yes": "#F44336"})
axes[1].set_title("charge_per_tenure Spread by Churn")
plt.suptitle("Feature 1: charge_per_tenure", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_charge_per_tenure.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/01_charge_per_tenure.png")

# Feature 2 : Service Count
print("\n" + "=" * 60)
print("FEATURE 2 — service_count")
print("=" * 60)
 
df["service_count"] = df[service_all].apply(lambda row: (row == "Yes").sum(), axis=1)
 
print(f"\nFormula  : count of subscribed add-on services (0-8)")
print(f"Min/Max  : {df['service_count'].min()} / {df['service_count'].max()}")
print(f"Mean by churn:\n{df.groupby('churn')['service_count'].mean().round(2)}")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc_churn = df.groupby(["service_count", "churn"]).size().unstack(fill_value=0)
sc_pct = sc_churn.div(sc_churn.sum(axis=1), axis=0) * 100
sc_pct.plot(kind="bar", stacked=True, ax=axes[0], color=["#2196F3", "#F44336"], edgecolor="white", rot=0)
axes[0].set_title("Churn Rate by Service Count")
axes[0].set_xlabel("Number of Services")
axes[0].legend(title="Churn", labels=["No", "Yes"])
sns.boxplot(data=df, x="churn", y="service_count", ax=axes[1], palette={"No": "#2196F3", "Yes": "#F44336"})
axes[1].set_title("Service Count Spread by Churn")
plt.suptitle("Feature 2: service_count", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_service_count.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/02_service_count.png")

# Feature 3: charge_to_value_ratio
print("\n" + "=" * 60)
print("FEATURE 3 — charge_to_value_ratio")
print("=" * 60)
 
df["charge_to_value_ratio"] = df["monthly_charges"] / (df["service_count"] + 1)
 
print(f"\nFormula  : monthly_charges / (service_count + 1)")
print(f"Min/Max  : {df['charge_to_value_ratio'].min():.2f} / {df['charge_to_value_ratio'].max():.2f}")
print(f"Mean by churn:\n{df.groupby('churn')['charge_to_value_ratio'].mean().round(2)}")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for val, color, label in [("No", "#2196F3", "Retained"), ("Yes", "#F44336", "Churned")]:
    df[df["churn"] == val]["charge_to_value_ratio"].plot.kde(ax=axes[0], color=color, label=label, linewidth=2.5)
axes[0].set_title("charge_to_value_ratio by Churn")
axes[0].set_xlabel("$/service")
axes[0].legend()
sns.boxplot(data=df, x="churn", y="charge_to_value_ratio", ax=axes[1], palette={"No": "#2196F3", "Yes": "#F44336"})
axes[1].set_title("Charge-to-Value Spread by Churn")
plt.suptitle("Feature 3: charge_to_value_ratio", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_charge_to_value_ratio.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/03_charge_to_value_ratio.png")

#Feature 4: support_dependency_score
print("\n" + "=" * 60)
print("FEATURE 4 — support_dependency_score")
print("=" * 60)
 
support_cols = ["online_security", "device_protection", "tech_support", "online_backup"]
df["support_dependency_score"] = df[support_cols].apply(lambda row: (row == "Yes").sum(), axis=1)
 
print(f"\nFormula  : count of support/protection services (0-4)")
print(f"Min/Max  : {df['support_dependency_score'].min()} / {df['support_dependency_score'].max()}")
print(f"Mean by churn:\n{df.groupby('churn')['support_dependency_score'].mean().round(2)}")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sd_churn = df.groupby(["support_dependency_score", "churn"]).size().unstack(fill_value=0)
sd_pct = sd_churn.div(sd_churn.sum(axis=1), axis=0) * 100
sd_pct.plot(kind="bar", stacked=True, ax=axes[0], color=["#2196F3", "#F44336"], edgecolor="white", rot=0)
axes[0].set_title("Churn Rate by Support Dependency Score")
axes[0].set_xlabel("Support Services (0-4)")
axes[0].legend(title="Churn", labels=["No", "Yes"])
sns.boxplot(data=df, x="churn", y="support_dependency_score", ax=axes[1], palette={"No": "#2196F3", "Yes": "#F44336"})
axes[1].set_title("Support Dependency Spread by Churn")
plt.suptitle("Feature 4: support_dependency_score", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_support_dependency_score.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/04_support_dependency_score.png")

#Feature 5: tenure_contract_risk
print("\n" + "=" * 60)
print("FEATURE 5 — tenure_contract_risk")
print("=" * 60)
 
df["tenure_norm"]   = df["tenure"] / df["tenure"].max()
contract_map        = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["contract_norm"] = df["contract"].map(contract_map) / 2
df["tenure_contract_risk"] = (1 - df["tenure_norm"]) * (1 - df["contract_norm"])
 
print(f"\nFormula  : (1 - tenure_norm) * (1 - contract_norm)")
print(f"Range    : {df['tenure_contract_risk'].min():.3f} to {df['tenure_contract_risk'].max():.3f}")
print(f"Mean by churn:\n{df.groupby('churn')['tenure_contract_risk'].mean().round(3)}")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for val, color, label in [("No", "#2196F3", "Retained"), ("Yes", "#F44336", "Churned")]:
    df[df["churn"] == val]["tenure_contract_risk"].plot.kde(ax=axes[0], color=color, label=label, linewidth=2.5)
axes[0].set_title("tenure_contract_risk Distribution by Churn")
axes[0].set_xlabel("Risk Score (0=low, 1=high)")
axes[0].legend()
sns.boxplot(data=df, x="churn", y="tenure_contract_risk", ax=axes[1], palette={"No": "#2196F3", "Yes": "#F44336"})
axes[1].set_title("Tenure-Contract Risk Spread by Churn")
plt.suptitle("Feature 5: tenure_contract_risk", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_tenure_contract_risk.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/05_tenure_contract_risk.png")

# Correlation Summary
print("\n" + "=" * 60)
print("FEATURE CORRELATION WITH CHURN")
print("=" * 60)
 
df["churn_binary"] = (df["churn"] == "Yes").astype(int)
engineered = ["charge_per_tenure", "service_count", "charge_to_value_ratio",
              "support_dependency_score", "tenure_contract_risk"]
 
correlations = df[engineered + ["churn_binary"]].corr()["churn_binary"].drop("churn_binary")
print("\nPearson Correlation with Churn:")
print(correlations.sort_values(key=abs, ascending=False).round(4))
 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
corr_matrix = df[engineered + ["churn_binary"]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=axes[0], vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
axes[0].set_title("Engineered Feature Correlation Matrix", fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)
 
colors = ["#F44336" if c > 0 else "#2196F3" for c in correlations]
axes[1].barh(correlations.index, correlations.values, color=colors, edgecolor="white")
axes[1].axvline(x=0, color="black", linewidth=0.8)
axes[1].set_title("Feature Correlation with Churn", fontweight="bold")
axes[1].set_xlabel("Pearson Correlation Coefficient")
for i, v in enumerate(correlations.values):
    axes[1].text(v + (0.005 if v >= 0 else -0.005), i, f"{v:.3f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=9)
plt.suptitle("Engineered Feature Correlations with Churn", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_feature_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {OUTPUT_DIR}/06_feature_correlations.png")

# Save to PostgreSQL
df_save = df.drop(columns=["tenure_norm", "contract_norm", "churn_binary"], errors="ignore")
df_save.to_sql("customers_features", engine, if_exists="replace", index=False)
 
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM customers_features")).scalar()
 
print(f"\n✅ Saved to table: customers_features | Rows: {count} | Cols: {df_save.shape[1]}")

