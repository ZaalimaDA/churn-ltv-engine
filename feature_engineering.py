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

# ════════════════════════════════════════════════════════════════════════════
# FEATURE 4: support_dependency_score
# ════════════════════════════════════════════════════════════════════════════
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