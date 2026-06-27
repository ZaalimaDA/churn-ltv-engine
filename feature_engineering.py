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

df["charge_to_value_ratio"] = (df["monthly_charges"] / (df["service_count"] + 1)).round(4)

print(f"\nFormula  : monthly_charges / (service_count + 1)")
print(f"Min/Max  : {df['charge_to_value_ratio'].min():.2f} / {df['charge_to_value_ratio'].max():.2f}")
print(f"Mean by churn:\n{df.groupby('churn')['charge_to_value_ratio'].mean().round(2)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for val, color, label in [("No", "#2196F3", "Retained"), ("Yes", "#F44336", "Churned")]:
    df[df["churn"] == val]["charge_to_value_ratio"].plot.kde(
        ax=axes[0], color=color, label=label, linewidth=2.5)
axes[0].set_title("charge_to_value_ratio Distribution by Churn")
axes[0].set_xlabel("Charge to Value Ratio ($)")
axes[0].legend()
sns.boxplot(data=df, x="churn", y="charge_to_value_ratio", ax=axes[1],
            palette={"No": "#2196F3", "Yes": "#F44336"})
axes[1].set_title("charge_to_value_ratio Spread by Churn")
plt.suptitle("Feature 3: charge_to_value_ratio", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_charge_to_value_ratio.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved: {OUTPUT_DIR}/03_charge_to_value_ratio.png")
