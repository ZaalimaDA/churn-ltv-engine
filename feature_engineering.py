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

