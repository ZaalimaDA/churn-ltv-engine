import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

#Setup
load_dotenv()
 
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
 
# Load data from PostgreSQL
df = pd.read_sql("SELECT * FROM customers", engine)
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns from PostgreSQL\n")
 
# Set plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
 
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#1-Overview
print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape       : {df.shape}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDuplicates  : {df.duplicated().sum()}")

#2-Churn Distribution
print("\n" + "=" * 60)
print("2. CHURN DISTRIBUTION")
print("=" * 60)
 
churn_counts = df["churn"].value_counts()
churn_pct    = df["churn"].value_counts(normalize=True) * 100
print(f"\n{pd.DataFrame({'Count': churn_counts, 'Percentage': churn_pct.round(2)})}")
 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
# Bar chart
axes[0].bar(churn_counts.index, churn_counts.values,
            color=["#2196F3", "#F44336"], edgecolor="white", width=0.5)
axes[0].set_title("Churn Count")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")
 
# Pie chart
axes[1].pie(churn_counts.values, labels=churn_counts.index,
            autopct="%1.1f%%", colors=["#2196F3", "#F44336"],
            startangle=90, wedgeprops={"edgecolor": "white"})
axes[1].set_title("Churn Proportion")
 
plt.suptitle("Overall Churn Distribution", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_churn_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Plot saved: {OUTPUT_DIR}/01_churn_distribution.png")

#Contract Type VS Churn
print("\n" + "=" * 60)
print("3. CONTRACT TYPE vs CHURN")
print("=" * 60)
 
contract_churn = df.groupby(["contract", "churn"]).size().unstack(fill_value=0)
contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
print(f"\nChurn Rate by Contract Type:\n{contract_churn_pct.round(2)}")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Stacked bar - absolute counts
contract_churn.plot(kind="bar", ax=axes[0], color=["#2196F3", "#F44336"],
                    edgecolor="white", rot=0)
axes[0].set_title("Churn Count by Contract Type")
axes[0].set_xlabel("Contract Type")
axes[0].set_ylabel("Number of Customers")
axes[0].legend(title="Churn", labels=["No", "Yes"])
 
# Stacked bar - percentage
contract_churn_pct.plot(kind="bar", stacked=True, ax=axes[1],
                        color=["#2196F3", "#F44336"], edgecolor="white", rot=0)
axes[1].set_title("Churn Rate (%) by Contract Type")
axes[1].set_xlabel("Contract Type")
axes[1].set_ylabel("Percentage (%)")
axes[1].legend(title="Churn", labels=["No", "Yes"])
 
# Add percentage labels
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.1f%%", label_type="center",
                      fontsize=10, color="white", fontweight="bold")
 
plt.suptitle("Contract Type vs Churn", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_contract_vs_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot saved: {OUTPUT_DIR}/02_contract_vs_churn.png")