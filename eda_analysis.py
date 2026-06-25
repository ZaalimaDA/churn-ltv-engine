import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
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

# 1. BASIC OVERVIEW 
print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape       : {df.shape}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDuplicates  : {df.duplicated().sum()}")

# 2. TARGET VARIABLE: CHURN DISTRIBUTION 
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

# 3. CONTRACT TYPE vs CHURN ─────────────────────────────────────────────────
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

# 4. TENURE vs CHURN
print("\n" + "=" * 60)
print("4. TENURE vs CHURN")
print("=" * 60)

tenure_stats = df.groupby("churn")["tenure"].describe()
print(f"\nTenure Statistics by Churn:\n{tenure_stats}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
for churn_val, color, label in [("No", "#2196F3", "Not Churned"),
                                  ("Yes", "#F44336", "Churned")]:
    axes[0].hist(df[df["churn"] == churn_val]["tenure"],
                 bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
axes[0].set_title("Tenure Distribution by Churn")
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Count")
axes[0].legend()

# KDE Plot
for churn_val, color in [("No", "#2196F3"), ("Yes", "#F44336")]:
    subset = df[df["churn"] == churn_val]["tenure"]
    subset.plot.kde(ax=axes[1], color=color,
                    label=f"Churn={churn_val}", linewidth=2)
axes[1].set_title("Tenure Density by Churn")
axes[1].set_xlabel("Tenure (months)")
axes[1].legend()

# Box plot
sns.boxplot(data=df, x="churn", y="tenure", ax=axes[2],
            palette={"No": "#2196F3", "Yes": "#F44336"})
axes[2].set_title("Tenure Spread by Churn")
axes[2].set_xlabel("Churn")
axes[2].set_ylabel("Tenure (months)")

plt.suptitle("Tenure vs Churn", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_tenure_vs_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot saved: {OUTPUT_DIR}/03_tenure_vs_churn.png")

# 5. TENURE SEGMENTS 
print("\n" + "=" * 60)
print("5. TENURE SEGMENTS vs CHURN")
print("=" * 60)

df["tenure_group"] = pd.cut(df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"])

seg_churn = df.groupby(["tenure_group", "churn"], observed=True).size().unstack(fill_value=0)
seg_churn_pct = seg_churn.div(seg_churn.sum(axis=1), axis=0) * 100
print(f"\nChurn Rate by Tenure Group:\n{seg_churn_pct.round(2)}")

fig, ax = plt.subplots(figsize=(10, 5))
seg_churn_pct.plot(kind="bar", stacked=True, ax=ax,
                   color=["#2196F3", "#F44336"], edgecolor="white", rot=15)
ax.set_title("Churn Rate by Tenure Segment")
ax.set_xlabel("Tenure Group")
ax.set_ylabel("Percentage (%)")
ax.legend(title="Churn", labels=["No", "Yes"])

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", label_type="center",
                 fontsize=10, color="white", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_tenure_segments_vs_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot saved: {OUTPUT_DIR}/04_tenure_segments_vs_churn.png")

# 6. CONTRACT + TENURE HEATMAP
print("\n" + "=" * 60)
print("6. CONTRACT TYPE + TENURE SEGMENT HEATMAP")
print("=" * 60)

pivot = df[df["churn"] == "Yes"].groupby(
    ["contract", "tenure_group"], observed=True).size().unstack(fill_value=0)
pivot_pct = pivot.div(
    df.groupby(["contract", "tenure_group"], observed=True).size().unstack(fill_value=0)
) * 100
print(f"\nChurn Rate % (Contract x Tenure):\n{pivot_pct.round(2)}")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Churn Rate (%)"})
ax.set_title("Churn Rate (%) — Contract Type × Tenure Segment", pad=15)
ax.set_xlabel("Tenure Group")
ax.set_ylabel("Contract Type")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_contract_tenure_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot saved: {OUTPUT_DIR}/05_contract_tenure_heatmap.png")

# 7. MONTHLY CHARGES vs CHURN 
print("\n" + "=" * 60)
print("7. MONTHLY CHARGES vs CHURN")
print("=" * 60)

mc_stats = df.groupby("churn")["monthly_charges"].describe()
print(f"\nMonthly Charges by Churn:\n{mc_stats}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(data=df, x="monthly_charges", hue="churn", bins=40,
             palette={"No": "#2196F3", "Yes": "#F44336"}, ax=axes[0], alpha=0.7)
axes[0].set_title("Monthly Charges Distribution by Churn")
axes[0].set_xlabel("Monthly Charges ($)")

sns.boxplot(data=df, x="churn", y="monthly_charges", ax=axes[1],
            palette={"No": "#2196F3", "Yes": "#F44336"})
axes[1].set_title("Monthly Charges Spread by Churn")
axes[1].set_xlabel("Churn")
axes[1].set_ylabel("Monthly Charges ($)")

plt.suptitle("Monthly Charges vs Churn", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_monthly_charges_vs_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot saved: {OUTPUT_DIR}/06_monthly_charges_vs_churn.png")

# 8. CATEGORICAL FEATURES vs CHURN 
print("\n" + "=" * 60)
print("8. KEY CATEGORICAL FEATURES vs CHURN")
print("=" * 60)

cat_features = ["internet_service", "payment_method", "paperless_billing", "partner"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    ct = df.groupby([col, "churn"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", stacked=True, ax=axes[i],
                color=["#2196F3", "#F44336"], edgecolor="white", rot=20)
    axes[i].set_title(f"{col.replace('_', ' ').title()} vs Churn")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Percentage (%)")
    axes[i].legend(title="Churn", labels=["No", "Yes"])
    for container in axes[i].containers:
        axes[i].bar_label(container, fmt="%.0f%%", label_type="center",
                          fontsize=9, color="white", fontweight="bold")

plt.suptitle("Categorical Features vs Churn", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_categorical_vs_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot saved: {OUTPUT_DIR}/07_categorical_vs_churn.png")

# 9. CORRELATION HEATMAP (NUMERIC FEATURES) 
print("\n" + "=" * 60)
print("9. CORRELATION HEATMAP")
print("=" * 60)

df_corr = df.copy()
df_corr["churn_binary"]       = (df_corr["churn"] == "Yes").astype(int)
df_corr["senior_citizen"]     = df_corr["senior_citizen"].astype(int)

numeric_cols = ["tenure", "monthly_charges", "total_charges",
                "senior_citizen", "churn_binary"]
corr_matrix = df_corr[numeric_cols].corr()
print(f"\nCorrelation Matrix:\n{corr_matrix.round(3)}")

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            mask=mask, square=True, linewidths=0.5, ax=ax,
            vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap — Numeric Features", pad=15)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot saved: {OUTPUT_DIR}/08_correlation_heatmap.png")

# 10. SENIOR CITIZEN vs CHURN 
print("\n" + "=" * 60)
print("10. SENIOR CITIZEN vs CHURN")
print("=" * 60)

sc_churn = df.groupby(["senior_citizen", "churn"]).size().unstack(fill_value=0)
sc_pct   = sc_churn.div(sc_churn.sum(axis=1), axis=0) * 100
sc_pct.index = ["Non-Senior", "Senior"]
print(f"\n{sc_pct.round(2)}")

fig, ax = plt.subplots(figsize=(7, 5))
sc_pct.plot(kind="bar", stacked=True, ax=ax,
            color=["#2196F3", "#F44336"], edgecolor="white", rot=0)
ax.set_title("Senior Citizen vs Churn Rate")
ax.set_xlabel("Customer Segment")
ax.set_ylabel("Percentage (%)")
ax.legend(title="Churn", labels=["No", "Yes"])
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", label_type="center",
                 fontsize=11, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_senior_citizen_vs_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot saved: {OUTPUT_DIR}/09_senior_citizen_vs_churn.png")

# 11. EDA SUMMARY INSIGHTS
print("\n" + "=" * 60)
print("EDA SUMMARY — KEY BUSINESS INSIGHTS")
print("=" * 60)

overall_churn_rate = (df["churn"] == "Yes").mean() * 100
mtm_churn = contract_churn_pct.loc["Month-to-month", "Yes"] if "Month-to-month" in contract_churn_pct.index else "N/A"
two_yr_churn = contract_churn_pct.loc["Two year", "Yes"] if "Two year" in contract_churn_pct.index else "N/A"
churned_avg_tenure   = df[df["churn"] == "Yes"]["tenure"].mean()
retained_avg_tenure  = df[df["churn"] == "No"]["tenure"].mean()
churned_avg_charges  = df[df["churn"] == "Yes"]["monthly_charges"].mean()
retained_avg_charges = df[df["churn"] == "No"]["monthly_charges"].mean()

print(f"""
  Overall Churn Rate         : {overall_churn_rate:.1f}%
  Month-to-Month Churn Rate  : {mtm_churn:.1f}%
  Two-Year Contract Churn    : {two_yr_churn:.1f}%

  Avg Tenure (Churned)       : {churned_avg_tenure:.1f} months
  Avg Tenure (Retained)      : {retained_avg_tenure:.1f} months

  Avg Monthly Charges (Churned)   : ${churned_avg_charges:.2f}
  Avg Monthly Charges (Retained)  : ${retained_avg_charges:.2f}
""")