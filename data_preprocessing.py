import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")
 
# ── 0. SETUP ────────────────────────────────────────────────────────────────
load_dotenv()
 
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
 
sns.set_theme(style="whitegrid", palette="muted")
OUTPUT_DIR = "preprocessing_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
df = pd.read_sql("SELECT * FROM customers", engine)
print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns from PostgreSQL\n")

print("=" * 60)
print("PART 1 — MISSING VALUE ANALYSIS & TREATMENT")
print("=" * 60)

# 1 - Handle Missing Values 
# 1a. Identify missing values
print("\n1a. Missing Values Before Treatment:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct
}).query("`Missing Count` > 0")
print(missing_report if not missing_report.empty else "   No missing values found.")
 
# 1b. Visualize missing values (if any)
if not missing_report.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(missing_report.index, missing_report["Missing %"],
            color="#F44336", edgecolor="white")
    ax.set_title("Missing Values by Column (%)", fontweight="bold")
    ax.set_xlabel("Missing %")
    for i, v in enumerate(missing_report["Missing %"]):
        ax.text(v + 0.1, i, f"{v}%", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_missing_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved: {OUTPUT_DIR}/01_missing_values.png")
 
# 1c. Treat missing values
# total_charges is NULL for customers with tenure=0 (new customers, not billed yet)
# Strategy: fill with 0 (they have no charges yet)
print("\n1c. Treating Missing Values in total_charges...")
print(f"   Rows with NULL total_charges: {df['total_charges'].isnull().sum()}")
print(f"   Their tenure values: {df[df['total_charges'].isnull()]['tenure'].unique()}")
 
df["total_charges"] = df["total_charges"].fillna(0)
print(f"   Filled NULLs with 0 (new customers with tenure=0 have no charges yet)")
print(f"   Missing values after treatment: {df['total_charges'].isnull().sum()}")
 
# 1d. Verify no missing values remain
print("\n1d. Missing Values After Treatment:")
remaining = df.isnull().sum().sum()
print(f"   Total remaining nulls: {remaining} ✅" if remaining == 0
      else f"   ⚠ Still {remaining} nulls remaining — review!")

#2 - Encode Categorical Variables
print("\n" + "=" * 60)
print("PART 2 — CATEGORICAL ENCODING")
print("=" * 60)
 
# 2a. Identify categorical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\n2a. Categorical Columns ({len(cat_cols)}):")
for col in cat_cols:
    unique_vals = df[col].unique().tolist()
    print(f"   {col:25s} → {unique_vals}")
 
# 2b. Binary encoding (Yes/No → 1/0)
print("\n2b. Binary Encoding (Yes/No → 1/0):")
binary_cols = ["partner", "dependents", "phone_service",
               "paperless_billing", "churn"]
 
for col in binary_cols:
    if col in df.columns:
        df[f"{col}_encoded"] = (df[col] == "Yes").astype(int)
        print(f"   {col:25s} → {col}_encoded  {df[col].unique()} → {df[f'{col}_encoded'].unique()}")
 
# 2c. Binary encoding for gender
print("\n2c. Encoding Gender (Male=1, Female=0):")
df["gender_encoded"] = (df["gender"] == "Male").astype(int)
print(f"   gender → gender_encoded: Male=1, Female=0")
 
# 2d. Multi-value binary encoding (No/Yes/No internet service → simplify)
print("\n2d. Simplifying Service Columns (No internet service → No):")
service_cols = ["online_security", "online_backup", "device_protection",
                "tech_support", "streaming_tv", "streaming_movies", "multiple_lines"]
 
for col in service_cols:
    if col in df.columns:
        df[col] = df[col].replace("No internet service", "No")
        df[col] = df[col].replace("No phone service", "No")
        df[f"{col}_encoded"] = (df[col] == "Yes").astype(int)
        print(f"   {col:25s} → {col}_encoded")
 
# 2e. Label encoding for ordinal: contract type
print("\n2e. Label Encoding — Contract Type (ordinal):")
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["contract_encoded"] = df["contract"].map(contract_map)
print(f"   contract → contract_encoded: {contract_map}")
 
# 2f. One-hot encoding for nominal: internet_service, payment_method
print("\n2f. One-Hot Encoding — internet_service & payment_method:")
df = pd.get_dummies(df, columns=["internet_service", "payment_method"],
                    prefix=["internet", "payment"], drop_first=False)
new_cols = [c for c in df.columns if c.startswith("internet_") or c.startswith("payment_")]
print(f"   New columns created: {new_cols}")
 
# 2g. Show final encoded dataframe shape
print(f"\n2g. Dataframe shape after encoding: {df.shape}")
print(f"   Original: 7043 rows × 21 columns")
print(f"   After   : {df.shape[0]} rows × {df.shape[1]} columns")

#3 - Save Clean Encoded Data Back to PostgreSQL
print("\n" + "=" * 60)
print("PART 3 — SAVE PREPROCESSED DATA TO POSTGRESQL")
print("=" * 60)
 
# Save the fully preprocessed dataframe to a new table
df.to_sql("customers_processed", engine, if_exists="replace", index=False)
print(f"\n✅ Saved preprocessed data to table: customers_processed")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
 
# Verify in DB
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM customers_processed")).scalar()
    print(f"   PostgreSQL row count: {count} ✅")

#4 - Baseline Analytics Report
print("\n" + "=" * 60)
print("PART 4 — BASELINE ANALYTICS REPORT")
print("=" * 60)
 
# Reload clean original df for readable reporting
df_orig = pd.read_sql("SELECT * FROM customers", engine)
df_orig["total_charges"] = df_orig["total_charges"].fillna(0)
 
# ── Metric 1: Overall Churn Rate
churn_rate = (df_orig["churn"] == "Yes").mean() * 100
retained_rate = 100 - churn_rate
print(f"\n📌 Overall Churn Rate     : {churn_rate:.1f}%")
print(f"📌 Retention Rate         : {retained_rate:.1f}%")
 
# ── Metric 2: Churn by Contract
contract_churn = (df_orig.groupby("contract")["churn"]
                  .apply(lambda x: (x == "Yes").mean() * 100).round(1))
print(f"\n📌 Churn Rate by Contract:")
for contract, rate in contract_churn.items():
    print(f"   {contract:20s} : {rate}%")
 
# ── Metric 3: Avg Tenure
avg_tenure_churned  = df_orig[df_orig["churn"] == "Yes"]["tenure"].mean()
avg_tenure_retained = df_orig[df_orig["churn"] == "No"]["tenure"].mean()
print(f"\n📌 Avg Tenure (Churned)   : {avg_tenure_churned:.1f} months")
print(f"📌 Avg Tenure (Retained)  : {avg_tenure_retained:.1f} months")
 
# ── Metric 4: Revenue at Risk
monthly_revenue     = df_orig["monthly_charges"].sum()
churn_revenue_risk  = df_orig[df_orig["churn"] == "Yes"]["monthly_charges"].sum()
print(f"\n📌 Total Monthly Revenue  : ${monthly_revenue:,.2f}")
print(f"📌 Revenue Lost (Churned) : ${churn_revenue_risk:,.2f}")
print(f"📌 Revenue at Risk %      : {(churn_revenue_risk/monthly_revenue*100):.1f}%")
 
# ── Metric 5: High Risk Segment
high_risk = df_orig[
    (df_orig["contract"] == "Month-to-month") &
    (df_orig["tenure"] <= 12)
]
high_risk_churn = (high_risk["churn"] == "Yes").mean() * 100
print(f"\n📌 High-Risk Segment      : Month-to-month + Tenure ≤12 months")
print(f"   Customers in segment  : {len(high_risk)}")
print(f"   Churn rate in segment : {high_risk_churn:.1f}%")
 
# ── Metric 6: Senior Citizen churn
senior_churn = (df_orig[df_orig["senior_citizen"] == 1]["churn"] == "Yes").mean() * 100
non_senior_churn = (df_orig[df_orig["senior_citizen"] == 0]["churn"] == "Yes").mean() * 100
print(f"\n📌 Churn Rate (Seniors)   : {senior_churn:.1f}%")
print(f"📌 Churn Rate (Non-Senior): {non_senior_churn:.1f}%")

#5 - Baseline Report Dashboard Plot
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Baseline Analytics Report — Telco Customer Churn",
             fontsize=20, fontweight="bold", y=0.98)
 
# Plot 1: Overall churn pie
ax1 = fig.add_subplot(2, 3, 1)
ax1.pie([retained_rate, churn_rate], labels=["Retained", "Churned"],
        autopct="%1.1f%%", colors=["#2196F3", "#F44336"],
        startangle=90, wedgeprops={"edgecolor": "white"})
ax1.set_title("Overall Churn Rate", fontweight="bold")
 
# Plot 2: Churn rate by contract
ax2 = fig.add_subplot(2, 3, 2)
colors = ["#F44336" if r > 30 else "#FFA726" if r > 10 else "#66BB6A"
          for r in contract_churn.values]
bars = ax2.bar(contract_churn.index, contract_churn.values,
               color=colors, edgecolor="white", width=0.5)
ax2.set_title("Churn Rate by Contract Type", fontweight="bold")
ax2.set_ylabel("Churn Rate (%)")
ax2.set_ylim(0, 60)
for bar, val in zip(bars, contract_churn.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val}%", ha="center", fontweight="bold")
ax2.tick_params(axis="x", labelsize=9)
 
# Plot 3: Avg tenure comparison
ax3 = fig.add_subplot(2, 3, 3)
labels = ["Churned", "Retained"]
values = [avg_tenure_churned, avg_tenure_retained]
bars3 = ax3.bar(labels, values, color=["#F44336", "#2196F3"],
                edgecolor="white", width=0.5)
ax3.set_title("Average Tenure (months)", fontweight="bold")
ax3.set_ylabel("Months")
for bar, val in zip(bars3, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", fontweight="bold")
 
# Plot 4: Revenue breakdown
ax4 = fig.add_subplot(2, 3, 4)
revenue_data = [monthly_revenue - churn_revenue_risk, churn_revenue_risk]
ax4.pie(revenue_data, labels=["Active Revenue", "Revenue Lost to Churn"],
        autopct="%1.1f%%", colors=["#2196F3", "#F44336"],
        startangle=90, wedgeprops={"edgecolor": "white"})
ax4.set_title("Monthly Revenue at Risk", fontweight="bold")
 
# Plot 5: Tenure distribution by churn
ax5 = fig.add_subplot(2, 3, 5)
for churn_val, color, label in [("No", "#2196F3", "Retained"), ("Yes", "#F44336", "Churned")]:
    df_orig[df_orig["churn"] == churn_val]["tenure"].plot.kde(
        ax=ax5, color=color, label=label, linewidth=2.5)
ax5.set_title("Tenure Distribution by Churn", fontweight="bold")
ax5.set_xlabel("Tenure (months)")
ax5.legend()
ax5.fill_between(ax5.lines[0].get_xdata(), ax5.lines[0].get_ydata(), alpha=0.1, color="#2196F3")
ax5.fill_between(ax5.lines[1].get_xdata(), ax5.lines[1].get_ydata(), alpha=0.1, color="#F44336")
 
# Plot 6: Senior vs Non-senior
ax6 = fig.add_subplot(2, 3, 6)
seg_labels = ["Non-Senior", "Senior"]
seg_values = [non_senior_churn, senior_churn]
bars6 = ax6.bar(seg_labels, seg_values,
                color=["#42A5F5", "#EF5350"], edgecolor="white", width=0.5)
ax6.set_title("Churn Rate: Senior vs Non-Senior", fontweight="bold")
ax6.set_ylabel("Churn Rate (%)")
ax6.set_ylim(0, 60)
for bar, val in zip(bars6, seg_values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", fontweight="bold")
 
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUTPUT_DIR}/02_baseline_analytics_report.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Baseline Report Dashboard saved: {OUTPUT_DIR}/02_baseline_analytics_report.png")

print("\n Data Preprocessing completed!")