import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
import warnings

warnings.filterwarnings("ignore")

# ==========================
# DATABASE CONNECTION
# ==========================

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)


# ==========================
# LOAD DATA
# ==========================

df = pd.read_sql("SELECT * FROM customers", engine)

df = df.rename(columns={
    "monthlycharges": "monthly_charges",
    "totalcharges": "total_charges",
    "paperlessbilling": "paperless_billing",
    "paymentmethod": "payment_method",
    "internetservice": "internet_service",
    "seniorcitizen": "senior_citizen",
    "phoneservice": "phone_service",
    "multiplelines": "multiple_lines",
    "onlinesecurity": "online_security",
    "onlinebackup": "online_backup",
    "deviceprotection": "device_protection",
    "techsupport": "tech_support",
    "streamingtv": "streaming_tv",
    "streamingmovies": "streaming_movies"
})

print(f"\nLoaded {df.shape[0]} rows and {df.shape[1]} columns")

# ==========================
# DATA CLEANING
# ==========================

df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

if "total_charges" in df.columns:
    df["total_charges"] = pd.to_numeric(
        df["total_charges"],
        errors="coerce"
    )

os.makedirs("eda_outputs", exist_ok=True)

sns.set_style("whitegrid")

# ==========================
# BASIC OVERVIEW
# ==========================

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

# ==========================
# CHURN DISTRIBUTION
# ==========================

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="churn"
)

plt.title("Customer Churn Distribution")

plt.savefig(
    "eda_outputs/01_churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# CONTRACT VS CHURN
# ==========================

contract_churn = pd.crosstab(
    df["contract"],
    df["churn"],
    normalize="index"
) * 100

contract_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Contract Type vs Churn")
plt.ylabel("Percentage")

plt.savefig(
    "eda_outputs/02_contract_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# TENURE VS CHURN
# ==========================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="churn",
    y="tenure"
)

plt.title("Tenure vs Churn")

plt.savefig(
    "eda_outputs/03_tenure_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# MONTHLY CHARGES VS CHURN
# ==========================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="churn",
    y="monthly_charges"
)

plt.title("Monthly Charges vs Churn")

plt.savefig(
    "eda_outputs/04_monthly_charges_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# TENURE SEGMENTS
# ==========================

df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0,12,24,48,72],
    labels=[
        "0-12",
        "13-24",
        "25-48",
        "49-72"
    ]
)

tenure_churn = pd.crosstab(
    df["tenure_group"],
    df["churn"],
    normalize="index"
) * 100

tenure_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(8,5)
)

plt.title("Tenure Segment vs Churn")

plt.savefig(
    "eda_outputs/05_tenure_segment_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# CATEGORICAL FEATURES
# ==========================

cat_cols = [
    "internet_service",
    "payment_method",
    "paperless_billing",
    "partner"
]

for col in cat_cols:

    plt.figure(figsize=(8,5))

    tmp = pd.crosstab(
        df[col],
        df["churn"],
        normalize="index"
    ) * 100

    tmp.plot(
        kind="bar",
        stacked=True
    )

    plt.title(f"{col} vs Churn")

    plt.savefig(
        f"eda_outputs/{col}_vs_churn.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ==========================
# CORRELATION HEATMAP
# ==========================

df["churn_binary"] = (
    df["churn"] == "Yes"
).astype(int)

numeric_cols = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "churn_binary"
]

corr = df[numeric_cols].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "eda_outputs/06_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# BUSINESS INSIGHTS
# ==========================

overall_churn = (
    (df["churn"] == "Yes")
    .mean()
    * 100
)

avg_tenure_churn = df[
    df["churn"] == "Yes"
]["tenure"].mean()

avg_tenure_no = df[
    df["churn"] == "No"
]["tenure"].mean()

print("\n" + "="*60)
print("BUSINESS INSIGHTS")
print("="*60)

print(f"\nOverall Churn Rate: {overall_churn:.2f}%")
print(f"Average Tenure (Churned): {avg_tenure_churn:.2f}")
print(f"Average Tenure (Retained): {avg_tenure_no:.2f}")

print("\nEDA Completed Successfully")
print("Plots saved in ./eda_outputs/")
