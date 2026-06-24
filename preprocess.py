import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# DATABASE CONNECTION
# =====================================================

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

print("Loading data from PostgreSQL...")

df = pd.read_sql(
    "SELECT * FROM customers",
    engine
)

print(f"Dataset Shape: {df.shape}")

# =====================================================
# STANDARDIZE COLUMN NAMES
# =====================================================

df.columns = [col.strip().lower() for col in df.columns]

print("\nColumns:")
print(df.columns.tolist())

# =====================================================
# DATA QUALITY CHECK
# =====================================================

print("\n===== DATA QUALITY =====")

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# =====================================================
# HANDLE TOTAL CHARGES
# =====================================================

if "totalcharges" in df.columns:

    df["totalcharges"] = pd.to_numeric(
        df["totalcharges"],
        errors="coerce"
    )

    print(
        f"\nMissing TotalCharges Before Fill: "
        f"{df['totalcharges'].isnull().sum()}"
    )

    df["totalcharges"] = df["totalcharges"].fillna(
        df["totalcharges"].median()
    )

    print(
        f"Missing TotalCharges After Fill: "
        f"{df['totalcharges'].isnull().sum()}"
    )

# =====================================================
# REMOVE CUSTOMER ID
# =====================================================

if "customerid" in df.columns:

    df.drop(
        columns=["customerid"],
        inplace=True
    )

    print("customerid removed")

# =====================================================
# ENCODE TARGET VARIABLE
# =====================================================

if "churn" in df.columns:

    df["churn"] = df["churn"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    print("churn encoded")

# =====================================================
# FIND CATEGORICAL COLUMNS
# =====================================================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()

# SAFETY CHECK
categorical_columns = [
    col
    for col in categorical_columns
    if col != "totalcharges"
]

print("\nCategorical Columns:")
print(categorical_columns)

# =====================================================
# ONE HOT ENCODING
# =====================================================

df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print(
    f"\nEncoded Dataset Shape: "
    f"{df_encoded.shape}"
)

# SAFETY CHECK
if df_encoded.shape[1] > 100:
    raise ValueError(
        f"Too many columns created ({df_encoded.shape[1]}). "
        f"Check totalcharges encoding."
    )

# =====================================================
# VERIFY NO OBJECT COLUMNS
# =====================================================

remaining_objects = df_encoded.select_dtypes(
    include="object"
).columns

print("\nRemaining Object Columns:")
print(list(remaining_objects))

# =====================================================
# SAVE CLEAN CSV
# =====================================================

os.makedirs("data", exist_ok=True)

csv_path = "data/cleaned_telco.csv"

df_encoded.to_csv(
    csv_path,
    index=False
)

print(f"\nClean CSV Saved: {csv_path}")

# =====================================================
# SAVE TO POSTGRESQL
# =====================================================

df_encoded.to_sql(
    "customers_cleaned",
    engine,
    if_exists="replace",
    index=False
)

print(
    "\nCleaned table saved to PostgreSQL: customers_cleaned"
)

# =====================================================
# BASELINE ANALYTICS
# =====================================================

print("\n===== BASELINE ANALYTICS =====")

churn_rate = round(
    df["churn"].mean() * 100,
    2
)

avg_tenure = round(
    df["tenure"].mean(),
    2
)

avg_monthly_charges = round(
    df["monthlycharges"].mean(),
    2
)

baseline_metrics = {
    "Rows": len(df),
    "Columns": len(df.columns),
    "Churn Rate (%)": churn_rate,
    "Average Tenure": avg_tenure,
    "Average Monthly Charges": avg_monthly_charges
}

for k, v in baseline_metrics.items():
    print(f"{k}: {v}")

# =====================================================
# SAVE REPORT
# =====================================================

os.makedirs("reports", exist_ok=True)

report_path = "reports/baseline_analytics_report.txt"

with open(report_path, "w") as f:

    f.write("BASELINE ANALYTICS REPORT\n")
    f.write("=" * 50 + "\n\n")

    for k, v in baseline_metrics.items():
        f.write(f"{k}: {v}\n")

    f.write("\nKEY INSIGHTS\n")
    f.write("- Month-to-month contracts show highest churn risk\n")
    f.write("- Customers with low tenure churn more frequently\n")
    f.write("- Higher monthly charges correlate with churn\n")
    f.write("- Fiber optic users tend to churn more\n")
    f.write("- Senior citizens show higher churn rates\n")

print(f"\nReport Saved: {report_path}")

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n===== FINAL SUMMARY =====")
print(f"Original Shape : {df.shape}")
print(f"Encoded Shape  : {df_encoded.shape}")

print("\nTop 10 Columns:")
print(df_encoded.columns[:10].tolist())

print("\nSUCCESS: Day 6-7 Completed")
print("Ready for Week 2 - Feature Engineering & Churn Prediction")