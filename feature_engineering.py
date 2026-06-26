import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# =====================================================
# STEP 1 — CONNECT TO POSTGRESQL
# =====================================================

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# =====================================================
# STEP 2 — LOAD PREPROCESSED DATA
# =====================================================

print("=" * 60)
print("FEATURE ENGINEERING — FEATURE 1")
print("=" * 60)

df = pd.read_sql(
    "SELECT * FROM customers_processed",
    engine
)

print(f"Dataset Loaded Successfully")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# =====================================================
# STEP 3 — CREATE FEATURE
# =====================================================

if "charge_per_tenure" in df.columns:
    print("\nExisting 'charge_per_tenure' found. Recalculating feature...")
else:
    print("\nCreating Feature: charge_per_tenure")

# Always calculate (create or overwrite)
df["charge_per_tenure"] = (
    df["total_charges"] /
    (df["tenure"] + 1)
)

print("✓ Feature created successfully.")

# =====================================================
# STEP 4 — FEATURE VALIDATION
# =====================================================

print("\nFeature Statistics")
print("-" * 40)

print(df["charge_per_tenure"].describe().round(2))

print("\nMissing Values :", df["charge_per_tenure"].isnull().sum())
print("Duplicate Rows :", df.duplicated().sum())

# =====================================================
# STEP 5 — SAVE UPDATED DATASET
# =====================================================

df.to_sql(
    "customers_processed",
    engine,
    if_exists="replace",
    index=False,
    method="multi"
)

print("\n✓ Updated dataset saved successfully.")

# Verify saved data
verify_df = pd.read_sql(
    "SELECT charge_per_tenure FROM customers_processed LIMIT 5",
    engine
)

print("\nVerification (First 5 Values)")
print(verify_df)

# =====================================================
# FINAL FINDINGS
# =====================================================

print("\n" + "=" * 60)
print("Feature Engineering Pipeline Completed Successfully")
print("FEATURE ENGINEERING REPORT")
print("=" * 60)

print(f"Feature Name               : charge_per_tenure")
print(f"Formula                    : total_charges / (tenure + 1)")
print(f"Total Customers            : {len(df)}")
print(f"Average Value              : {df['charge_per_tenure'].mean():.2f}")
print(f"Median Value               : {df['charge_per_tenure'].median():.2f}")
print(f"Minimum Value              : {df['charge_per_tenure'].min():.2f}")
print(f"Maximum Value              : {df['charge_per_tenure'].max():.2f}")

high_spenders = (
    df["charge_per_tenure"] >
    df["charge_per_tenure"].quantile(0.75)
).sum()

print(f"Top 25% High Charge-per-Tenure Customers : {high_spenders}")

print("\nBusiness Findings:")
print("-" * 40)
print("• Successfully engineered the 'charge_per_tenure' feature by combining customer tenure and total spending.")
print("• The feature captures customer spending intensity throughout their relationship with the company.")
print("• Customers with higher charge_per_tenure have contributed more revenue relative to their tenure and can be analyzed as potential high-value customers.")
print("• Extremely high values may occur for customers with short tenure and high initial charges, making them important candidates for further churn analysis.")
print("• This engineered feature provides additional business information beyond using total_charges or tenure independently.")
print("• The feature will be included in the machine learning model to evaluate its contribution to customer churn prediction.")

print("\n✓ Feature Engineering completed successfully.")
