import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

df = pd.read_sql("SELECT * FROM customers_processed", engine)
print(f"✅ Loaded {df.shape[0]} rows from customers_processed\n")

# ── Dependency check ──────────────────────────────────────────────────────────
if "service_count" not in df.columns:
    raise ValueError("❌ service_count column missing. Run feature_02_service_count.py first.")

# ── Feature 3: charge_to_value_ratio ─────────────────────────────────────────
print("=" * 60)
print("FEATURE 3 — CHARGE TO VALUE RATIO")
print("=" * 60)

df["charge_to_value_ratio"] = (
    df["monthly_charges"] / (df["service_count"] + 1)
).round(4)

print(f"\ncharge_to_value_ratio created ✅")
print(f"   min  : {df['charge_to_value_ratio'].min():.2f}")
print(f"   max  : {df['charge_to_value_ratio'].max():.2f}")
print(f"   mean : {df['charge_to_value_ratio'].mean():.2f}")

by_churn = df.groupby("churn_encoded")["charge_to_value_ratio"].mean().round(4)
print(f"\nAvg charge_to_value_ratio by churn:")
print(f"   Retained (0): {by_churn.get(0, 'N/A')}")
print(f"   Churned  (1): {by_churn.get(1, 'N/A')}")


df.to_sql("customers_processed", engine, if_exists="replace", index=False)

with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM customers_processed")).scalar()
    print(f"\n✅ Saved to customers_processed — row count: {count}")