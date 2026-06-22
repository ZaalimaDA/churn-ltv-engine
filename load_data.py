import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load credentials from .env
load_dotenv()

DB_USER     = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = os.getenv("DB_HOST")
DB_PORT     = os.getenv("DB_PORT")
DB_NAME     = os.getenv("DB_NAME")

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Load CSV
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 1: Lowercase all columns first
df.columns = [c.lower() for c in df.columns]

# Step 2: Explicit rename for camelCase columns
df.rename(columns={
    "customerid":      "customer_id",
    "seniorcitizen":   "senior_citizen",
    "phoneservice":    "phone_service",
    "multiplelines":   "multiple_lines",
    "internetservice": "internet_service",
    "onlinesecurity":  "online_security",
    "onlinebackup":    "online_backup",
    "deviceprotection":"device_protection",
    "techsupport":     "tech_support",
    "streamingtv":     "streaming_tv",
    "streamingmovies": "streaming_movies",
    "paperlessbilling":"paperless_billing",
    "paymentmethod":   "payment_method",
    "monthlycharges":  "monthly_charges",
    "totalcharges":    "total_charges"
}, inplace=True)

# Step 3: Convert total_charges to numeric
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

print(f"Shape: {df.shape}")
print(df.columns.tolist())
print(df.head(3))

# Load into PostgreSQL
df.to_sql("customers", engine, if_exists="replace", index=False)
print("✅ Data loaded successfully into telco_churn_db!")