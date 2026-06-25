import pandas as pd

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values with median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

print("Missing Values:")
print(df.isnull().sum())

# Convert column names to lowercase
df.columns = [c.lower() for c in df.columns]

# Rename columns
df.rename(columns={
    "customerid": "customer_id",
    "seniorcitizen": "senior_citizen",
    "phoneservice": "phone_service",
    "multiplelines": "multiple_lines",
    "internetservice": "internet_service",
    "onlinesecurity": "online_security",
    "onlinebackup": "online_backup",
    "deviceprotection": "device_protection",
    "techsupport": "tech_support",
    "streamingtv": "streaming_tv",
    "streamingmovies": "streaming_movies",
    "paperlessbilling": "paperless_billing",
    "paymentmethod": "payment_method",
    "monthlycharges": "monthly_charges",
    "totalcharges": "total_charges"
}, inplace=True)

print("\nColumns:")
print(df.columns.tolist())

# Encode target variable
df["churn"] = df["churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn Distribution:")
print(df["churn"].value_counts())

# Encode gender
df["gender"] = df["gender"].map({
    "Female": 0,
    "Male": 1
})

# Encode Yes/No columns
binary_cols = [
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing"
]

for col in binary_cols:
    df[col] = df[col].map({
        "No": 0,
        "Yes": 1
    })

print("\nEncoded Sample:")
print(df[[
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing"
]].head())