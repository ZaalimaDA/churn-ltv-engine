## Import Libraries

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")

### Connect PostgreSQL

load_dotenv()

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

print("✓ Connected Successfully")


### Load Dataset

df = pd.read_sql(
    "SELECT * FROM customers_features",
    engine
)

print("="*60)
print("LOGISTIC REGRESSION")
print("="*60)

print(df.shape)
df.head()

### Verify Engineered Features

ENGINEERED = [
    "charge_per_tenure",
    "service_count",
    "charge_to_value_ratio",
    "support_dependency_score",
    "tenure_contract_risk"
]

print("Checking engineered features...")

missing = [col for col in ENGINEERED if col not in df.columns]

if len(missing)==0:
    print("✓ All engineered features found")
else:
    print("Missing Features :",missing)



### Data Preparation

df["total_charges"] = pd.to_numeric(
    df["total_charges"],
    errors="coerce"
).fillna(0)

df["churn_binary"] = (
    df["churn"]=="Yes"
).astype(int)



### Encode Categorical Variables
NUMERIC = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "senior_citizen"
]

df["contract_enc"] = df["contract"].map(
    {
        "Month-to-month":0,
        "One year":1,
        "Two year":2
    }
)

df["gender_enc"] = (
    df["gender"]=="Male"
).astype(int)

df["partner_enc"] = (
    df["partner"]=="Yes"
).astype(int)

df["dependents_enc"] = (
    df["dependents"]=="Yes"
).astype(int)

df["paperless_enc"] = (
    df["paperless_billing"]=="Yes"
).astype(int)

df = pd.get_dummies(
    df,
    columns=["internet_service","payment_method"],
    prefix=["internet","payment"],
    drop_first=False
)


### Prepare Features
ALL_FEATURES = (
    ENGINEERED +
    NUMERIC +
    [
        "contract_enc",
        "gender_enc",
        "partner_enc",
        "dependents_enc",
        "paperless_enc"
    ] +
    [c for c in df.columns if c.startswith("internet_")] +
    [c for c in df.columns if c.startswith("payment_")]
)

X = df[ALL_FEATURES]

y = df["churn_binary"]

print(X.shape)

### Train Test Split
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("Train :",X_train.shape)
print("Test  :",X_test.shape)


### Standard Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

print("✓ Scaling Completed")

###  Train Logistic Regression

lr = LogisticRegression(

    C=1.0,

    max_iter=1000,

    class_weight="balanced",

    random_state=42,

    solver="lbfgs"

)

lr.fit(
    X_train_scaled,
    y_train
)

print("✓ Logistic Regression Trained")


### Prediction
y_pred = lr.predict(
    X_test_scaled
)

y_prob = lr.predict_proba(
    X_test_scaled
)[:,1]


### Model Evaluation

accuracy = accuracy_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

f1 = f1_score(y_test,y_pred)

roc = roc_auc_score(y_test,y_prob)

print("="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc:.4f}")



### Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.show()


### Classification Report

print(
    classification_report(
        y_test,
        y_pred
    )
)

### ROC Curve

RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

plt.title(
    "ROC Curve"
)

plt.show()


### Feature Importance (Coefficients)
coef = pd.DataFrame({

    "Feature":ALL_FEATURES,

    "Coefficient":lr.coef_[0]

})

coef = coef.sort_values(
    "Coefficient",
    ascending=False
)

coef.head(15)


### Save Model
os.makedirs(
    "saved_models",
    exist_ok=True
)

joblib.dump(
    lr,
    "saved_models/logistic_regression.pkl"
)

joblib.dump(
    scaler,
    "saved_models/scaler.pkl"
)

print("✓ Model Saved")




### final report 
print("=" * 60)
print("LOGISTIC REGRESSION REPORT")
print("=" * 60)

print(f"Dataset Rows            : {len(df)}")
print(f"Total Features          : {X.shape[1]}")
print(f"Training Samples        : {len(X_train)}")
print(f"Testing Samples         : {len(X_test)}")

print("\nModel Performance")
print("-" * 40)

print(f"Accuracy               : {accuracy:.4f}")
print(f"Precision              : {precision:.4f}")
print(f"Recall                 : {recall:.4f}")
print(f"F1 Score               : {f1:.4f}")
print(f"ROC-AUC                : {roc:.4f}")

print("\nConfusion Matrix Summary")
print("-" * 40)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"True Positives         : {tp}")
print(f"True Negatives         : {tn}")
print(f"False Positives        : {fp}")
print(f"False Negatives        : {fn}")

print("\nTop Positive Predictors")
print("-" * 40)

top_positive = coef.sort_values("Coefficient", ascending=False).head(5)

for _, row in top_positive.iterrows():
    print(f"{row['Feature']:30s} : {row['Coefficient']:.4f}")

print("\nBusiness Findings")
print("-" * 40)
print("• Logistic Regression was successfully trained using engineered and customer profile features.")
print("• Class imbalance was handled using class_weight='balanced', improving churn detection.")
print(f"• The model achieved an ROC-AUC of {roc:.2f}, indicating good discrimination between churn and non-churn customers.")
print(f"• Recall of {recall:.2f} shows the model successfully identified approximately {recall*100:.1f}% of actual churners.")
print(f"• Precision of {precision:.2f} indicates that approximately {precision*100:.1f}% of predicted churners were actual churners.")
print("• Feature importance analysis identified 'service_count', 'internet_Fiber optic', and 'tenure_contract_risk' as the strongest positive predictors of churn.")
print("• This Logistic Regression model provides an interpretable baseline and will be compared with Random Forest and XGBoost in the next phase.")

print("\nModel Artifacts")
print("-" * 40)
print("✓ logistic_regression.pkl saved")
print("✓ scaler.pkl saved")

print("\n✓ Logistic Regression Pipeline Completed Successfully.")