# 🔄 Churn-LTV Engine

> **Customer Churn Prediction & Lifetime Value Engine**  
> A production-level data analytics project built for telecommunications businesses.  
> Predicts which customers are at risk of leaving and segments them by value — so retention efforts go where they matter most.

---

## 📌 Project Overview

This engine ingests historical customer data from a telecom provider and answers two critical business questions:

1. **Who is about to leave?** — Classification models predict churn risk for every active customer
2. **Who is worth saving?** — LTV calculations prioritise which customers to spend retention budget on

---

## 📊 Dataset

**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle / IBM Watson)

| Property | Details |
|---|---|
| Rows | 7,043 customers |
| Columns | 21 features |
| Target | `Churn` (Yes / No) |
| Churn Rate | ~26.5% |

**Key columns:** `tenure`, `monthly_charges`, `total_charges`, `contract`, `internet_service`, `payment_method`, and 8 service subscription columns.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11, SQL |
| Database | PostgreSQL 18 |
| Data layer | SQLAlchemy, pandas |
| Machine Learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | matplotlib, seaborn |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
churn-ltv-engine/
│
├── load_data.py                  # Week 1 Day 1-2 : DB setup & data ingestion
├── eda_analysis.py               # Week 1 Day 3-5 : Exploratory data analysis
├── data_preprocessing.py         # Week 1 Day 6-7 : Cleaning, encoding, baseline report
│
├── feature_engineering.py        # Week 2 Day 1-3 : 5 engineered features
├── model_training.py             # Week 2 Day 4-6 : LR, RF, XGBoost training & eval
├── shap_explainability.py        # Week 2 Day 7   : SHAP values & business explainability
│
├── eda_outputs/                  # EDA plots (9 PNG files)
├── preprocessing_outputs/        # Baseline report plots
├── feature_engineering_outputs/  # Feature distribution plots
├── model_outputs/                # Confusion matrices, ROC curves, importance plots
├── shap_outputs/                 # SHAP summary, waterfall, business dashboard
│
├── saved_models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── .env                          # Local DB credentials (never committed)
├── .gitignore
└── README.md
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 18
- Git

### 1. Clone the repository

```bash
git clone https://github.com/ZaalimaDA/churn-ltv-engine.git
cd churn-ltv-engine
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas sqlalchemy psycopg2-binary python-dotenv kaggle \
            seaborn matplotlib scipy scikit-learn xgboost shap joblib
```

### 4. Set up PostgreSQL

Open psql and run:

```sql
CREATE DATABASE telco_churn_db;
CREATE USER churn_admin WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE telco_churn_db TO churn_admin;
```

### 5. Create your `.env` file

```
DB_USER=churn_admin
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432
DB_NAME=telco_churn_db
```

### 6. Download the dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root.

### 7. Run the pipeline in order

```bash
python load_data.py            # Loads data into PostgreSQL
python eda_analysis.py         # Runs EDA and saves plots
python data_preprocessing.py   # Cleans data and builds baseline report
python feature_engineering.py  # Engineers 5 new features
python model_training.py       # Trains 3 models and evaluates them
python shap_explainability.py  # Generates SHAP explanations
```

---

## 🗃️ PostgreSQL Tables

| Table | Created By | Contents |
|---|---|---|
| `customers` | `load_data.py` | Raw cleaned dataset (7,043 rows) |
| `customers_processed` | `data_preprocessing.py` | Encoded version of customers |
| `customers_features` | `feature_engineering.py` | Original + 5 engineered features |
| `model_metrics` | `model_training.py` | Precision, Recall, F1, ROC-AUC per model |
| `model_predictions` | `model_training.py` | Test set predictions with churn probability |
| `shap_feature_importance` | `shap_explainability.py` | Global SHAP ranking per feature |
| `shap_customer_values` | `shap_explainability.py` | Per-customer SHAP scores (1,000 sample) |

---

## 🔬 What Has Been Built (Week 1 & 2)

### Week 1 — Data Foundation

#### Day 1-2 · Data Ingestion
- PostgreSQL database and schema set up
- Telco dataset loaded via SQLAlchemy
- Column names normalised to snake_case
- Blank `total_charges` values handled (11 rows with `tenure=0`)

#### Day 3-5 · Exploratory Data Analysis
10 focused analyses covering:
- Overall churn distribution (26.5% churn rate)
- Churn by contract type — month-to-month: ~42%, two-year: ~3%
- Tenure vs churn — churned avg: 18 months, retained avg: 38 months
- Monthly charges, senior citizen segment, categorical features
- Correlation heatmap across all numeric columns

**Key finding:** Contract type and tenure are the two strongest individual predictors of churn in the dataset.

#### Day 6-7 · Preprocessing & Baseline Report
- Missing values in `total_charges` filled with 0 (correct for new customers)
- Binary encoding for all Yes/No columns
- Service columns simplified (No internet service → No)
- Label encoding for contract type (0/1/2 — preserves natural order)
- One-hot encoding for internet service and payment method
- Preprocessed data saved to `customers_processed` table
- Baseline analytics report generated with 6 key business metrics

---

### Week 2 — Modelling

#### Day 1-3 · Feature Engineering
5 new features created from the raw data:

| Feature | Formula | Business Meaning |
|---|---|---|
| `charge_per_tenure` | `total_charges / (tenure + 1)` | Spend relative to loyalty — high = risk |
| `service_count` | Count of subscribed services (0-8) | Platform stickiness — high = safer |
| `charge_to_value_ratio` | `monthly_charges / (service_count + 1)` | Price per service — high = poor value perception |
| `support_dependency_score` | Count of support services (0-4) | Safety-net lock-in — high = safer |
| `tenure_contract_risk` | `(1 - tenure_norm) × (1 - contract_norm)` | Composite risk — new + month-to-month = highest |

All 5 features were validated against churn using Pearson correlation before saving to `customers_features`.

#### Day 4-6 · Model Training & Evaluation

Three classification models trained and compared:

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| **XGBoost** | — | — | — | — |

> Run `python model_training.py` to populate this table with your actual results.

**Class imbalance handling:**
- Logistic Regression & Random Forest: `class_weight=balanced`
- XGBoost: `scale_pos_weight = neg_count / pos_count (~2.77)`

**Evaluation plots generated:**
- Confusion matrices for all 3 models
- ROC curves with AUC values
- Metrics bar chart comparison
- Feature importances (engineered features highlighted)
- 5-fold cross-validation box plot
- Engineered feature importance spotlight

#### Day 7 · SHAP Explainability

SHAP (SHapley Additive exPlanations) applied to the best-performing XGBoost model:

- **Global:** which features drive churn across all customers
- **Local:** exactly why the model flagged a specific customer as high-risk
- Engineered features validated — `tenure_contract_risk` expected as top engineered signal

**7 SHAP plots generated:**

| Plot | What It Shows |
|---|---|
| Beeswarm summary | Global importance + direction for every feature and customer |
| Global bar chart | Mean SHAP importance ranked, engineered features in red |
| Engineered spotlight | SHAP importance for the 5 engineered features only |
| Dependence plots | How top 3 engineered features' values affect churn probability |
| Waterfall — high risk | Step-by-step explanation for a specific churner |
| Waterfall — low risk | Step-by-step explanation for a specific retained customer |
| Business dashboard | One-page stakeholder summary combining all key insights |

---

## 📈 Key Business Findings (Week 1 EDA)

| Metric | Value |
|---|---|
| Overall churn rate | 26.5% |
| Month-to-month contract churn | ~42% |
| Two-year contract churn | ~3% |
| Avg tenure of churned customers | ~18 months |
| Avg tenure of retained customers | ~38 months |
| Monthly revenue at risk | ~$139,130/month |
| High-risk segment churn | ~58% (month-to-month + tenure ≤ 12 months) |

---
