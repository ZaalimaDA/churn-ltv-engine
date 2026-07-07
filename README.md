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
| API | FastAPI, Uvicorn |
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
├── ltv_regression.py             # Week 3 Day 1-4 : LTV target + regression models
├── main.py                       # Week 3 Day 5-7 : FastAPI entry point
│
├── app/
│   ├── core/
│   │   ├── model_loader.py       # Loads all 4 model artefacts at startup
│   │   └── feature_builder.py    # Builds churn & LTV feature vectors from raw input
│   ├── routers/
│   │   ├── single.py             # POST /predict/churn, /ltv, /churn-and-ltv
│   │   ├── batch.py              # POST /predict/batch
│   │   └── lookup.py             # GET /customer/{id}, /metrics
│   └── schemas/
│       └── customer.py           # Pydantic request/response models
│
├── ltv_outputs/                  # LTV EDA + model evaluation plots (7 PNG files)
├── eda_outputs/                  # EDA plots
├── model_outputs/                # Confusion matrices, ROC curves, importance plots
├── shap_outputs/                 # SHAP summary, waterfall, business dashboard
│
├── saved_models/
│   ├── xgboost.pkl               # Churn classifier
│   ├── scaler.pkl                # Churn scaler
│   ├── ltv_xgboost.pkl           # LTV regressor (best model)
│   ├── ltv_linear_regression.pkl
│   ├── ltv_random_forest.pkl
│   └── ltv_scaler.pkl
│
├── requirements.txt
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

### 1. Clone & install

```bash
git clone https://github.com/ZaalimaDA/churn-ltv-engine.git
cd churn-ltv-engine
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure PostgreSQL & `.env`

```sql
CREATE DATABASE telco_churn_db;
CREATE USER churn_admin WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE telco_churn_db TO churn_admin;
```

```
DB_USER=churn_admin  |  DB_PASSWORD=yourpassword
DB_HOST=localhost    |  DB_PORT=5432  |  DB_NAME=telco_churn_db
```

### 3. Run the pipeline in order

```bash
python load_data.py            # Ingest raw data → PostgreSQL
python eda_analysis.py         # EDA plots
python data_preprocessing.py   # Clean, encode, baseline report
python feature_engineering.py  # Engineer 5 new features
python model_training.py       # Train churn classifiers
python shap_explainability.py  # SHAP explanations
python ltv_regression.py       # LTV target + regression models
uvicorn main:app --reload      # Start FastAPI (http://localhost:8000/docs)
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
| `customers_ltv` | `ltv_regression.py` | Historical, projected & predicted LTV per customer |
| `ltv_model_metrics` | `ltv_regression.py` | MAE, RMSE, R², MAPE for each regression model |

---

## 🔬 What Has Been Built

### Week 1 — Data Foundation *(summarised)*

- **Data Ingestion:** PostgreSQL schema set up; Telco dataset (7,043 rows, 21 cols) loaded via SQLAlchemy; column names normalised; 11 blank `total_charges` rows handled.
- **EDA (10 analyses):** Churn rate = 26.5%; month-to-month contracts churn at ~42% vs two-year at ~3%; churned customers avg 18-month tenure vs 38-month for retained; correlation heatmap across all numerics.
- **Preprocessing:** Binary encoding for all Yes/No fields; `No internet service` → `No`; label encoding for contract type; one-hot for internet service & payment method; `customers_processed` table saved; 6-metric baseline report generated.

**Key finding:** Contract type and tenure are the two strongest individual predictors of churn.

---

### Week 2 — Churn Modelling *(summarised)*

- **Feature Engineering (5 features):** `charge_per_tenure`, `service_count`, `charge_to_value_ratio`, `support_dependency_score`, `tenure_contract_risk` — all validated via Pearson correlation; saved to `customers_features`.
- **Model Training:** Logistic Regression, Random Forest, XGBoost trained on 80/20 split; class imbalance handled (`class_weight=balanced` / `scale_pos_weight ≈ 2.77`); 5-fold CV; 6 evaluation plots generated.
- **SHAP Explainability:** XGBoost explained globally (beeswarm, bar chart) and locally (waterfall per customer); `tenure_contract_risk` confirmed as top engineered signal; business dashboard produced.

---

### Week 3 — LTV Regression & FastAPI

#### Day 1-4 · LTV Target Construction & Regression Models

**Target variable (`ltv_projected`)** blends what each customer has already paid with a forward-looking projection:

```
ltv_projected = total_charges + monthly_charges × projected_months
```

| Contract | Projected Months |
|---|---|
| Month-to-month | 6 (conservative — ~50% churn within 6 mo) |
| One year | 12 |
| Two year | 24 |

Values capped at the 99th percentile to reduce outlier pull. Customers segmented into `Low / Medium / High / Premium` quartiles for business use and stored in `customers_ltv`.

**4 regression models** trained and compared on an 80/20 split (LTV target excludes `total_charges` as a feature to prevent data leakage):

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | — | — | — | — |
| Ridge Regression | — | — | — | — |
| Random Forest Regressor | — | — | — | — |
| **XGBoost Regressor** *(best)* | — | — | — | — |

> Run `python ltv_regression.py` to populate actual metrics.

**XGBoost configuration:** 300 estimators, depth=6, learning rate=0.05, L1 + L2 regularisation, 5-fold CV on R².

**7 evaluation plots saved to `ltv_outputs/`:**

| Plot | What It Shows |
|---|---|
| `01_ltv_distribution.png` | Historical LTV, projected LTV distributions + segment bar chart |
| `02_ltv_by_contract.png` | Projected LTV boxplot & mean bar chart by contract type |
| `03_ltv_vs_churn.png` | LTV by churn status + LTV vs tenure scatter coloured by churn |
| `04_ltv_segment_churn_heatmap.png` | Churn rate % per LTV segment — quantifies revenue-at-risk |
| `05_actual_vs_predicted.png` | Actual vs predicted LTV for all 3 models with R² / MAE |
| `06_residuals.png` | Residual distributions for all 3 models |
| `07_regression_metrics.png` | Side-by-side bar chart: MAE, RMSE, R², MAPE |

All trained models serialised to `saved_models/` with `joblib`.

---

#### Day 5-7 · FastAPI Inference Service

A production-ready REST API (`main.py`) that loads all 4 model artefacts once at startup (via FastAPI lifespan) and serves real-time churn + LTV predictions.

**Architecture:**

```
main.py  →  app/routers/single.py   (single-customer endpoints)
         →  app/routers/batch.py    (batch CSV inference)
         →  app/routers/lookup.py   (DB lookup & metrics)
         →  app/core/model_loader.py  (loads xgboost.pkl, ltv_xgboost.pkl, scalers)
         →  app/core/feature_builder.py  (builds feature vectors from raw JSON input)
         →  app/schemas/customer.py  (Pydantic validation)
```

**Endpoints:**

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict/churn` | Churn probability + risk label (Low/Medium/High/Critical) |
| `POST` | `/predict/ltv` | Predicted LTV ($) + segment (Low/Medium/High/Premium) |
| `POST` | `/predict/churn-and-ltv` | Combined: churn + LTV + revenue-at-risk + priority score |
| `POST` | `/predict/batch` | Batch inference from CSV upload |
| `GET` | `/customer/{id}` | Lookup stored predictions for a customer by ID |
| `GET` | `/metrics` | Retrieve model performance metrics from DB |

**Priority scoring logic** (`/predict/churn-and-ltv`):

| Churn Risk | LTV | Priority |
|---|---|---|
| ≥ 50% | ≥ $2,000 | **Critical** |
| ≥ 50% | < $2,000 | High |
| < 50% | ≥ $2,000 | Medium |
| < 50% | < $2,000 | Low |

**Revenue at Risk** = `ltv_predicted × churn_probability` — tells the business the expected dollar loss if no retention action is taken.

**Start the API:**

```bash
uvicorn main:app --reload
# Swagger UI → http://localhost:8000/docs
```

---

## 📈 Key Business Findings

| Metric | Value |
|---|---|
| Overall churn rate | 26.5% |
| Month-to-month contract churn | ~42% |
| Two-year contract churn | ~3% |
| Avg tenure — churned customers | ~18 months |
| Avg tenure — retained customers | ~38 months |
| Monthly revenue at risk | ~$139,130/month |
| High-risk segment churn | ~58% (month-to-month + tenure ≤ 12 months) |

---