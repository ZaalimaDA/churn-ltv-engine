# 🔄 Churn-LTV Engine

> **Customer Churn Prediction & Lifetime Value Engine**  
> A production-level data analytics project built for telecommunications businesses.

---

## 📌 Project Overview
This engine is a complete, end-to-end Machine Learning pipeline that ingests historical customer data from a telecom provider, predicts churn, calculates Lifetime Value (LTV), and serves these predictions via a FastAPI backend and a Streamlit frontend dashboard.

## 🎯 Business Problem
Telecom companies lose millions annually due to customer churn. Retention budgets are limited, meaning they cannot afford to offer discounts to every single customer. 
This project solves two critical business questions to optimise retention strategy:
1. **Who is about to leave?** — Identifies high-risk customers before they cancel.
2. **Who is worth saving?** — Prioritises customers based on their projected Lifetime Value (LTV) so retention budgets are spent profitably.

## 📊 Dataset Source
**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle / IBM Watson)
- **Size:** 7,043 customers, 21 features
- **Target Variable:** `Churn` (Yes / No)
- **Key Features:** `tenure`, `monthly_charges`, `contract`, `internet_service`, and service subscriptions.

## 🛠️ Tools Used
- **Language:** Python 3.11, SQL
- **Database:** PostgreSQL 18
- **Machine Learning:** scikit-learn, XGBoost
- **Explainability:** SHAP (Shapley Additive exPlanations)
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Frontend Dashboard:** Streamlit, Plotly
- **Infrastructure:** Docker, Docker Compose

## 🔬 Analysis Process
The project was built in a systematic, direct pipeline:
1. **Data Ingestion & EDA:** Loaded raw data into PostgreSQL, cleaned missing values, and visualised churn distributions.
2. **Feature Engineering:** Created 5 new business-logic features (e.g., `charge_per_tenure`, `tenure_contract_risk`).
3. **Churn Classification:** Trained XGBoost and Random Forest models to predict churn probability, handling class imbalance.
4. **LTV Regression:** Engineered a forward-looking LTV target and trained an XGBoost Regressor to predict customer value.
5. **Model Serving:** Wrapped the trained models into a RESTful FastAPI service.
6. **Dashboarding:** Built a Streamlit UI to allow non-technical stakeholders to perform single, batch, and portfolio-level predictions.
7. **Dockerization:** Containerised the API, Frontend, and Database for 1-click deployment.

## 📈 Key Insights (Business)
- **Overall Churn Rate:** 26.5% of the portfolio.
- **Contract Risk:** Month-to-month contracts have a staggering ~42% churn rate compared to just ~3% for two-year contracts.
- **Tenure Risk:** Customers who churn typically leave within their first 18 months. Retained customers average 38+ months.
- **Revenue at Risk:** The model identified roughly $139,130/month in high-risk recurring revenue that requires immediate intervention.
- **Top Predictor:** `tenure_contract_risk` and contract type are the strongest signals for whether a customer will leave.

## 📁 Folder/Repo Structure
```text
churn-ltv-engine/
│
├── app/                          # FastAPI Backend Application
│   ├── core/                     # ML Model loaders and feature builders
│   ├── routers/                  # API endpoints (single, batch, lookup)
│   └── schemas/                  # Pydantic data validation models
│
├── saved_models/                 # Serialised XGBoost models & scalers (.pkl)
├── eda_outputs/                  # Exploratory Data Analysis charts
├── ltv_outputs/                  # LTV regression evaluation plots
├── model_outputs/                # Churn classification metrics & ROC curves
├── shap_outputs/                 # SHAP explainability visuals
│
├── docker-compose.yml            # Multi-container orchestration
├── Dockerfile                    # FastAPI Backend Docker image
├── Dockerfile.frontend           # Streamlit Frontend Docker image
│
├── load_data.py                  # Step 1: DB setup & ingestion
├── eda_analysis.py               # Step 2: Data visualisation
├── data_preprocessing.py         # Step 3: Cleaning & encoding
├── feature_engineering.py        # Step 4: Feature creation
├── model_training.py             # Step 5: Churn classification
├── shap_explainability.py        # Step 6: SHAP analysis
├── ltv_regression.py             # Step 7: LTV regression
├── main.py                       # Step 8: FastAPI entry point
└── streamlit_app.py              # Step 9: Streamlit Frontend Dashboard
```

---
**Deployment:** 
Run `docker compose up -d --build` to launch the API (`http://localhost:8000`) and the Dashboard (`http://localhost:8502`).