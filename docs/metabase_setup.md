# Metabase Setup — churn-ltv-engine

## Tool Details
- **Tool:** Metabase Community Edition (OSS)
- **Version:** Latest (download from https://www.metabase.com/start/oss/)
- **Access URL:** http://localhost:3000
- **Runs via:** `java -jar metabase.jar`

---

## Database Connection

| Field         | Value              |
|---------------|--------------------|
| Database type | PostgreSQL         |
| Display name  | churn-ltv-engine   |
| Host          | localhost          |
| Port          | 5432               |
| Database name | telco_churn_db     |
| Username      | churn_admin        |
| Password      | (stored in .env)   |
| SSL           | OFF                |

---

## Tables Available (9 total)

| Table | Rows | Primary Use |
|---|---|---|
| `customers` | 7,043 | Raw data — churn by contract, tenure, services |
| `customers_processed` | 7,043 | Encoded data — all categoricals as numbers |
| `customers_features` | 7,043 | Original + 5 engineered features |
| `customers_ltv` | 7,043 | **PRIMARY** — LTV values, segments, churn status |
| `model_metrics` | 3 | Churn model performance (LR, RF, XGBoost) |
| `ltv_model_metrics` | 3 | LTV regression performance |
| `model_predictions` | 1,409 | Test set churn probabilities |
| `shap_feature_importance` | N | Global SHAP feature rankings |
| `shap_customer_values` | 1,000 | Per-customer SHAP scores |

---

## Saved Questions (Collection: Churn-LTV Engine)

| # | Question Name | Table | Chart Type |
|---|---|---|---|
| 1 | Overall Churn Rate | customers | Pie chart |
| 2 | Churn by Contract Type | customers | Bar chart |
| 3 | LTV Segment Distribution | customers_ltv | Bar chart |
| 4 | Revenue at Risk | customers_ltv | Number card |
| 5 | High-Risk Premium Customers | customers_ltv | Table |
| 6 | Average LTV by Contract Type | customers_ltv | Bar chart |
| 7 | Churn Rate by Tenure Band | customers | Bar chart |
| 8 | SHAP Feature Importance | shap_feature_importance | Bar chart |

---

## Dashboards

### Churn & LTV Overview
- **Purpose:** Executive summary of churn risk and LTV segmentation
- **Cards:** All 8 questions above

---

## SQL Queries Reference

### 1. Overall Churn Rate
```sql
SELECT
  churn,
  COUNT(*) AS customers,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM customers
GROUP BY churn
ORDER BY churn;
```

### 2. Churn by Contract Type
```sql
SELECT
  contract,
  COUNT(*) AS total,
  SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END) AS churned,
  ROUND(SUM(CASE WHEN churn='Yes' THEN 1 ELSE 0 END)*100.0/COUNT(*),1) AS churn_rate
FROM customers
GROUP BY contract
ORDER BY churn_rate DESC;
```

### 3. LTV Segment Distribution
```sql
SELECT
  ltv_segment,
  COUNT(*) AS customers,
  ROUND(AVG(ltv_predicted),2) AS avg_ltv
FROM customers_ltv
GROUP BY ltv_segment
ORDER BY avg_ltv DESC;
```

### 4. Revenue at Risk
```sql
SELECT
  ROUND(SUM(ltv_predicted), 2) AS total_portfolio_ltv,
  ROUND(SUM(CASE WHEN churn = 'Yes' THEN ltv_predicted ELSE 0 END), 2) AS revenue_at_risk,
  ROUND(SUM(CASE WHEN churn = 'Yes' THEN ltv_predicted ELSE 0 END)
        * 100.0 / SUM(ltv_predicted), 2) AS pct_at_risk
FROM customers_ltv;
```

### 5. High-Risk Premium Customers
```sql
SELECT
  customer_id,
  contract,
  tenure,
  ROUND(monthly_charges::numeric, 2) AS monthly_charges,
  ROUND(ltv_predicted::numeric, 2) AS ltv_predicted,
  ltv_segment
FROM customers_ltv
WHERE churn = 'Yes'
  AND ltv_segment IN ('Premium', 'High')
ORDER BY ltv_predicted DESC
LIMIT 20;
```

### 6. Average LTV by Contract Type
```sql
SELECT
  contract,
  ROUND(AVG(ltv_historical),2) AS hist_ltv,
  ROUND(AVG(ltv_projected),2) AS proj_ltv,
  ROUND(AVG(ltv_predicted),2) AS pred_ltv
FROM customers_ltv
GROUP BY contract
ORDER BY pred_ltv DESC;
```

### 7. Churn Rate by Tenure Band
```sql
SELECT
  CASE
    WHEN tenure <= 12 THEN '0-12 mo'
    WHEN tenure <= 24 THEN '13-24 mo'
    WHEN tenure <= 48 THEN '25-48 mo'
    ELSE '49-72 mo'
  END AS tenure_band,
  COUNT(*) total,
  ROUND(AVG(CASE WHEN churn='Yes' THEN 1 ELSE 0 END)*100,1) AS churn_pct
FROM customers
GROUP BY tenure_band
ORDER BY tenure_band;
```

### 8. SHAP Feature Importance
```sql
SELECT
  feature,
  ROUND(mean_abs_shap::numeric, 4) AS importance,
  is_engineered,
  global_rank
FROM shap_feature_importance
ORDER BY global_rank ASC
LIMIT 10;
```