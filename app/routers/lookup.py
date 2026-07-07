"""
app/routers/lookup.py
DB lookup endpoints — retrieve stored predictions and model metrics.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException
from app.core.database import engine

router = APIRouter()


# ── GET /customers/{customer_id} ──────────────────────────────────────────
@router.get("/customers/{customer_id}")
def get_customer_prediction(customer_id: str):
    """
    Look up a stored prediction for a known customer ID from
    the model_predictions and customers_ltv tables.
    """
    try:
        # Check customers_ltv table (Week 3 output)
        ltv_df = pd.read_sql(
            f"SELECT * FROM customers_ltv WHERE customer_id = '{customer_id}'",
            engine,
        )
        if ltv_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{customer_id}' not found in customers_ltv table.",
            )

        row = ltv_df.iloc[0].to_dict()
        return {
            "customer_id"    : customer_id,
            "contract"       : row.get("contract"),
            "tenure"         : row.get("tenure"),
            "monthly_charges": row.get("monthly_charges"),
            "churn_status"   : row.get("churn"),
            "ltv_historical" : row.get("ltv_historical"),
            "ltv_projected"  : row.get("ltv_projected"),
            "ltv_predicted"  : row.get("ltv_predicted"),
            "ltv_segment"    : row.get("ltv_segment"),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /metrics/models ───────────────────────────────────────────────────
@router.get("/metrics/models")
def get_model_metrics():
    """
    Return stored model evaluation metrics from the database.
    Includes both churn classification (model_metrics) and
    LTV regression (ltv_model_metrics) results.
    """
    response = {}

    try:
        churn_df = pd.read_sql("SELECT * FROM model_metrics", engine)
        response["churn_model_metrics"] = churn_df.to_dict(orient="records")
    except Exception:
        response["churn_model_metrics"] = "model_metrics table not found"

    try:
        ltv_df = pd.read_sql("SELECT * FROM ltv_model_metrics", engine)
        response["ltv_model_metrics"] = ltv_df.to_dict(orient="records")
    except Exception:
        response["ltv_model_metrics"] = "ltv_model_metrics table not found"

    return response


# ── GET /metrics/summary ──────────────────────────────────────────────────
@router.get("/metrics/summary")
def get_business_summary():
    """
    High-level business summary from the customers_ltv table.
    Useful for dashboard widgets and executive reporting.
    """
    try:
        df = pd.read_sql("SELECT * FROM customers_ltv", engine)

        total          = len(df)
        churned        = (df["churn"] == "Yes").sum()
        churn_rate     = round(churned / total * 100, 2)
        avg_ltv        = round(df["ltv_predicted"].mean(), 2)
        total_rev_risk = round(
            df[df["churn"] == "Yes"]["ltv_predicted"].sum(), 2)

        seg_counts = df["ltv_segment"].value_counts().to_dict()

        return {
            "total_customers"      : int(total),
            "churned_customers"    : int(churned),
            "churn_rate_pct"       : churn_rate,
            "avg_predicted_ltv"    : avg_ltv,
            "total_revenue_at_risk": total_rev_risk,
            "ltv_segments"         : seg_counts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
