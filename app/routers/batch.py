"""
app/routers/batch.py
Batch prediction endpoints — accepts CSV file upload.
"""

import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File

from app.schemas.customer import BatchPredictionResponse, BatchResultRow
from app.core.model_loader import get_churn_model, get_ltv_model
from app.core.feature_builder import build_churn_features, build_ltv_features
from app.routers.single import _churn_risk_label, _ltv_segment

router = APIRouter()

# Required columns in the uploaded CSV
REQUIRED_COLS = [
    "tenure", "monthly_charges", "total_charges",
    "contract", "internet_service", "payment_method",
]


def _process_batch(df: pd.DataFrame):
    """
    Run churn + LTV predictions on a DataFrame of customers.
    Returns a list of BatchResultRow dicts.
    """
    churn_model = get_churn_model()
    ltv_model   = get_ltv_model()

    if churn_model is None or ltv_model is None:
        raise HTTPException(status_code=503,
                            detail="Models not loaded.")

    results = []
    for idx, row in df.iterrows():
        data = row.to_dict()

        # Fill defaults for optional columns
        defaults = {
            "gender": "Male", "senior_citizen": 0,
            "partner": "No",  "dependents": "No",
            "paperless_billing": "No",
            "phone_service": "No", "multiple_lines": "No",
            "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "No", "streaming_movies": "No",
        }
        for k, v in defaults.items():
            data.setdefault(k, v)

        # Fill numeric defaults
        data["tenure"]          = float(data.get("tenure", 0) or 0)
        data["monthly_charges"] = float(data.get("monthly_charges", 0) or 0)
        data["total_charges"]   = float(data.get("total_charges", 0) or 0)
        data["senior_citizen"]  = int(data.get("senior_citizen", 0) or 0)

        X_churn = build_churn_features(data)
        X_ltv   = build_ltv_features(data)

        prob = float(churn_model.predict_proba(X_churn)[0][1])
        ltv  = float(max(ltv_model.predict(X_ltv)[0], 0))

        results.append(BatchResultRow(
            row_index         = int(idx),
            churn_prediction  = int(prob >= 0.5),
            churn_probability = round(prob, 4),
            churn_risk_label  = _churn_risk_label(prob),
            ltv_predicted     = round(ltv, 2),
            ltv_segment       = _ltv_segment(ltv),
            revenue_at_risk   = round(ltv * prob, 2),
        ))

    return results


# ── POST /predict/batch/churn ─────────────────────────────────────────────
@router.post("/batch/churn", response_model=BatchPredictionResponse)
async def batch_churn(file: UploadFile = File(...)):
    """
    Batch churn + LTV predictions from a CSV file upload.

    **Required CSV columns:** tenure, monthly_charges, total_charges,
    contract, internet_service, payment_method

    **Optional columns:** all other customer attributes (defaults applied if missing)

    Returns summary statistics and per-row predictions.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only .csv files are accepted.",
        )

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Could not parse CSV: {str(e)}")

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required CSV columns: {missing}",
        )

    if len(df) > 5000:
        raise HTTPException(
            status_code=413,
            detail="Batch size limit is 5,000 rows per request.",
        )

    results = _process_batch(df)

    churners          = sum(r.churn_prediction for r in results)
    total_risk        = sum(r.revenue_at_risk  for r in results)
    churn_rate        = round(churners / len(results), 4) if results else 0.0

    return BatchPredictionResponse(
        total_customers      = len(results),
        churners_predicted   = churners,
        churn_rate           = churn_rate,
        total_revenue_at_risk= round(total_risk, 2),
        results              = results,
    )


# ── POST /predict/batch/ltv ───────────────────────────────────────────────
@router.post("/batch/ltv", response_model=BatchPredictionResponse)
async def batch_ltv(file: UploadFile = File(...)):
    """
    Alias for batch/churn — returns the same combined payload.
    Kept as a separate endpoint for semantic clarity in downstream systems.
    """
    return await batch_churn(file)