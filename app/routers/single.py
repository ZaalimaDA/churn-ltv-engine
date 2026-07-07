import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas.customer import (
    CustomerInput,
    ChurnPredictionResponse,
    LTVPredictionResponse,
    CombinedPredictionResponse,
)
from app.core.model_loader import get_churn_model, get_ltv_model
from app.core.feature_builder import build_churn_features, build_ltv_features

router = APIRouter()


def _churn_risk_label(prob: float) -> str:
    if prob < 0.25:  return "Low"
    if prob < 0.50:  return "Medium"
    if prob < 0.75:  return "High"
    return "Critical"


def _ltv_segment(ltv: float) -> str:
    """
    Approximate segment thresholds derived from training data quartiles.
    These are set from the customers_ltv table quartile values.
    """
    if ltv < 1000:   return "Low"
    if ltv < 2000:   return "Medium"
    if ltv < 3500:   return "High"
    return "Premium"


def _priority_score(churn_prob: float, ltv: float) -> str:
    """
    Retention priority = combination of churn risk and LTV.
    High LTV + high churn = most urgent.
    """
    high_churn = churn_prob >= 0.50
    high_ltv   = ltv >= 2000
    if high_churn and high_ltv:    return "Critical"
    if high_churn and not high_ltv: return "High"
    if not high_churn and high_ltv: return "Medium"
    return "Low"


# ── POST /predict/churn ───────────────────────────────────────────────────
@router.post("/churn", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a single customer.

    - **churn_prediction**: 1 = predicted to churn, 0 = predicted to stay
    - **churn_probability**: model confidence (0.0 → 1.0)
    - **churn_risk_label**: Low / Medium / High / Critical
    """
    model = get_churn_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Churn model not loaded. Check saved_models/xgboost.pkl exists.",
        )

    X = build_churn_features(customer.dict())
    prob       = float(model.predict_proba(X)[0][1])
    prediction = int(prob >= 0.5)

    return ChurnPredictionResponse(
        churn_prediction  = prediction,
        churn_probability = round(prob, 4),
        churn_risk_label  = _churn_risk_label(prob),
    )


# ── POST /predict/ltv ─────────────────────────────────────────────────────
@router.post("/ltv", response_model=LTVPredictionResponse)
def predict_ltv(customer: CustomerInput):
    """
    Predict the expected Lifetime Value (LTV) for a single customer.

    - **ltv_predicted**: forecasted total revenue ($)
    - **ltv_segment**: Low / Medium / High / Premium
    """
    model = get_ltv_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="LTV model not loaded. Check saved_models/ltv_xgboost.pkl exists.",
        )

    X   = build_ltv_features(customer.dict())
    ltv = float(max(model.predict(X)[0], 0))  # no negative LTV

    return LTVPredictionResponse(
        ltv_predicted = round(ltv, 2),
        ltv_segment   = _ltv_segment(ltv),
    )


# ── POST /predict/churn-and-ltv ───────────────────────────────────────────
@router.post("/churn-and-ltv", response_model=CombinedPredictionResponse)
def predict_combined(customer: CustomerInput):
    """
    Combined endpoint: churn probability + LTV prediction + Revenue at Risk.

    - **revenue_at_risk**: ltv_predicted × churn_probability
      (expected revenue loss if no retention action is taken)
    - **priority_score**: Critical / High / Medium / Low
      (guides retention campaign prioritisation)
    """
    churn_model = get_churn_model()
    ltv_model   = get_ltv_model()

    if churn_model is None or ltv_model is None:
        raise HTTPException(
            status_code=503,
            detail="One or more models not loaded.",
        )

    data          = customer.dict()
    X_churn       = build_churn_features(data)
    X_ltv         = build_ltv_features(data)

    prob          = float(churn_model.predict_proba(X_churn)[0][1])
    prediction    = int(prob >= 0.5)
    ltv           = float(max(ltv_model.predict(X_ltv)[0], 0))
    revenue_risk  = round(ltv * prob, 2)

    return CombinedPredictionResponse(
        churn_prediction  = prediction,
        churn_probability = round(prob, 4),
        churn_risk_label  = _churn_risk_label(prob),
        ltv_predicted     = round(ltv, 2),
        ltv_segment       = _ltv_segment(ltv),
        revenue_at_risk   = revenue_risk,
        priority_score    = _priority_score(prob, ltv),
    )