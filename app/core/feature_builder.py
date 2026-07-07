import pandas as pd
import numpy as np
from typing import Dict, Any
from app.core.model_loader import get_ltv_feature_names

# ── Feature lists (must match training scripts exactly) ───────────────────
ENGINEERED = [
    "charge_per_tenure",
    "service_count",
    "charge_to_value_ratio",
    "support_dependency_score",
    "tenure_contract_risk",
]

NUMERIC = ["tenure", "monthly_charges", "senior_citizen"]

# For churn model: total_charges IS included
NUMERIC_CHURN = ["tenure", "monthly_charges", "total_charges", "senior_citizen"]

SERVICE_COLS = [
    "phone_service", "multiple_lines", "online_security",
    "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies",
]
SERVICE_ENC = [f"{c}_enc" for c in SERVICE_COLS]

EXTRA_CATS = [
    "gender_enc", "partner_enc", "dependents_enc",
    "paperless_enc", "contract_enc",
]

INTERNET_DUMMIES  = ["internet_DSL", "internet_Fiber optic", "internet_No"]
PAYMENT_DUMMIES   = [
    "payment_Bank transfer (automatic)",
    "payment_Credit card (automatic)",
    "payment_Electronic check",
    "payment_Mailed check",
]
ONEHOT = INTERNET_DUMMIES + PAYMENT_DUMMIES


def _build_base_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw API input dict into a 1-row DataFrame with all
    engineered features and encodings applied.
    """
    row = {}

    # ── Raw fields ──────────────────────────────────────────────────────
    tenure          = float(data.get("tenure", 0))
    monthly_charges = float(data.get("monthly_charges", 0))
    total_charges   = float(data.get("total_charges", 0))
    senior_citizen  = int(data.get("senior_citizen", 0))
    contract        = data.get("contract", "Month-to-month")
    gender          = data.get("gender", "Male")
    partner         = data.get("partner", "No")
    dependents      = data.get("dependents", "No")
    paperless       = data.get("paperless_billing", "No")
    internet_svc    = data.get("internet_service", "No")
    payment         = data.get("payment_method", "Mailed check")

    # Normalise service values
    def svc_val(key):
        v = data.get(key, "No")
        if v in ("No internet service", "No phone service"):
            v = "No"
        return v

    services = {col: svc_val(col) for col in SERVICE_COLS}

    # ── Numeric ─────────────────────────────────────────────────────────
    row["tenure"]           = tenure
    row["monthly_charges"]  = monthly_charges
    row["total_charges"]    = total_charges
    row["senior_citizen"]   = senior_citizen

    # ── Engineered features ──────────────────────────────────────────────
    service_count = sum(1 for v in services.values() if v == "Yes")
    support_cols  = ["online_security", "device_protection",
                     "tech_support", "online_backup"]
    support_score = sum(1 for c in support_cols if services.get(c) == "Yes")

    contract_norm_map = {"Month-to-month": 0.0, "One year": 0.5, "Two year": 1.0}
    tenure_max        = 72.0  # dataset max tenure
    tenure_norm       = min(tenure / tenure_max, 1.0)
    contract_norm     = contract_norm_map.get(contract, 0.0)

    row["charge_per_tenure"]        = total_charges / (tenure + 1)
    row["service_count"]            = service_count
    row["charge_to_value_ratio"]    = monthly_charges / (service_count + 1)
    row["support_dependency_score"] = support_score
    row["tenure_contract_risk"]     = (1 - tenure_norm) * (1 - contract_norm)

    # ── Binary encodings ─────────────────────────────────────────────────
    row["gender_enc"]     = 1 if gender == "Male" else 0
    row["partner_enc"]    = 1 if partner == "Yes" else 0
    row["dependents_enc"] = 1 if dependents == "Yes" else 0
    row["paperless_enc"]  = 1 if paperless == "Yes" else 0
    row["contract_enc"]   = {"Month-to-month": 0,
                              "One year": 1, "Two year": 2}.get(contract, 0)

    for col in SERVICE_COLS:
        row[f"{col}_enc"] = 1 if services[col] == "Yes" else 0

    # ── One-hot: internet_service ─────────────────────────────────────────
    for label in ["DSL", "Fiber optic", "No"]:
        row[f"internet_{label}"] = 1 if internet_svc == label else 0

    # ── One-hot: payment_method ───────────────────────────────────────────
    for label in [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]:
        row[f"payment_{label}"] = 1 if payment == label else 0

    return pd.DataFrame([row])


def build_churn_features(data: Dict[str, Any]) -> pd.DataFrame:
    """Feature vector for the churn XGBoost model."""
    df = _build_base_df(data)
    churn_cols = (ENGINEERED + NUMERIC_CHURN +
                  SERVICE_ENC + EXTRA_CATS + ONEHOT)
    # Ensure all columns exist
    for col in churn_cols:
        if col not in df.columns:
            df[col] = 0
    return df[churn_cols]


def build_ltv_features(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Feature vector for the LTV XGBoost model (total_charges excluded).
    """
    df = _build_base_df(data)

    # Base set of columns you expect to exist
    ltv_cols = (ENGINEERED + NUMERIC +
                SERVICE_ENC + EXTRA_CATS + ONEHOT)

    # Ensure all columns exist with defaults
    for col in ltv_cols:
        if col not in df.columns:
            df[col] = 0

    # Get the actual feature names from the model, if available
    model_feature_names = get_ltv_feature_names()

    if model_feature_names is not None:
        # Reorder and filter to exactly match the model
        # (ignore any extra columns)
        missing = [c for c in model_feature_names if c not in df.columns]
        if missing:
            # Defensive: create missing ones as 0 so predict() doesn't break
            for c in missing:
                df[c] = 0
        return df[model_feature_names]
    else:
        # Fallback: use local ltv_cols order
        return df[ltv_cols]