import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "saved_models"

_models = {}
_ltv_feature_names = None  # NEW: store LTV feature names


def load_all_models():
    global _models, _ltv_feature_names

    required = {
        "churn_model": "xgboost.pkl",
        "ltv_model": "ltv_xgboost.pkl",
        "churn_scaler": "scaler.pkl",
        "ltv_scaler": "ltv_scaler.pkl",
    }

    for key, filename in required.items():
        path = MODELS_DIR / filename
        if path.exists():
            _models[key] = joblib.load(path)
            print(f" Loaded: {filename}")
        else:
            print(f" WARNING: {filename} not found at {path}")
            _models[key] = None

    # NEW: after loading ltv model, grab its feature names
    ltv_model = _models.get("ltv_model")
    if ltv_model is not None:
        booster = ltv_model.get_booster()
        _ltv_feature_names = booster.feature_names
        print(" LTV feature names:", _ltv_feature_names)


def get_churn_model():
    return _models.get("churn_model")


def get_ltv_model():
    return _models.get("ltv_model")


def get_churn_scaler():
    return _models.get("churn_scaler")


def get_ltv_scaler():
    return _models.get("ltv_scaler")

# NEW: expose feature names to the feature builder
def get_ltv_feature_names():
    return _ltv_feature_names