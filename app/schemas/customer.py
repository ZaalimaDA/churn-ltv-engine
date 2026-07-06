from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year       = "One year"
    two_year       = "Two year"

class InternetService(str, Enum):
    dsl         = "DSL"
    fiber_optic = "Fiber optic"
    no          = "No"

class PaymentMethod(str, Enum):
    bank_transfer    = "Bank transfer (automatic)"
    credit_card      = "Credit card (automatic)"
    electronic_check = "Electronic check"
    mailed_check     = "Mailed check"

class YesNo(str, Enum):
    yes = "Yes"
    no  = "No"


# ── Single customer request ────────────────────────────────────────────────
class CustomerInput(BaseModel):
    """Raw customer data — exactly what comes from a CRM or data warehouse."""

    # Demographics
    gender          : str   = Field("Male",  description="Male or Female")
    senior_citizen  : int   = Field(0,       description="1 if senior citizen, 0 otherwise", ge=0, le=1)
    partner         : YesNo = Field("No",    description="Has a partner?")
    dependents      : YesNo = Field("No",    description="Has dependents?")

    # Account
    tenure          : float = Field(...,     description="Months with the company", ge=0, le=72)
    contract        : ContractType = Field("Month-to-month")
    paperless_billing: YesNo = Field("No")
    payment_method  : PaymentMethod = Field("Mailed check")
    monthly_charges : float = Field(...,     description="Monthly bill amount ($)", ge=0)
    total_charges   : float = Field(0.0,     description="Total billed to date ($)", ge=0)

    # Services
    phone_service   : YesNo = Field("No")
    multiple_lines  : str   = Field("No",    description="No / Yes / No phone service")
    internet_service: InternetService = Field("No")
    online_security : str   = Field("No",    description="No / Yes / No internet service")
    online_backup   : str   = Field("No")
    device_protection: str  = Field("No")
    tech_support    : str   = Field("No")
    streaming_tv    : str   = Field("No")
    streaming_movies: str   = Field("No")

    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "senior_citizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 65.50,
                "total_charges": 786.0,
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No",
            }
        }


# ── Churn prediction response ──────────────────────────────────────────────
class ChurnPredictionResponse(BaseModel):
    churn_prediction   : int   = Field(..., description="1 = will churn, 0 = will stay")
    churn_probability  : float = Field(..., description="Probability of churning (0-1)")
    churn_risk_label   : str   = Field(..., description="Low / Medium / High / Critical")
    model_used         : str   = "XGBoost"


# ── LTV prediction response ────────────────────────────────────────────────
class LTVPredictionResponse(BaseModel):
    ltv_predicted   : float = Field(..., description="Predicted lifetime value ($)")
    ltv_segment     : str   = Field(..., description="Low / Medium / High / Premium")
    model_used      : str   = "XGBoost Regressor"


# ── Combined response ──────────────────────────────────────────────────────
class CombinedPredictionResponse(BaseModel):
    churn_prediction  : int
    churn_probability : float
    churn_risk_label  : str
    ltv_predicted     : float
    ltv_segment       : str
    revenue_at_risk   : float = Field(...,
        description="Expected revenue lost if customer churns = ltv_predicted × churn_probability")
    priority_score    : str   = Field(...,
        description="Retention priority: Critical / High / Medium / Low")
    model_used        : str   = "XGBoost (Churn) + XGBoost Regressor (LTV)"


# ── Batch response ────────────────────────────────────────────────────────
class BatchResultRow(BaseModel):
    row_index          : int
    churn_prediction   : int
    churn_probability  : float
    churn_risk_label   : str
    ltv_predicted      : float
    ltv_segment        : str
    revenue_at_risk    : float

class BatchPredictionResponse(BaseModel):
    total_customers    : int
    churners_predicted : int
    churn_rate         : float
    total_revenue_at_risk: float
    results            : List[BatchResultRow]