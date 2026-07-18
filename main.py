import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.routers import single, batch, lookup
from app.core.model_loader import load_all_models
from app.core.security import _check_api_key_config, limiter, verify_api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("churn_ltv.main")


# ── Lifespan: startup / shutdown ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate API key is configured before accepting any traffic
    _check_api_key_config()
    logger.info("API key validated ✓")

    logger.info("Loading ML models...")
    load_all_models()
    logger.info("All models loaded. API ready ✓")

    yield

    logger.info("Shutting down Churn-LTV Engine.")


# ── App instance ───────────────────────────────────────────────────────────
# verify_api_key is applied globally — every route requires a valid X-API-Key
app = FastAPI(
    title       = "Churn-LTV Engine API",
    description = (
        "FastAPI service for Customer Churn Prediction and "
        "Lifetime Value forecasting. "
        "Built on XGBoost models trained on the IBM Telco dataset.\n\n"
        "**Authentication**: All endpoints require an `X-API-Key` header.\n"
        "**Rate Limit**: 60 requests / minute per IP address."
    ),
    version      = "1.1.0",
    lifespan     = lifespan,
    dependencies = [Depends(verify_api_key)],   # ← Global API key guard
)

# Attach the rate-limiter state to the app
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


# ── Rate-limit exceeded handler ────────────────────────────────────────────
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}. "
                      "Please slow down and retry after a moment."
        },
    )


# ── CORS ───────────────────────────────────────────────────────────────────
# Restrict to known trusted origins.
# Add your Metabase / Streamlit / frontend domain here when deployed.
ALLOWED_ORIGINS = [
    "http://localhost:3000",    # Metabase (local)
    "http://localhost:8501",    # Streamlit (local)
    "http://localhost:8000",    # Swagger UI self-reference
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],        # Only methods we actually use
    allow_headers     = ["X-API-Key", "Content-Type", "Accept"],
)


# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(single.router, prefix="/predict", tags=["Single Inference"])
app.include_router(batch.router,  prefix="/predict", tags=["Batch Inference"])
app.include_router(lookup.router, prefix="",         tags=["Lookup & Metrics"])


# ── Root health check (public) ─────────────────────────────────────────────
# The verify_api_key dependency exempts "/" via PUBLIC_PATHS — no key needed.
@app.get("/", tags=["Health"])
def health_check():
    """
    Public health check endpoint. No API key required.
    Used by Docker HEALTHCHECK and load balancers.
    """
    return {
        "status"  : "ok",
        "service" : "Churn-LTV Engine API",
        "version" : "1.1.0",
        "auth"    : "X-API-Key header required on all /predict and /metrics routes",
        "docs"    : "/docs",
    }