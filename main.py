from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import single, batch, lookup
from app.core.model_loader import load_all_models

# ── Lifespan: load models once at startup ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    load_all_models()
    print("All models loaded. API ready.")
    yield
    print("Shutting down.")

# ── App instance ───────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Churn-LTV Engine API",
    description = (
        "FastAPI service for Customer Churn Prediction and "
        "Lifetime Value forecasting. "
        "Built on XGBoost models trained on the IBM Telco dataset."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS (allows browser / Superset to call the API) ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(single.router,  prefix="/predict",   tags=["Single Inference"])
app.include_router(batch.router,   prefix="/predict",   tags=["Batch Inference"])
app.include_router(lookup.router,  prefix="",           tags=["Lookup & Metrics"])

# ── Root health check ──────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status"  : "ok",
        "service" : "Churn-LTV Engine API",
        "version" : "1.0.0",
        "docs"    : "/docs",
    }