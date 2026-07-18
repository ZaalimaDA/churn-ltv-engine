# ── Stage 1: Base Python image ──────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="ZaalimaDA"
LABEL project="churn-ltv-engine"
LABEL description="Customer Churn Prediction & LTV Engine - FastAPI Service"

# Set working directory
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────
# libgomp1 is required by XGBoost for parallel processing
RUN apt-get update && apt-get install -y --fix-missing \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────
# Copy requirements first (Docker layer caching — only re-installs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────
COPY main.py .
COPY app/ ./app/
COPY saved_models/ ./saved_models/

# ── Environment variables (overridden by docker-compose) ──────────────────
ENV DB_USER=churn_admin
ENV DB_PASSWORD=yourpassword
ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_NAME=telco_churn_db

# ── Expose FastAPI port ───────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ── Run the FastAPI server ────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]