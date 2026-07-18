"""
app/core/security.py
────────────────────────────────────────────────────────────────────────────
API Security layer for the Churn-LTV Engine.

Implements:
  1. API Key Authentication  — via X-API-Key request header
  2. Rate Limiting           — via slowapi (Redis-backed or in-memory)

Usage (applied globally in main.py):
  app = FastAPI(dependencies=[Depends(verify_api_key)])
"""

import os
import secrets
import logging
from dotenv import load_dotenv

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader

from slowapi import Limiter
from slowapi.util import get_remote_address

load_dotenv()

logger = logging.getLogger("churn_ltv.security")

# ── API Key configuration ─────────────────────────────────────────────────
_API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_NAME, auto_error=False)

# Read the valid API key from environment; raise at startup if missing
_VALID_API_KEY: str | None = os.getenv("API_KEY")


def _check_api_key_config() -> None:
    """Called once at app startup to validate the key is configured."""
    if not _VALID_API_KEY:
        raise RuntimeError(
            "API_KEY environment variable is not set. "
            "Add 'API_KEY=<your-secret-key>' to your .env file and restart."
        )
    if len(_VALID_API_KEY) < 32:
        logger.warning(
            "API_KEY is shorter than 32 characters — consider using a longer key "
            "for production deployments."
        )


# Paths that are publicly accessible without an API key
PUBLIC_PATHS = {"/", "/openapi.json"}


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """
    FastAPI dependency — validates the X-API-Key header on every protected request.
    Paths listed in PUBLIC_PATHS (e.g. health check, OpenAPI schema) are exempt.

    Raises:
        401 Unauthorized  — if no key is provided on a protected path
        403 Forbidden     — if the key is present but incorrect
    """
    # Allow public paths through without a key
    if request.url.path in PUBLIC_PATHS:
        return None

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'X-API-Key: <your-key>' in the request headers.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison prevents timing attacks
    if not secrets.compare_digest(api_key, _VALID_API_KEY or ""):
        logger.warning("Rejected request with invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key


# ── Rate limiter ──────────────────────────────────────────────────────────
# Uses the client IP address as the rate-limit key.
# Default limit is defined here; individual routes can override via @limiter.limit().
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],   # 60 requests per minute per IP
    headers_enabled=True,           # Adds X-RateLimit-* headers to responses
)
