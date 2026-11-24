"""Central configuration for the backend.

Loads environment variables from the project `.env` file and exposes the paths used
by every other backend module.

API keys are **never** hard-coded here - they are read from the environment and are
never returned by the API or logged.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "static"

# Load the .env file that sits in the project root (if it exists).
load_dotenv(PROJECT_ROOT / ".env")

MODEL_FILES = {
    "denial": MODELS_DIR / "denial_model.pkl",
    "recovery": MODELS_DIR / "reimbursement_model.pkl",
    "denial_reason": MODELS_DIR / "denial_reason_model.pkl",
}
METADATA_FILE = MODELS_DIR / "model_metadata.json"
FEATURE_COLUMNS_FILE = DATA_DIR / "feature_columns.json"
PAYER_AGG_FILE = DATA_DIR / "payer_aggregates.csv"
DENIAL_CODE_AGG_FILE = DATA_DIR / "denial_code_aggregates.csv"
LOOKUP_PAYER_FILE = DATA_DIR / "lookup_payer.csv"
LOOKUP_PAYER_RULES_FILE = DATA_DIR / "lookup_payer_rules.csv"
DASHBOARD_FILE = DATA_DIR / "dashboard_stats.json"

# --------------------------------------------------------------------------------------
# LLM configuration
# --------------------------------------------------------------------------------------
MAX_LLM_KEYS = 10
DEFAULT_LLM_MODEL = "gpt-4o-mini"
_EMPTY_PLACEHOLDERS = {"", "your_key_here", "none", "null", "keep it empty for now, ill provide it later"}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    if value.lower() in _EMPTY_PLACEHOLDERS:
        return None
    return value


def get_api_keys() -> list[str]:
    """Return the ordered pool of configured OpenAI API keys (may be empty)."""
    keys: list[str] = []
    for i in range(1, MAX_LLM_KEYS + 1):
        key = _clean(os.getenv(f"OPENAI_API_KEY_{i}"))
        if key and key not in keys:
            keys.append(key)
    return keys


LLM_MODEL = _clean(os.getenv("LLM_MODEL")) or DEFAULT_LLM_MODEL
OPENAI_BASE_URL = _clean(os.getenv("OPENAI_BASE_URL"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Denial-reason predictions are only shown for claims above this denial probability.
DENIAL_REASON_MIN_PROBABILITY = float(os.getenv("DENIAL_REASON_MIN_PROBABILITY", "0.35"))
