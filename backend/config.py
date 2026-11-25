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
# LLM configuration - NVIDIA NIM model fallback chain
# --------------------------------------------------------------------------------------
# The LLM layer no longer uses the OpenAI sk- key pool. A single NVIDIA NIM API key
# is used to call an ordered list of models; if a model fails the next model in the
# chain is tried, and only when every model fails does the API fall back to the
# deterministic local rules engine.
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Ordered fallback chain - the first model that can serve the request wins.
# Ordered fastest-and-most-reliable first (measured against a real claim payload) so
# the common case stays quick; slower / intermittently-unavailable models sit later.
# (Requested IDs deepseek-ai/deepseek-v4.1-flash and qwen/qwen2.5-coder-32b-instruct
# are not offered by the NVIDIA NIM endpoint and are therefore omitted.)
DEFAULT_NVIDIA_MODELS = [
    "nvidia/nemotron-3.5-bolt-e4b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "minimaxai/minimax-m3.1-flash",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "moonshotai/kimi-k3.1-pro",
    "mistralai/mistral-large-4-1t-a52b",
    "stepfun-ai/step-5-preview",
    "z-ai/glm-5.3-flash",
]

_EMPTY_PLACEHOLDERS = {"", "your_key_here", "none", "null", "keep it empty for now, ill provide it later"}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    if value.lower() in _EMPTY_PLACEHOLDERS:
        return None
    return value


NVIDIA_API_KEY = _clean(os.getenv("NVIDIA_API_KEY"))
NVIDIA_BASE_URL = _clean(os.getenv("NVIDIA_BASE_URL")) or DEFAULT_NVIDIA_BASE_URL


def get_nvidia_models() -> list[str]:
    """Return the ordered NVIDIA model fallback chain.

    ``NVIDIA_MODELS`` (comma-separated) overrides the built-in default chain.
    Duplicates are removed while preserving order.
    """
    override = _clean(os.getenv("NVIDIA_MODELS"))
    models = [m.strip() for m in override.split(",") if m.strip()] if override \
        else list(DEFAULT_NVIDIA_MODELS)
    ordered: list[str] = []
    for model in models:
        if model not in ordered:
            ordered.append(model)
    return ordered


def get_llm_targets() -> list[dict]:
    """Ordered fallback chain of LLM call targets: ``{base_url, api_key, model}``.

    Empty when no NVIDIA API key is configured.
    """
    if not NVIDIA_API_KEY:
        return []
    return [
        {"base_url": NVIDIA_BASE_URL, "api_key": NVIDIA_API_KEY, "model": model}
        for model in get_nvidia_models()
    ]


# Primary model (head of the fallback chain) - used for status display only.
LLM_MODEL = (get_nvidia_models() or ["(none)"])[0]
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Denial-reason predictions are only shown for claims above this denial probability.
DENIAL_REASON_MIN_PROBABILITY = float(os.getenv("DENIAL_REASON_MIN_PROBABILITY", "0.35"))
