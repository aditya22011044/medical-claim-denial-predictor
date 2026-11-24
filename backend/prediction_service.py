"""Loads the trained models once and turns a raw claim into ML predictions.

This module is intentionally dependency-light: it only needs pandas, numpy, joblib
and the saved Scikit-learn pipelines. It never talks to the LLM, so ML inference keeps
working even when every API key is unavailable.

Only the **feature-selected top-5** attributes are used to build the model input
(read from ``feature_columns.json`` -> ``selected_features``).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config
from .schemas import AUTH_STATUSES, ClaimInput, PayerIntelligence, PredictionResult, RiskFactor


# --------------------------------------------------------------------------------------
# Artifact loading (once per process)
# --------------------------------------------------------------------------------------
class Artifacts:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        # full feature space (for reference / docs)
        self.features: list[str] = []
        self.categorical: list[str] = []
        self.numeric: list[str] = []
        # the selected top-5 feature space actually fed to the models
        self.selected_features: list[str] = []
        self.selected_categorical: list[str] = []
        self.selected_numeric: list[str] = []
        self.feature_selection: dict[str, Any] = {}
        self.payer_lookup: pd.DataFrame = pd.DataFrame()
        self.payer_rules: pd.DataFrame = pd.DataFrame()
        self.payer_agg: pd.DataFrame = pd.DataFrame()
        self.code_agg: pd.DataFrame = pd.DataFrame()
        self.dashboard: dict[str, Any] = {}
        self.category_to_code: dict[str, str] = {}
        self.quantiles: dict[str, float] = {}
        self.errors: list[str] = []


@lru_cache(maxsize=1)
def get_artifacts() -> Artifacts:
    art = Artifacts()

    # features / thresholds
    try:
        with open(config.FEATURE_COLUMNS_FILE, encoding="utf-8") as fh:
            cols = json.load(fh)
        art.features = cols["features"]
        art.categorical = cols["categorical"]
        art.numeric = cols["numeric"]
        art.selected_features = cols["selected_features"]
        art.selected_categorical = cols["selected_categorical"]
        art.selected_numeric = cols["selected_numeric"]
        art.feature_selection = cols.get("feature_selection", {})
        art.quantiles = cols.get("quantiles", {})
    except Exception as exc:  # pragma: no cover
        art.errors.append(f"feature columns: {exc}")

    # models
    for name, path in config.MODEL_FILES.items():
        try:
            art.models[name] = joblib.load(path)
        except Exception as exc:
            art.errors.append(f"model '{name}': {exc}")

    # metadata
    try:
        art.metadata = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        art.errors.append(f"metadata: {exc}")

    # lookups
    try:
        art.payer_lookup = pd.read_csv(config.LOOKUP_PAYER_FILE)
        art.payer_rules = pd.read_csv(config.LOOKUP_PAYER_RULES_FILE)
        art.payer_agg = pd.read_csv(config.PAYER_AGG_FILE)
        art.code_agg = pd.read_csv(config.DENIAL_CODE_AGG_FILE)
    except Exception as exc:
        art.errors.append(f"lookups: {exc}")

    try:
        art.dashboard = json.loads(config.DASHBOARD_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        art.errors.append(f"dashboard: {exc}")

    # representative X12 code per denial category (taken from the real aggregates)
    if not art.code_agg.empty:
        top = art.code_agg.sort_values("claims", ascending=False).drop_duplicates("denial_category")
        art.category_to_code = dict(zip(top["denial_category"], top["denial_reason_code"]))

    return art


def models_loaded() -> dict[str, bool]:
    art = get_artifacts()
    return {name: name in art.models for name in ("denial", "recovery", "denial_reason")}


# --------------------------------------------------------------------------------------
# Feature engineering - only the top-5 selected features
# --------------------------------------------------------------------------------------
def build_feature_row(claim: ClaimInput) -> pd.DataFrame:
    art = get_artifacts()
    row: dict[str, Any] = {
        "documentation_completeness": float(claim.documentation_completeness),
        "cpt_code": str(claim.cpt_code).strip(),
        "primary_icd10_dx": str(claim.primary_icd10_dx).strip(),
        "claim_amount_usd": float(claim.claim_amount_usd),
        "auth_status": claim.auth_status,
    }
    return pd.DataFrame([row])[art.selected_features]


# --------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------
def _risk_level(prob: float) -> str:
    bands = get_artifacts().metadata.get("denial_classification", {}).get("risk_bands", {})
    low = bands.get("low_max", 0.33)
    high = bands.get("medium_max", 0.66)
    if prob <= low:
        return "Low"
    if prob <= high:
        return "Medium"
    return "High"


def _py(value: Any) -> Any:
    """Convert numpy scalars to native Python types so the API can serialise them."""
    return value.item() if isinstance(value, np.generic) else value


def _risk_factors(claim: ClaimInput) -> list[RiskFactor]:
    """Explain the prediction using the five selected features (plus optional payer context)."""
    art = get_artifacts()
    factors: list[RiskFactor] = []
    q = art.quantiles
    overall = float(art.dashboard.get("overall_denial_rate", 0.28))

    # 1) documentation completeness - the strongest driver in the model
    doc = float(claim.documentation_completeness)
    if doc < 0.6:
        factors.append(RiskFactor(
            factor="documentation_completeness", value=round(doc, 2), direction="increases_risk",
            detail="Documentation completeness is below 0.60 - this is the single strongest "
                   "denial driver in the model."))
    else:
        factors.append(RiskFactor(
            factor="documentation_completeness", value=round(doc, 2), direction="decreases_risk",
            detail="Documentation completeness is at or above 0.60, which lowers denial risk."))

    # 2) prior-authorisation status
    if claim.auth_status == "not_obtained":
        factors.append(RiskFactor(
            factor="auth_status", value="not_obtained", direction="increases_risk",
            detail="Prior authorisation was required but has NOT been obtained."))
    elif claim.auth_status == "unknown":
        factors.append(RiskFactor(
            factor="auth_status", value="unknown", direction="increases_risk",
            detail="Prior-authorisation status is unknown - confirm it before submission."))
    elif claim.auth_status == "obtained":
        factors.append(RiskFactor(
            factor="auth_status", value="obtained", direction="decreases_risk",
            detail="Prior authorisation was obtained, reducing auth-related denial risk."))
    else:  # not_required
        factors.append(RiskFactor(
            factor="auth_status", value="not_required", direction="neutral",
            detail="No prior authorisation is required for this claim."))

    # 3) billed amount vs historical distribution
    q75 = q.get("claim_amount_q75")
    if q75 is not None and claim.claim_amount_usd > q75:
        factors.append(RiskFactor(
            factor="claim_amount_usd", value=round(float(claim.claim_amount_usd), 2), direction="increases_risk",
            detail=f"Billed amount is above the historical 75th percentile (${q75:,.0f}); "
                   "high-value claims attract closer medical-necessity scrutiny."))
    else:
        factors.append(RiskFactor(
            factor="claim_amount_usd", value=round(float(claim.claim_amount_usd), 2), direction="neutral",
            detail="Billed amount is within the typical historical range."))

    # 4) procedure code
    factors.append(RiskFactor(
        factor="cpt_code", value=str(claim.cpt_code), direction="neutral",
        detail="Procedure code is one of the two most predictive categorical drivers "
               "(after documentation completeness)."))

    # 5) diagnosis code
    factors.append(RiskFactor(
        factor="primary_icd10_dx", value=str(claim.primary_icd10_dx), direction="neutral",
        detail="Primary diagnosis strongly influences medical-necessity and coding decisions."))

    # optional payer context
    if claim.payer_type and not art.payer_lookup.empty:
        m = art.payer_lookup[art.payer_lookup["payer_type"] == claim.payer_type]
        if not m.empty:
            payer_rate = float(m.iloc[0]["payer_denial_rate_full"])
            direction = "increases_risk" if payer_rate > overall * 1.05 else (
                "decreases_risk" if payer_rate < overall * 0.95 else "neutral")
            factors.append(RiskFactor(
                factor="payer_historical_denial_rate", value=round(payer_rate, 3), direction=direction,
                detail=f"{claim.payer_type} historically denies at {payer_rate:.1%} "
                       f"(overall {overall:.1%})."))

    return factors


def predict_claim(claim: ClaimInput) -> PredictionResult:
    art = get_artifacts()
    notes: list[str] = []
    row = build_feature_row(claim)

    result = PredictionResult(
        denial_probability=0.0, approval_probability=0.0,
        predicted_status="unknown", risk_level="unknown",
        features_used=list(art.selected_features),
    )

    # ---- Model 1: denial probability ----
    denial_model = art.models.get("denial")
    if denial_model is None:
        notes.append("Denial model is unavailable; denial probability could not be computed.")
        result.notes = notes
        return result

    prob = float(denial_model.predict_proba(row)[0, 1])
    result.denial_probability = round(prob, 4)
    result.approval_probability = round(1 - prob, 4)
    result.predicted_status = "Denied" if prob >= 0.5 else "Approved"
    result.risk_level = _risk_level(prob)
    result.risk_score = int(round(prob * 100))

    # ---- Model 3: denial reason (only when the claim is actually at risk) ----
    reason_obj = art.models.get("denial_reason")
    if prob >= config.DENIAL_REASON_MIN_PROBABILITY and reason_obj is not None:
        pipe = reason_obj["pipeline"]
        classes = reason_obj["classes"]
        probs = pipe.predict_proba(row)[0]
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        result.predicted_denial_reason = classes[top_idx]
        result.denial_reason_confidence = round(float(probs[top_idx]), 4)
        result.denial_reason_top3 = [
            {"category": classes[int(i)], "probability": round(float(probs[int(i)]), 4)}
            for i in order[:3]
        ]
        result.predicted_x12_code = art.category_to_code.get(classes[top_idx])
        notes.append("Denial reason is a conditional prediction based on similar historical claims, "
                     "not a confirmed reason for this claim.")
    elif reason_obj is None:
        notes.append("Denial-reason model is unavailable.")
    else:
        notes.append("Denial probability is below the threshold, so no denial reason is predicted.")

    # ---- Model 2: expected recovery ----
    recovery_model = art.models.get("recovery")
    if recovery_model is None:
        notes.append("Recovery model is unavailable.")
    elif prob >= config.DENIAL_REASON_MIN_PROBABILITY:
        value = max(0.0, float(recovery_model.predict(row)[0]))
        result.predicted_recovery_usd = round(value, 2)
        result.recovery_label = "Expected recovery if denied and appealed (USD)"
        result.recovery_note = ("Expected RECOVERY amount if this claim is denied and appealed. "
                                "The dataset contains no paid/reimbursement amount, so this is the "
                                "only real monetary label available.")
    else:
        result.recovery_label = "Expected recovery (USD)"
        result.recovery_note = ("Recovery is only relevant for denied claims; the denial probability "
                                "is low, so no recovery estimate is returned.")

    result.risk_factors = _risk_factors(claim)
    result.notes = notes
    return result


# --------------------------------------------------------------------------------------
# Payer intelligence
# --------------------------------------------------------------------------------------
def payer_intelligence(payer_type: str) -> PayerIntelligence:
    art = get_artifacts()
    if art.payer_agg.empty:
        return PayerIntelligence(payer_type=payer_type, note="Payer aggregates are not available.")

    m = art.payer_agg[art.payer_agg["payer_type"] == payer_type]
    if m.empty:
        return PayerIntelligence(payer_type=payer_type,
                                 note="No historical data found for this payer type.")

    r = m.iloc[0]
    overall = float(art.dashboard.get("overall_denial_rate", np.nan))
    denial_rate = float(r["denial_rate"])
    if not math.isnan(overall):
        if denial_rate > overall * 1.05:
            vs = "above the overall denial rate"
        elif denial_rate < overall * 0.95:
            vs = "below the overall denial rate"
        else:
            vs = "in line with the overall denial rate"
    else:
        vs = "unknown"

    rule_rate = None
    if not art.payer_rules.empty:
        sub = art.payer_rules[art.payer_rules["payer_type"] == payer_type]
        if not sub.empty:
            rule_rate = round(float(sub["historical_denial_rate"].mean()), 4)

    return PayerIntelligence(
        payer_type=payer_type,
        historical_claims=int(r["claims"]),
        historical_denial_rate=round(denial_rate, 4),
        overall_denial_rate=round(overall, 4) if not math.isnan(overall) else None,
        denial_rate_vs_overall=vs,
        top_denial_reason=str(r.get("denial_category")) if pd.notna(r.get("denial_category")) else None,
        top_denial_code=str(r.get("denial_reason_code")) if pd.notna(r.get("denial_reason_code")) else None,
        avg_claim_amount=round(float(r["avg_claim_amount"]), 2),
        avg_recovery_usd=round(float(r["avg_recovery_usd"]), 2) if pd.notna(r.get("avg_recovery_usd")) else None,
        rule_table_denial_rate=rule_rate,
    )


def payer_rule_details(payer_type: str, limit: int = 12) -> list[dict]:
    """Per-(payer, CPT) rule rows for the payer-intelligence view."""
    art = get_artifacts()
    if art.payer_rules.empty:
        return []
    sub = art.payer_rules[art.payer_rules["payer_type"] == payer_type].head(limit)
    cols = ["cpt_code", "requires_prior_auth", "auth_lead_time_days", "historical_denial_rate",
            "avg_payment_turnaround_days", "timely_filing_limit_days"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].to_dict(orient="records")


def available_payers() -> list[str]:
    art = get_artifacts()
    if art.payer_agg.empty:
        return []
    return sorted(art.payer_agg["payer_type"].dropna().unique().tolist())


def available_options() -> dict[str, list]:
    """Valid categorical values for the frontend form dropdowns (read from the data)."""
    art = get_artifacts()
    small = _categorical_values()
    return {
        "cpt_codes": sorted(small["cpt_code"].dropna().astype(str).unique().tolist()),
        "icd10_codes": sorted(small["primary_icd10_dx"].dropna().astype(str).unique().tolist()),
        "auth_statuses": list(AUTH_STATUSES),
        "payer_types": available_payers(),
        "denial_categories": list(art.dashboard.get("denial_category_counts", {}).keys()),
        "selected_features": list(art.selected_features),
        "feature_ranking": art.feature_selection.get("ranking", {}),
    }


@lru_cache(maxsize=1)
def _categorical_values() -> pd.DataFrame:
    return pd.read_csv(
        config.DATA_DIR / "claims_model_ready.csv",
        usecols=["cpt_code", "primary_icd10_dx"],
    )


def dashboard_stats() -> dict:
    art = get_artifacts()
    return art.dashboard


def model_report() -> dict:
    art = get_artifacts()
    return {
        "metadata": art.metadata,
        "selected_features": art.selected_features,
        "feature_selection": art.feature_selection,
    }
