"""A lightweight agentic workflow built from plain Python functions.

Agent 1 - Claim Analysis Agent   : runs all three ML models and combines their outputs.
Agent 2 - Payer Intelligence Agent: gathers historical payer statistics / trends.
Agent 3 - Recommendation Agent   : prepares the LLM input, calls the LLM, validates output.

The LLM never computes or changes a number produced by the models.
"""
from __future__ import annotations

from typing import Any, Optional

from . import llm_service, prediction_service
from .prediction_service import payer_intelligence, predict_claim
from .schemas import ClaimInput, PayerIntelligence


# --------------------------------------------------------------------------------------
# Agent 1: Claim Analysis Agent
# --------------------------------------------------------------------------------------
def claim_analysis_agent(claim: ClaimInput) -> dict:
    """Run the three ML models and combine their outputs into one structured result."""
    prediction = predict_claim(claim)
    return {
        "agent": "ClaimAnalysisAgent",
        "prediction": prediction,
        "model_supported_risk_indicators": [f.model_dump() for f in prediction.risk_factors],
    }


# --------------------------------------------------------------------------------------
# Agent 2: Payer Intelligence Agent
# --------------------------------------------------------------------------------------
def payer_intelligence_agent(payer_type: Optional[str]) -> dict:
    """Retrieve historical payer statistics and compare against the overall denial rate."""
    if not payer_type:
        intel = PayerIntelligence(
            payer_type="(not provided)",
            note="No payer type was supplied, so payer-history intelligence is unavailable.")
        return {"agent": "PayerIntelligenceAgent", "payer_intelligence": intel,
                "insufficient_data": True, "note": intel.note}

    intel = payer_intelligence(payer_type)
    result: dict[str, Any] = {"agent": "PayerIntelligenceAgent", "payer_intelligence": intel}

    if intel.historical_claims is None:
        result["insufficient_data"] = True
        result["note"] = "No historical claims were found for this payer type."
    else:
        result["insufficient_data"] = intel.historical_claims < 50
        if result["insufficient_data"]:
            result["note"] = ("Fewer than 50 historical claims for this payer; the statistics "
                              "should be treated as indicative only.")
    return result


# --------------------------------------------------------------------------------------
# Agent 3: Recommendation Agent
# --------------------------------------------------------------------------------------
def _build_llm_payload(claim: ClaimInput, prediction, payer_intel) -> dict:
    """Assemble only verified, model-supported information for the LLM."""
    art = prediction_service.get_artifacts()
    return {
        "claim": claim.model_dump(mode="json"),
        "model_input_features": {
            "selected_top5": list(art.selected_features),
            "note": ("The ML models were trained on a feature-selected top-5 subset only. "
                     "These five attributes are the sole model inputs; everything else is "
                     "advisory context."),
        },
        "ml_predictions": {
            "denial_probability": prediction.denial_probability,
            "approval_probability": prediction.approval_probability,
            "predicted_status": prediction.predicted_status,
            "risk_level": prediction.risk_level,
            "risk_score_0_100": prediction.risk_score,
            "predicted_denial_reason": prediction.predicted_denial_reason,
            "predicted_x12_code": prediction.predicted_x12_code,
            "denial_reason_confidence": prediction.denial_reason_confidence,
            "denial_reason_alternatives": prediction.denial_reason_top3,
            "predicted_recovery_usd": prediction.predicted_recovery_usd,
            "recovery_label": prediction.recovery_label,
            "recovery_note": prediction.recovery_note,
        },
        "model_supported_risk_indicators": [f.model_dump() for f in prediction.risk_factors],
        "historical_payer_statistics": payer_intel.model_dump(),
        "notes": prediction.notes,
        "instructions": {
            "note": ("All numbers above come from trained ML models and historical data. "
                     "Do not modify them. Distinguish historical statistics from this "
                     "claim-specific prediction. Base your reasoning on the five model "
                     "features listed in model_input_features.")
        },
    }


OFFLINE_STATUS = ("The AI assistant is running in offline mode. A complete recommendation was "
                  "generated locally by the built-in rules engine.")


def recommendation_agent(claim: ClaimInput, prediction, payer_intel) -> dict:
    """Prepare the LLM input, call the LLM and validate the structured response.

    If the LLM cannot serve the request (no keys, all keys out of credits, transient
    rate limit, etc.) the request is **never** failed and no raw error is shown: the
    deterministic local advisor produces the full report instead, with a friendly
    status message.
    """
    payload = _build_llm_payload(claim, prediction, payer_intel)
    result: dict[str, Any] = {"agent": "RecommendationAgent", "llm_input_payload": payload}

    try:
        out = llm_service.generate_report(payload)
        result["llm"] = {"available": True, "model": out["model"], "report": out["report"],
                         "status": None, "error": None}
        return result
    except llm_service.LLMUnavailable:
        # Fall through to the local advisor below - never surface the failure.
        pass

    result["llm"] = {
        "available": True,
        "model": "local-analyst (offline)",
        "report": llm_service.build_fallback_report(payload),
        "status": OFFLINE_STATUS,
        "error": None,
    }
    return result


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------
def run_analysis(claim: ClaimInput) -> dict:
    """Full workflow: Agent 1 -> Agent 2 -> Agent 3."""
    a1 = claim_analysis_agent(claim)
    a2 = payer_intelligence_agent(claim.payer_type)
    a3 = recommendation_agent(claim, a1["prediction"], a2["payer_intelligence"])
    return {
        "claim_analysis": a1,
        "payer_intelligence": a2,
        "recommendation": a3,
    }
