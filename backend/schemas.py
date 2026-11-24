"""Pydantic request / response schemas for the FastAPI backend.

The models are trained on a **feature-selected** top-5 subset of claim attributes
(see ``data/processed/feature_columns.json`` -> ``selected_features``):

    documentation_completeness, cpt_code, primary_icd10_dx,
    claim_amount_usd, auth_status

``ClaimInput`` therefore only asks for those five drivers (plus optional
context that is *not* fed to the models, such as the submission date and the
payer name used by the Payer-Intelligence agent).
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# The four actionable values of the selected ``auth_status`` feature.
AUTH_STATUSES = ["not_required", "obtained", "not_obtained", "unknown"]


class ClaimInput(BaseModel):
    """The five predictive features (+ optional non-model context) for one claim."""

    # ---- the 5 selected model features ----
    documentation_completeness: float = Field(
        ..., ge=0, le=1,
        description="Documentation completeness score 0-1 (strongest denial driver)")
    cpt_code: str = Field(..., description="CPT procedure code")
    primary_icd10_dx: str = Field(..., description="Primary ICD-10 diagnosis code")
    claim_amount_usd: float = Field(..., gt=0, description="Billed charge amount in USD")
    auth_status: str = Field(
        ..., description="Prior-authorisation status: one of " + ", ".join(AUTH_STATUSES))

    # ---- optional context (NOT model features) ----
    claim_submission_date: Optional[date] = Field(
        None, description="Date the claim is submitted (YYYY-MM-DD); used for advisory context only")
    payer_type: Optional[str] = Field(
        None, description="Insurance payer type; used only for payer-history intelligence")

    model_config = {
        "json_schema_extra": {
            "example": {
                "documentation_completeness": 0.45,
                "cpt_code": "93454",
                "primary_icd10_dx": "I48.91",
                "claim_amount_usd": 1500.0,
                "auth_status": "not_obtained",
                "claim_submission_date": "2024-05-14",
                "payer_type": "Medicare_Advantage",
            }
        }
    }

    @field_validator("documentation_completeness")
    @classmethod
    def _round_doc(cls, v: float) -> float:
        return round(float(v), 4)

    @field_validator("cpt_code", "primary_icd10_dx")
    @classmethod
    def _strip(cls, v: str) -> str:
        return str(v).strip()

    @field_validator("auth_status")
    @classmethod
    def _valid_auth_status(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in AUTH_STATUSES:
            raise ValueError(f"auth_status must be one of {AUTH_STATUSES}")
        return v

    @field_validator("payer_type")
    @classmethod
    def _blank_payer_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None or str(v).strip() == "":
            return None
        return str(v).strip()


class RiskFactor(BaseModel):
    factor: str
    value: Any
    detail: str
    direction: Literal["increases_risk", "decreases_risk", "neutral"] = "neutral"


class PredictionResult(BaseModel):
    denial_probability: float
    approval_probability: float
    predicted_status: str
    risk_level: str
    risk_score: Optional[int] = None
    predicted_denial_reason: Optional[str] = None
    predicted_x12_code: Optional[str] = None
    denial_reason_confidence: Optional[float] = None
    denial_reason_top3: Optional[list[dict]] = None
    predicted_recovery_usd: Optional[float] = None
    recovery_label: Optional[str] = None
    recovery_note: Optional[str] = None
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    features_used: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PayerIntelligence(BaseModel):
    payer_type: str
    historical_claims: Optional[int] = None
    historical_denial_rate: Optional[float] = None
    overall_denial_rate: Optional[float] = None
    denial_rate_vs_overall: Optional[str] = None
    top_denial_reason: Optional[str] = None
    top_denial_code: Optional[str] = None
    avg_claim_amount: Optional[float] = None
    avg_recovery_usd: Optional[float] = None
    rule_table_denial_rate: Optional[float] = None
    note: Optional[str] = None


class LLMReport(BaseModel):
    available: bool
    model: Optional[str] = None
    report: Optional[dict] = None
    status: Optional[str] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    claim: ClaimInput
    prediction: PredictionResult


class AnalyzeResponse(BaseModel):
    claim: ClaimInput
    prediction: PredictionResult
    payer_intelligence: PayerIntelligence
    recommendation: LLMReport


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    llm_available: bool
    llm_model: Optional[str] = None
    configured_keys: int
    detail: Optional[str] = None


class OptionsResponse(BaseModel):
    cpt_codes: list[str]
    icd10_codes: list[str]
    auth_statuses: list[str]
    payer_types: list[str]
    denial_categories: list[str]
    selected_features: list[str]
    feature_ranking: dict[str, float]
