"""FastAPI backend for the AI-Powered Medical Claim Denial Predictor.

Models are loaded once at application startup; the API never retrains on the fly.
The models were trained on a **feature-selected top-5** subset of claim attributes,
so the only required claim inputs are those five drivers.

Endpoints
---------
GET  /                                  Web UI (single-page app)
GET  /api                                API information (JSON)
GET  /health                            backend / model / LLM status
GET  /options                           valid categorical values + feature ranking
GET  /dashboard                         aggregated dataset statistics
GET  /models                            model metrics + feature-selection report
GET  /payers                            available payer names
GET  /payer-intelligence/{payer_name}   historical payer analysis
GET  /payer-rules/{payer_name}          per-(payer, CPT) rule rows
POST /predict                           ML predictions only
POST /analyze                           full ML + payer intelligence + LLM workflow
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import agent_workflow, config, llm_service, prediction_service
from .schemas import (AnalyzeResponse, ClaimInput, HealthResponse, LLMReport,
                      OptionsResponse, PayerIntelligence, PredictResponse)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("claim-denial-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    art = prediction_service.get_artifacts()
    logger.info("Models loaded: %s", prediction_service.models_loaded())
    logger.info("Model features (top-5): %s", art.selected_features)
    if art.errors:
        for err in art.errors:
            logger.warning("Artifact warning: %s", err)
    logger.info("LLM available: %s (model=%s, keys=%d)",
                llm_service.llm_available(), config.LLM_MODEL, llm_service.configured_key_count())
    yield


app = FastAPI(
    title="AI-Powered Medical Claim Denial Predictor API",
    description=("Predict medical-claim denial risk before submission, estimate expected "
                 "recovery, predict the most likely denial reason, and generate LLM-backed "
                 "recommendations using historical payer intelligence. Models use a "
                 "feature-selected top-5 set of claim attributes."),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


# --------------------------------------------------------------------------------------
# JSON API
# --------------------------------------------------------------------------------------
@app.get("/api", summary="API information")
def api_info():
    return {
        "name": "AI-Powered Medical Claim Denial Predictor & Payer Intelligence System",
        "version": "2.0.0",
        "model_features": prediction_service.get_artifacts().selected_features,
        "endpoints": ["/health", "/options", "/dashboard", "/models", "/payers",
                      "/payer-intelligence/{payer_name}", "/payer-rules/{payer_name}",
                      "/predict", "/analyze", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, summary="Backend, model and LLM status")
def health():
    loaded = prediction_service.models_loaded()
    return HealthResponse(
        status="ok" if all(loaded.values()) else "degraded",
        models_loaded=loaded,
        llm_available=llm_service.llm_available(),
        llm_model=config.LLM_MODEL if llm_service.llm_available() else None,
        configured_keys=llm_service.configured_key_count(),
        detail=None if all(loaded.values()) else "One or more models failed to load.",
    )


@app.get("/options", response_model=OptionsResponse, summary="Valid categorical values")
def options():
    return prediction_service.available_options()


@app.get("/dashboard", summary="Aggregated dataset statistics")
def dashboard():
    return prediction_service.dashboard_stats()


@app.get("/models", summary="Model metrics + feature-selection report")
def models():
    return prediction_service.model_report()


@app.get("/payers", summary="Available payer names")
def payers():
    return {"payers": prediction_service.available_payers()}


@app.get("/payer-intelligence/{payer_name}", response_model=PayerIntelligence,
         summary="Historical payer analysis")
def payer_intel(payer_name: str):
    intel = prediction_service.payer_intelligence(payer_name)
    if intel.historical_claims is None and intel.note:
        # unknown payer -> 404 so the UI can show a clear message
        raise HTTPException(status_code=404, detail=intel.note)
    return intel


@app.get("/payer-rules/{payer_name}", summary="Per-(payer, CPT) rule rows")
def payer_rules(payer_name: str):
    return {"payer_type": payer_name, "rules": prediction_service.payer_rule_details(payer_name)}


@app.post("/predict", response_model=PredictResponse, summary="ML predictions only")
def predict(claim: ClaimInput):
    try:
        prediction = prediction_service.predict_claim(claim)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictResponse(claim=claim, prediction=prediction)


@app.post("/analyze", response_model=AnalyzeResponse,
          summary="Complete ML + payer intelligence + LLM workflow")
def analyze(claim: ClaimInput):
    try:
        workflow = agent_workflow.run_analysis(claim)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    return AnalyzeResponse(
        claim=claim,
        prediction=workflow["claim_analysis"]["prediction"],
        payer_intelligence=workflow["payer_intelligence"]["payer_intelligence"],
        recommendation=LLMReport(**workflow["recommendation"]["llm"]),
    )


# --------------------------------------------------------------------------------------
# Static single-page application (served last so it never shadows the API routes)
# --------------------------------------------------------------------------------------
_STATIC_DIR = config.FRONTEND_DIR
if _STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR)), name="assets")

    @app.get("/", include_in_schema=False)
    def spa_index():
        return FileResponse(_STATIC_DIR / "index.html")
else:  # pragma: no cover
    logger.warning("Frontend static directory not found at %s; UI will not be served.", _STATIC_DIR)

    @app.get("/", include_in_schema=False)
    def root_fallback():
        return {"detail": "UI assets not found. API is available under /docs."}
