# AI-Powered Medical Claim Denial Predictor & Payer Intelligence System
An end-to-end academic project that predicts **medical claim denial risk before submission**,
identifies the **most likely denial reason**, estimates **expected recovery**, analyses
**payer behaviour**, and produces **LLM-generated recommendations** — all backed by three
independently trained and tuned machine-learning models.

The system is built with a single data/EDA notebook, three model-training notebooks, a
FastAPI backend, a simple three-agent workflow, an OpenAI-compatible LLM layer, and a
**custom single-page web frontend** (HTML/CSS/vanilla-JS served directly by FastAPI).

> **Feature selection:** instead of feeding all 29 engineered columns to the models, the
> pipeline ranks the *enterable* claim attributes by multi-model XGBoost gain importance and
> keeps only the **top 5** — `documentation_completeness`, `cpt_code`, `primary_icd10_dx`,
> `claim_amount_usd`, `auth_status`. The web form therefore asks for exactly those five
> inputs, and accuracy is essentially unchanged versus the full feature set.

---

## 1. Project Overview

Healthcare providers lose significant revenue to claim denials that are often preventable.
Denials are frequently caused by missing prior authorisation, incomplete documentation,
coding errors, and payer-specific policies. If a denial risk can be estimated **before the
claim is submitted**, the billing team can act early.

This project delivers:

| Capability | Delivered by |
|---|---|
| Denial probability, approval probability, predicted status, Low/Medium/High risk | Model 1 – Denial classification |
| Expected recovery amount for a denied claim | Model 2 – Recovery regression |
| Most likely denial reason + representative X12 code (conditional) | Model 3 – Denial reason boosting |
| Historical payer statistics and trends | Payer intelligence layer |
| Professional, structured recommendations | GenAI (LLM) layer |
| Orchestration of the whole flow | Lightweight three-agent workflow |

**Design goals:** simple, modular, beginner-friendly, Windows-compatible, and free of
unnecessary infrastructure. No Docker, no React, no databases, no agent frameworks.

---

## 2. Problem Statement

> Given the five key attributes of a medical claim (documentation completeness, CPT procedure
> code, primary ICD-10 diagnosis, billed amount, and prior-authorisation status), predict
> whether the claim will be **denied**, estimate the **expected recovery** if it is
> denied, and predict the **most likely denial reason** — then turn those predictions plus
> historical payer intelligence into clear, actionable guidance.

Denial prediction is **business-critical**, so the classification model is tuned with a
deliberate emphasis on **recall** (do not let a claim that will be denied slip through as
"approved") while keeping precision reasonable.

---

## 3. Dataset Description

**DenialIQ: 120K Medical Claims | X12 Denial Codes** (synthetic, for education/analysis).

Raw files live in `data/raw/` and are **never modified**:

| File | Rows | Key columns |
|---|---|---|
| `claims_main.csv` | 120,000 | `claim_id`, `claim_submission_date`, `claim_year`, `claim_quarter`, `payer_type`, `provider_specialty`, `place_of_service_code`, `cpt_code`, `modifier`, `primary_icd10_dx`, `secondary_icd10_dx`, `secondary_dx_count`, `prior_auth_required`, `prior_auth_obtained`, `documentation_completeness`, `claim_amount_usd`, `outcome`, `denial_reason_code`, `denial_category`, `dataset_version`, `synthetic_flag` |
| `denial_labels.csv` | 33,664 | `claim_id`, `denial_category`, `denial_reason_code`, `denial_code_description`, `appealable`, `appeal_success_probability`, `recovery_action`, `estimated_recovery_usd` |
| `payer_rules.csv` | payer × CPT | `payer_type`, `cpt_code`, `requires_prior_auth`, `auth_lead_time_days`, `historical_denial_rate`, `avg_payment_turnaround_days`, `timely_filing_limit_days` |
| `train_test_split.csv` | 120,000 | `claim_id`, `split` (train / val / test, 70/15/15) |
| `llm_finetune.jsonl` | — | balanced paid/denied examples (not required to run this project) |

**Outcome distribution (120,000 claims):**

| Outcome | Claims |
|---|---|
| paid | 62,379 |
| denied | 33,664 |
| partial_pay | 15,469 |
| pending | 8,488 |

* Overall denial rate (all claims): **28.05%**
* Unique payers: **6**
* Average claim amount: **$2,223.58**
* Average expected recovery (denied claims): **$965.58**

**Denial categories (7):** `medical_necessity`, `coding_error`, `auth_missing`,
`eligibility`, `duplicate`, `timely_filing`, `bundling`.

**Important dataset fact:** the dataset contains **no paid / reimbursement amount column**.
The only real monetary label is `estimated_recovery_usd` in `denial_labels.csv`, which is
why Model 2 predicts **expected recovery** rather than "reimbursement" (see §7).

---

## 4. Architecture

```
                         ┌──────────────────────────────────────────┐
   data/raw/*.csv  ──►   │ 01_data_cleaning_and_aggregation.ipynb    │  ──►  data/processed/*.csv
                         │ (cleaning, EDA, feature engineering,      │       feature_columns.json
                         │  leakage-safe payer aggregation,          │       dashboard_stats.json
                         │  multi-model gain-importance FEATURE      │
                         │  SELECTION → top-5 feature set)           │
                         └──────────────────────────────────────────┘
                                             │
             ┌───────────────────────────────┼───────────────────────────────┐
             ▼                               ▼                               ▼
   02_denial_classification.ipynb   03_reimbursement_regression.ipynb   04_denial_reason_boosting.ipynb
        XGBoost (binary)                 XGBoost (regressor)                 XGBoost (multiclass)
             │                               │                               │
             └──────────────► models/*.pkl + model_metadata.json ◄───────────┘
                                             │
                                             ▼
   Web UI (HTML/CSS/JS)  ──HTTP──►  FastAPI backend  ──►  Agent 1 (Claim Analysis: 3 ML models)
   frontend/static/*                 backend/*.py         Agent 2 (Payer Intelligence)
                                                          Agent 3 (Recommendation → LLM)
                                                                │
                                                                ▼
                                                   OpenAI-compatible LLM (structured JSON report)
```

* The web UI is a static single-page app served by FastAPI at `/` (assets under `/assets`).
* **ML inference is completely independent of the LLM.** If no API key works, predictions
  still return and the recommendation is produced by the built-in **local rules-based
  advisor** (the UI shows an offline status instead of an error).
* Models are loaded **once** at FastAPI startup; the application never retrains.

---

## 5. Data Cleaning (`notebooks/01_data_cleaning_and_aggregation.ipynb`)

All cleaning, EDA, feature engineering and aggregation happen in this **single** notebook.

Performed:

* Dataset shape and column inspection; data-dictionary driven typing.
* Missing-value analysis and duplicate-claim detection.
* Data-type corrections (dates, booleans, numeric codes).
* Outlier analysis on claim amount and documentation score (IQR based).
* Categorical value analysis (payer type, specialty, CPT, modifier, POS, ICD-10).
* Numerical distribution analysis.
* Correlation analysis of numeric features.
* Outcome distribution, payer-wise denial analysis, denial-code analysis.
* Claim amount / recovery analysis.

**Feature engineering (29 candidate model columns):**

* **Categorical (8):** `payer_type`, `provider_specialty`, `place_of_service_code`,
  `cpt_code`, `modifier`, `primary_icd10_dx`, `claim_quarter`, `auth_status`
  (`not_required` / `obtained` / `not_obtained` / `unknown`).
* **Numeric (21):** calendar (`claim_year`, `claim_month`, `claim_dayofweek`),
  `secondary_dx_count`, `documentation_completeness`, `claim_amount_usd`,
  `log_claim_amount`, `has_secondary_dx`, `has_modifier`, `prior_auth_required_int`,
  `pa_gap` (auth required but not obtained), `doc_low` (documentation < 0.60), and
  historical payer features (`payer_denial_rate_hist`, `payer_cpt_denial_rate_hist`,
  `payer_hist_claims`, `payer_cpt_hist_claims`, `payer_avg_amount_hist`,
  `payer_rule_denial_rate`, `payer_rule_auth_lead_time_days`,
  `payer_rule_turnaround_days`, `payer_rule_timely_filing_days`).

**Leakage control:**

* `denial_reason_code` and `denial_category` exist **only on denied rows** and are
  therefore excluded from the Model 1 and Model 3 feature sets (they are the Model 3 target).
* `outcome` is excluded from every feature set (it is the Model 1 target source).
* Payer-history features are computed with **5-fold out-of-fold (OOF) encoding** so a row
  never sees its own label in its payer features; the final full-data lookup tables are
  shipped separately for inference.

**Feature selection (top-5):**

All three quick XGBoost models are trained on the training split over the *directly
enterable* claim attributes, their **gain importance** is aggregated per original feature
(one-hot columns are summed back to their parent feature) and normalised to 100%. The
resulting ranking (stored in `feature_columns.json -> feature_selection.ranking`) is:

| Rank | Feature | Gain % |
|---|---|---|
| 1 | documentation_completeness | 26.61 |
| 2 | primary_icd10_dx | 20.69 |
| 3 | cpt_code | 19.54 |
| 4 | modifier | 6.70 |
| 5 | place_of_service_code | 6.41 |
| 6 | auth_status | 5.19 |
| 7 | provider_specialty | 5.11 |
| 8 | claim_amount_usd | 4.91 |
| 9 | payer_type | 4.20 |
| 10 | secondary_dx_count | 0.65 |

Candidate top-5 sets were retrained end-to-end and compared on held-out metrics; the chosen
set — `documentation_completeness`, `cpt_code`, `primary_icd10_dx`, `claim_amount_usd`,
`auth_status` — gave the best combined accuracy (see §7) while cutting the input space from
29 columns to 5. The selected set is persisted as `selected_features` /
`selected_categorical` / `selected_numeric` in `feature_columns.json` and drives the model
pipelines and the API/UI contract.

**Outputs (`data/processed/`):** `claims_model_ready.csv`, `claims_clean.csv`,
`payer_aggregates.csv`, `denial_code_aggregates.csv`, `lookup_payer.csv`,
`lookup_payer_cpt.csv`, `lookup_payer_rules.csv`, `feature_columns.json`
(includes `selected_features`, the feature-selection ranking and the amount quantiles used
by the risk-factor explainer), `dashboard_stats.json`.

---

## 6. The Three Machine Learning Models

Every model uses a Scikit-learn `ColumnTransformer` pipeline
(`OneHotEncoder(handle_unknown="ignore")` for categoricals, `SimpleImputer` + `StandardScaler`
for numerics) fitted **only on the training split**, then tuned with
**`RandomizedSearchCV(n_iter=10)`** using the provided 70/15/15 split and reproducible
random states. The best estimator is refit and evaluated on the untouched test split.

All three models consume the **same feature-selected top-5 input**
(`documentation_completeness`, `cpt_code`, `primary_icd10_dx`, `claim_amount_usd`,
`auth_status`, read from `feature_columns.json -> selected_features`), so the API and the
web form stay consistent across models. `model_metadata.json` records `features_used` for
each model.

### Model 1 — Denial Classification (`02_denial_classification.ipynb`)

* **Algorithm:** XGBoost `binary:logistic`.
* **Target:** `is_denied` (1 = denied, 0 = paid / partial_pay). `pending` rows are excluded.
* **Imbalance:** `scale_pos_weight` tuned around the training denial rate.
* **Tuning metric:** ROC-AUC, 3-fold `StratifiedKFold`.
* **Outputs:** denial probability, approval probability, predicted status, risk level.

### Model 2 — Recovery Regression (`03_reimbursement_regression.ipynb`)

* **Algorithm:** XGBoost `reg:squarederror`.
* **Target:** `estimated_recovery_usd` (real label, denied claims only).
* **Why not "reimbursement":** the dataset has no paid amount column, so predicting a
  reimbursement figure would require fabricating a target. Instead the model predicts the
  **expected recovery** — the only real monetary label — and the limitation is reported in
  the API/UI (`reimbursement_note`).
* **Tuning metric:** RMSE, `KFold`.

### Model 3 — Denial Reason Boosting (`04_denial_reason_boosting.ipynb`)

* **Algorithm:** XGBoost `multi:softprob` (multiclass boosting).
* **Target:** `denial_category` (7 classes).
* **Leakage:** all denial-code / denial-category columns are excluded from the features.
* **Tuning metric:** macro-F1 (robust to class imbalance), 3-fold `StratifiedKFold`.
* **Uncertainty:** the model is only surfaced for claims above the denial-probability
  threshold, and the API returns the top-3 reasons with probabilities and explicitly labels
  the result a **conditional prediction**.

---

## 7. Model Evaluation

All numbers below are the actual values stored in `models/model_metadata.json`
(evaluated on the untouched test split — nothing is hard-coded in the UI).

**Model 1 — Denial classification** (test rows: 16,709)

| Metric | Value |
|---|---|
| Accuracy | 0.9168 |
| Precision | 0.7859 |
| Recall | 1.0000 |
| F1 | 0.8801 |
| ROC-AUC | 0.9857 |
| PR-AUC | 0.9694 |
| CV ROC-AUC (best) | 0.9854 |

Risk bands: **Low ≤ 0.33**, **Medium ≤ 0.66**, **High > 0.66**.
Model inputs (feature-selected top-5): `documentation_completeness`, `cpt_code`,
`primary_icd10_dx`, `claim_amount_usd`, `auth_status`.

**Model 2 — Recovery regression** (test rows: 5,102; target mean $961.82)

| Metric | Value |
|---|---|
| MAE | $394.92 |
| RMSE | $1,013.63 |
| R² | 0.7847 |
| CV RMSE (best) | $1,110.42 |

Model inputs (feature-selected top-5): `documentation_completeness`, `cpt_code`,
`primary_icd10_dx`, `claim_amount_usd`, `auth_status`.

**Model 3 — Denial reason** (test rows: 5,102; 7 classes)

| Metric | Value |
|---|---|
| Accuracy | 0.3503 |
| Precision (macro) | 0.2215 |
| Recall (macro) | 0.2266 |
| Macro F1 | 0.2057 |
| Weighted F1 | 0.3041 |
| CV macro-F1 (best) | 0.2090 |

The reason model separates the **frequent, structured** causes better than the rare ones
and is therefore always presented as a **conditional** prediction with top-3 alternatives
and a confidence value — never as a confirmed reason (see §16 Limitations).

**Visualisations (`reports/`):** confusion matrices, ROC/PR curves, XGBoost feature
importance, permutation importance, actual-vs-predicted and residual plots, distributions.

**Explainability:** tree-based feature importance **and** permutation importance are
exported for the classification and regression models, and `reports/01_feature_selection.png`
shows the gain-importance ranking. The API reports only model-supported risk factors with the
claim's actual values — documentation completeness vs the 0.60 threshold, prior-authorisation
status, billed amount vs the historical 75th percentile, the procedure/diagnosis codes, and
(only when a payer is supplied) the payer's historical denial rate. The LLM never invents
claim-specific factors.

---

## 8. GenAI Integration

The LLM interprets the combined ML predictions and historical payer statistics and returns a
**structured JSON report**; it never computes or changes a model number.

* **Provider:** NVIDIA NIM — an OpenAI-compatible endpoint (`https://integrate.api.nvidia.com/v1`)
  called through the official `openai` SDK.
* **Key:** a single NVIDIA NIM key read from `NVIDIA_API_KEY` (never committed).
* **Models (fallback chain):** an **ordered chain of models** (configurable via the
  comma-separated `NVIDIA_MODELS` env var). The first model is tried first; if it cannot
  serve the request the next model in the chain is tried, and so on. Built-in order:
  `nvidia/nemotron-3.5-bolt-e4b`, `nvidia/nemotron-3.5-lightning-30b-a3b`,
  `minimaxai/minimax-m3.1-flash`, `nvidia/nemotron-3-ultra-550b-a55b`,
  `nvidia/nemotron-3.5-super-vl-120b-a12b`, `moonshotai/kimi-k3.1-pro`,
  `mistralai/mistral-large-4-1t-a52b`, `stepfun-ai/step-5-preview`, `z-ai/glm-5.3-flash`.
* **Fallback policy:** on authentication / permission / connection / timeout failures, or a
  **quota/billing** failure (`insufficient_quota` / "no credits remaining"), the model is
  treated as unavailable and the **next model in the chain is tried**. **Transient rate
  limits (HTTP 429)** are handled on the *same* model using the `Retry-After` header and
  exponential back-off before falling through. If a model rejects JSON mode or returns an
  incomplete/malformed report, it is retried once without `response_format`, then skipped.
* **Never shows an error:** if no model in the chain can serve the request, the **local
  rules-based advisor**
  (`llm_service.build_fallback_report`) deterministically builds the *same* 8-section report
  from the ML predictions and payer statistics. The UI therefore **always** renders a complete
  recommendation, labelled `local-analyst (offline)`, with a friendly status message instead of
  a raw error.
* Keys are **never hard-coded, logged, or returned** to the client.

**Report structure** (exact keys requested from the model):
`executive_summary`, `claim_risk_assessment`, `predicted_denial_reason`,
`important_risk_factors`, `recommended_actions`, `billing_code_review`,
`reimbursement_insights`, `final_recommendation`.

`final_recommendation` is one of:
*"Ready for submission after routine verification"*,
*"Review recommended before submission"*,
*"High-priority review required before submission"*.

The system prompt (in `backend/llm_service.py`) enforces: never modify predictions, never
present predictions as confirmed, never invent codes/policies, distinguish historical
statistics from claim-specific predictions, mention missing information, never guarantee
approval.

---

## 9. Agentic Workflow

A lightweight workflow built from plain Python functions (`backend/agent_workflow.py`) —
no external agent framework.

1. **Agent 1 – Claim Analysis Agent:** accepts the claim, runs all three ML models,
   combines the outputs, and collects model-supported risk indicators.
2. **Agent 2 – Payer Intelligence Agent:** retrieves historical payer statistics, identifies
   denial trends and common denial codes, compares the payer denial rate with the overall
   rate, and flags insufficient historical data (< 50 claims).
3. **Agent 3 – Recommendation Agent:** assembles a verified LLM payload, calls the LLM,
   validates/normalises the JSON output, and guarantees a safe `final_recommendation`.

`run_analysis()` orchestrates Agent 1 → Agent 2 → Agent 3.

---

## 10. FastAPI Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Single-page web UI (`frontend/static/index.html`). |
| `GET` | `/assets/*` | Static UI assets (`styles.css`, `app.js`). |
| `GET` | `/api` | API information, model feature list and endpoint index. |
| `GET` | `/health` | Backend status, per-model load status, LLM availability and configured key count. |
| `GET` | `/options` | Valid categorical values + `selected_features` + the gain-importance `feature_ranking` (populates the UI dropdowns and the feature chart). |
| `GET` | `/dashboard` | Aggregated dataset statistics (KPIs, outcomes, monthly trend, payer rates, denial codes). |
| `GET` | `/models` | Model metrics, risk bands and the feature-selection report. |
| `GET` | `/payers` | Available payer names. |
| `GET` | `/payer-intelligence/{payer_name}` | Historical payer analysis (404 for unknown payers). |
| `GET` | `/payer-rules/{payer_name}` | Per-(payer, CPT) rule rows for the payer-intelligence view. |
| `POST` | `/predict` | ML predictions only (no LLM). |
| `POST` | `/analyze` | Full ML + payer intelligence + LLM workflow. |

Models are loaded once during application startup (FastAPI `lifespan`). Pydantic validates
every request (`auth_status` must be one of the four allowed values; documentation score is
clamped to 0–1); unknown categorical values are handled by `handle_unknown="ignore"`.
Interactive Swagger docs are available at `/docs`.

---

## 11. Web UI Features

`frontend/static/` — a dependency-free single-page app (vanilla HTML/CSS/JS) served by
FastAPI at `/`. Dark, glassmorphic healthcare-analytics theme with a left navigation rail
and a live status header (models + AI engine). Four views:

1. **Analyze Claim** — a form built from **exactly the five selected model features**
   (documentation-completeness slider with meter, searchable CPT / ICD-10 inputs, billed
   amount, prior-authorisation segmented control) plus an optional collapsed section for
   non-model context (payer, submission date). On **Analyze Claim** it calls `/analyze` and
   renders:
   * denial-probability gauge + approval probability + Low/Medium/High risk badge and 0–100 risk score;
   * predicted denial reason with X12 code and top-3 probability bars (Model 3);
   * expected recovery amount with its explanatory note (Model 2);
   * a colour-coded **risk-factor** list with the claim's actual values;
   * the full **AI report** card — executive summary, risk assessment, denial reason,
     risk factors, recommended actions, billing-code review, reimbursement insights and a
     colour-coded final verdict. If the LLM cannot serve the request, the **local rules-based
     advisor** fills the same card (labelled `local-analyst (offline)`) with a friendly
     offline-status banner — the recommendation is **never** replaced by an error.
2. **Dashboard** — KPI cards (total/denied/approved claims, denial rate, unique payers,
   average claim, average recovery) plus hand-rolled SVG/CSS charts: outcome donut, denial-reason
   bars, monthly claims-vs-denials line chart, payer denial rates and top denial codes.
3. **Payer Intelligence** — payer selector with historical claims, denial rate and comparison
   to the overall rate, top denial reason/code, average claim/recovery, and a per-(payer, CPT)
   policy rule table (prior-auth requirement, auth lead time, historical denial rate, payment
   turnaround, timely-filing limit).
4. **Models & Features** — the gain-importance **feature-selection** chart (selected top-5
   highlighted) and metric cards for all three models read live from `/models`. Nothing is
   hard-coded.

The UI consumes only the FastAPI JSON endpoints, so the backend can be hosted separately if
desired.

---

## 12. Project Structure

```
medical-claim-denial-predictor/
├── data/
│   ├── raw/                      # original dataset (untouched)
│   └── processed/                # cleaned + aggregated + model-ready artefacts
├── notebooks/
│   ├── 01_data_cleaning_and_aggregation.ipynb
│   ├── 02_denial_classification.ipynb
│   ├── 03_reimbursement_regression.ipynb
│   └── 04_denial_reason_boosting.ipynb
├── models/
│   ├── denial_model.pkl
│   ├── reimbursement_model.pkl
│   ├── denial_reason_model.pkl
│   └── model_metadata.json
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── prediction_service.py
│   ├── llm_service.py
│   ├── agent_workflow.py
│   └── config.py
├── frontend/
│   └── static/
│       ├── index.html          # single-page UI
│       ├── styles.css          # theme + layout
│       └── app.js              # data loading, charts, form logic
├── reports/                      # generated charts
├── .env                          # real secrets (git-ignored)
├── .env.example                  # placeholder keys only
├── .gitignore
├── requirements.txt
└── README.md
```

> Each saved `.pkl` model is a complete Scikit-learn pipeline (preprocessing + estimator),
> so inference uses the exact same preprocessing that was fitted during training.

---

## 13. Installation

Requires **Python 3.10+** (developed on Python 3.13) on Windows, macOS or Linux.

```powershell
# 1. go to the project folder
cd medical-claim-denial-predictor

# 2. create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

# 3. install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Configure the LLM keys (optional but required for recommendations):

```powershell
copy .env.example .env
# then edit .env and paste your NVIDIA NIM API key
```

`.env` example:

```
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

`.env` is listed in `.gitignore`; `.env.example` contains placeholders only.

---

## 14. Execution Instructions

Run the steps **in order**:

```powershell
# 1) Data cleaning + aggregation  -> writes data/processed/*
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_cleaning_and_aggregation.ipynb

# 2) Train the three models       -> writes models/*.pkl + model_metadata.json
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_denial_classification.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_reimbursement_regression.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_denial_reason_boosting.ipynb
```

(You can also open the notebooks in Jupyter and run all cells.)

**Start the backend + web UI** (from the project root so `backend` is a package):

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Then open:

* **Web UI:** `http://localhost:8000/`
* **Swagger docs:** `http://localhost:8000/docs`

The UI is served by the same FastAPI process, so no second terminal is needed.

**Quick API smoke test:**

```powershell
curl http://localhost:8000/health
```

---

## 15. Testing

Verified during development:

* Feature selection computed and the three models retrained on the chosen top-5 feature set.
* All three models trained and evaluated independently on the held-out test split.
* Model loading once at startup; `/health` reflects per-model status.
* API endpoints (`/`, `/assets/*`, `/api`, `/health`, `/options`, `/dashboard`, `/models`,
  `/payers`, `/payer-intelligence/{payer}`, `/payer-rules/{payer}`, `/predict`, `/analyze`)
  exercised end-to-end.
* Invalid inputs rejected by Pydantic (`auth_status` enum, documentation score range);
  unknown categorical values handled by `handle_unknown="ignore"`.
* Missing API keys → LLM layer skipped, ML predictions still returned.
* LLM failure (missing keys, every key out of credits, transient rate limit) → the local
  rules-based advisor produces the full report; no error is surfaced to the user.
* Complete web-UI workflow (form → `/analyze` → rendered report, dashboard, payer view and
  feature-selection charts).

---

## 16. Limitations

* **No paid/reimbursement column.** The dataset only provides `estimated_recovery_usd`, so
  Model 2 predicts **expected recovery**, not actual reimbursement. This is surfaced
  explicitly in the API/UI rather than fabricated.
* **Denial-reason model is weak on rare classes.** Denial reason is only partially
  predictable from pre-submission features; accuracy ≈ 0.34 and macro-F1 ≈ 0.20. Frequent,
  structured causes are predicted better than rare ones. The reason is always shown as a
  conditional prediction with top-3 alternatives and a confidence value.
* **Near-perfect classification is expected.** `documentation_completeness` is the dominant
  signal in the synthetic dataset and separates most claims, so Model 1's ROC-AUC is ~0.985.
  This reflects the dataset's design; the tuner still maximises recall for the business case.
* **Feature selection is a deliberate trade-off.** The models use only the top-5 enterable
  attributes. Dropping the remaining columns costs almost nothing in accuracy (Model 1 AUC
  0.9857 vs 0.9854 with all features; recovery R² 0.7847) and makes the system explainable and
  easy to use, but it means engineered/payer-history signals are not used at inference time —
  those remain available as advisory context for the LLM only.
* **Synthetic data.** All records are synthetic, so payer policies and rates illustrate the
  pipeline rather than real-world behaviour.
* **No live payer policy engine.** Recommendations are advisory and based only on model
  outputs and historical aggregates; the LLM never invents codes or policies.
* **Recovery prediction applies to denied claims only** and is returned only when the denial
  probability exceeds the configured threshold.

---

## 17. Technology Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Plotly ·
RandomizedSearchCV · Joblib · FastAPI · Pydantic · Uvicorn · OpenAI SDK · python-dotenv ·
vanilla HTML/CSS/JS (web UI) · Jupyter.

*No Docker, Kubernetes, React, databases, or agent frameworks are used.*
