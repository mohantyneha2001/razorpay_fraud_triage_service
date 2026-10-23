
"""Part D: FastAPI fraud triage service."""

import logging
import time
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from train import build_final_model
from prepare import add_features
from retrieval import load_json, build_index
from casenote import run_case

app = FastAPI(title="Razorpay Merchant Risk Desk")

# Load the trained fraud model and policy retrieval index.
MODEL = build_final_model()

POLICIES = load_json(Path("fraud_policies.json"))
VECTORIZER, POLICY_MATRIX = build_index(POLICIES)

KNOWN_CATEGORIES = {
    "Grocery", "Electronics", "Travel",
    "Food Delivery", "Fashion", "Utilities"
}

logging.basicConfig(
    filename="service.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
LOGGER = logging.getLogger("fraud_triage")

COUNTERS = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "latencies": [],
    "human_review_total": 0
}


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: float | None = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)

    if request.url.path in {"/score", "/case"} and request.method == "POST":
        elapsed = (time.perf_counter() - start) * 1000
        COUNTERS["requests_total"] += 1
        COUNTERS["success_total"] += int(response.status_code < 400)
        COUNTERS["error_total"] += int(response.status_code >= 400)
        COUNTERS["latencies"].append(elapsed)

        LOGGER.info(
            "%s status=%s latency_ms=%.2f",
            request.url.path,
            response.status_code,
            elapsed
        )

    return response


@app.get("/health")
def health():
    return {"status": "ok"}


def validate_business(t: Txn):
    if not 1 <= t.amount_inr <= 1_000_000:
        return {"field": "amount_inr", "error": "Amount must be between 1 and 1000000"}

    if not 0 <= t.hour_of_day <= 23:
        return {"field": "hour_of_day", "error": "Hour must be between 0 and 23"}

    if t.txn_count_last_24h < 0:
        return {"field": "txn_count_last_24h", "error": "Transaction count cannot be negative"}

    if t.avg_amount_30d_inr <= 0:
        return {"field": "avg_amount_30d_inr", "error": "Average amount must be greater than zero"}

    if t.merchant_category.strip().title() not in KNOWN_CATEGORIES:
        return {"field": "merchant_category", "error": "Unknown merchant category"}

    if t.is_new_device not in (0, 1):
        return {"field": "is_new_device", "error": "Use 0 or 1"}

    if t.account_age_days < 0:
        return {"field": "account_age_days", "error": "Account age cannot be negative"}

    if t.distance_from_home_km is not None and t.distance_from_home_km < 0:
        return {"field": "distance_from_home_km", "error": "Distance cannot be negative"}

    return None


def score_transaction(t: Txn):
    payload = t.model_dump()
    payload["merchant_category"] = t.merchant_category.strip().title()
    payload["city"] = t.city.strip().title()

    row = pd.DataFrame([payload])
    row["distance_from_home_km"] = row["distance_from_home_km"].fillna(
        MODEL["median"]
    )

    featured = add_features(row)

    X = featured.drop(
        columns=[
            "txn_id", "customer_id", "chargeback_filed",
            "is_fraud", "question"
        ],
        errors="ignore"
    )
    X = pd.get_dummies(X, dtype=int)
    X = X.reindex(columns=MODEL["columns"], fill_value=0)

    probability = float(
        MODEL["model"].predict_proba(X)[:, 1][0]
    )

    flagged = probability >= MODEL["threshold"]
    decision = "hold_for_review" if flagged else "approve"

    # Additional business guardrail.
    override = False
    if not flagged and t.amount_inr >= 100_000 and t.is_new_device == 1:
        decision = "hold_for_review"
        override = True

    return probability, flagged, decision, override


def invalid_response(request: Request, txn_id: str, error):
    LOGGER.warning("Invalid transaction input: %s", txn_id)
    return JSONResponse(status_code=400, content=error)


@app.post("/score")
def score(t: Txn, request: Request):
    error = validate_business(t)
    if error:
        return invalid_response(request, t.txn_id, error)

    probability, flagged, decision, override = score_transaction(t)
    human_review = decision == "hold_for_review"

    COUNTERS["human_review_total"] += int(human_review)

    return {
        "txn_id": t.txn_id,
        "fraud_probability": round(probability, 4),
        "flagged": flagged,
        "decision": decision,
        "human_review": human_review,
        "guardrail_override": override
    }


@app.post("/case")
def case(t: CaseIn, request: Request):
    error = validate_business(t)
    if error:
        return invalid_response(request, t.txn_id, error)

    probability, flagged, decision, override = score_transaction(t)

    case_note = None
    attempts = 0
    safe_default = False
    retrieval_status = None

    if flagged:
        try:
            result = run_case(
                t.txn_id,
                probability,
                t.amount_inr,
                t.question,
                POLICIES,
                VECTORIZER,
                POLICY_MATRIX
            )
        except ConnectionError:
            return JSONResponse(
                status_code=503,
                content={"error": "Case-note service temporarily unavailable"}
            )

        attempts = result["attempts"]
        safe_default = result["safe_default"]
        override = override or result["guardrail_override"]
        case_note = result["case_note"]
        retrieval_status = result.get("status")

        if retrieval_status == "no_policy_match":
            decision = "hold_for_review"
        elif case_note is not None:
            if case_note["recommended_action"] == "approve":
                case_note["recommended_action"] = "hold_for_review"
                override = True

            decision = case_note["recommended_action"]

    human_review = (
        decision in {"hold_for_review", "block_and_notify"}
        or safe_default
    )
    COUNTERS["human_review_total"] += int(human_review)

    return {
        "txn_id": t.txn_id,
        "fraud_probability": round(probability, 4),
        "flagged": flagged,
        "decision": decision,
        "human_review": human_review,
        "case_note": case_note,
        "attempts": attempts,
        "safe_default": safe_default,
        "guardrail_override": override
    }


@app.get("/metrics")
def metrics():
    total = COUNTERS["requests_total"]
    successes = COUNTERS["success_total"]
    errors = COUNTERS["error_total"]
    latencies = COUNTERS["latencies"]

    return {
        "requests_total": total,
        "success_total": successes,
        "error_total": errors,
        "success_rate": round(successes / total, 4) if total else 0,
        "error_rate": round(errors / total, 4) if total else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3)
        if latencies else 0,
        "human_review_total": COUNTERS["human_review_total"]
    }
