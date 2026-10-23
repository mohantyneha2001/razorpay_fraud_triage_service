
"""Part C: Policy-grounded case-note generation and validation."""

import json
from pathlib import Path

from retrieval import (
    MIN_SCORE,
    build_index,
    load_json,
    retrieve,
)
from mock_llm import generate, reset


MAX_ATTEMPTS = 3

ALLOWED_PRIORITIES = {"low", "medium", "high"}

ALLOWED_TEAMS = {
    "card_risk",
    "upi_risk",
    "merchant_ops",
    "customer_care",
}

ALLOWED_ACTIONS = {
    "approve",
    "hold_for_review",
    "block_and_notify",
}

ALLOWED_POLICIES = {"P1", "P2", "P3", "P4", "P5", "P6"}

REQUIRED_FIELDS = [
    "summary",
    "priority",
    "recommended_team",
    "recommended_action",
    "policy_id",
]


def build_prompt(
    txn_id,
    fraud_probability,
    amount_inr,
    policy_id,
    policy_text,
):
    """Build a prompt with the fields required by mock_llm.py."""
    return f"""ROLE:
You are a careful merchant-risk case-note assistant.

TASK:
Write a concise case note for the transaction using the supplied policy.

CONSTRAINTS:
Use only the supplied policy as policy evidence.
Do not invent policy rules or transaction facts.
Return one valid JSON object and nothing else.

OUTPUT FORMAT:
Return exactly these five string keys:
- summary
- priority: low, medium, or high
- recommended_team: card_risk, upi_risk, merchant_ops, or customer_care
- recommended_action: approve, hold_for_review, or block_and_notify
- policy_id: P1, P2, P3, P4, P5, or P6

POLICY TEXT:
{policy_text}

TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}
"""


def validate(response_text, retrieved_policy_id):
    """
    Validation gates:
    1. Valid JSON object.
    2. Required fields and string types.
    3. Valid enum values and matching policy ID.

    Returns (note, gate).
    Gate 0 means valid; gates 1-3 indicate failure.
    """
    try:
        note = json.loads(response_text)

        if not isinstance(note, dict):
            return None, 1

    except (json.JSONDecodeError, TypeError):
        return None, 1

    if any(
        field not in note or not isinstance(note[field], str)
        for field in REQUIRED_FIELDS
    ):
        return note, 2

    if not note["summary"].strip():
        return note, 2

    if note["priority"] not in ALLOWED_PRIORITIES:
        return note, 3

    if note["recommended_team"] not in ALLOWED_TEAMS:
        return note, 3

    if note["recommended_action"] not in ALLOWED_ACTIONS:
        return note, 3

    if note["policy_id"] not in ALLOWED_POLICIES:
        return note, 3

    if note["policy_id"] != retrieved_policy_id:
        return note, 3

    return note, 0


def run_case(
    txn_id,
    fraud_probability,
    amount_inr,
    question,
    policies,
    vectorizer,
    policy_matrix,
    max_attempts=MAX_ATTEMPTS,
):
    """Retrieve policy, generate a case note, validate, and retry."""
    retrieval_result = retrieve(
        question,
        policies,
        vectorizer,
        policy_matrix,
    )

    policy_id = retrieval_result["policy_id"]
    score = retrieval_result["score"]

    if (
        retrieval_result["status"] != "policy_match"
        or score < MIN_SCORE
        or policy_id is None
    ):
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False,
        }

    policy = next(
        (p for p in policies if p["policy_id"] == policy_id),
        None,
    )

    if policy is None:
        return {
            "status": "no_policy_match",
            "attempts": 0,
            "failures": [],
            "case_note": None,
            "safe_default": False,
            "guardrail_override": False,
        }

    prompt = build_prompt(
        txn_id,
        fraud_probability,
        amount_inr,
        policy_id,
        policy["text"],
    )

    failures = []
    note = None
    attempts = 0

    for _ in range(max_attempts):
        attempts += 1

        # A ConnectionError is intentionally allowed to reach
        # the API layer, which should return a controlled error.
        response = generate(prompt)

        note, gate = validate(response, policy_id)

        if gate == 0:
            break

        failures.append(gate)
        note = None

    safe_default = note is None

    if safe_default:
        note = {
            "summary": (
                "Automated case note unavailable; "
                "manual review is required."
            ),
            "priority": "high",
            "recommended_team": "customer_care",
            "recommended_action": "hold_for_review",
            "policy_id": policy_id,
        }

    # Safety guardrail: a high-risk transaction must not be
    # approved by the generated case note.
    override = False

    if (
        note["recommended_action"] == "approve"
        and fraud_probability >= 0.5
    ):
        note["recommended_action"] = "hold_for_review"
        override = True

    return {
        "status": "ok",
        "attempts": attempts,
        "failures": failures,
        "case_note": note,
        "safe_default": safe_default,
        "guardrail_override": override,
    }


def main():
    """Run the supplied case scenarios and report validation results."""
    reset()

    policies = load_json(Path("fraud_policies.json"))
    scenarios = load_json(Path("case_scenarios.json"))
    vectorizer, policy_matrix = build_index(policies)

    totals = {
        "calls": 0,
        "first_call_valid": 0,
        "recovered": 0,
        "safe_default": 0,
        "overrides": 0,
    }

    print("CASE-NOTE VALIDATION RESULTS")
    print("-" * 90)

    for index, scenario in enumerate(scenarios, start=1):
        result = run_case(
            txn_id=scenario["txn_id"],
            fraud_probability=scenario["fraud_probability"],
            amount_inr=scenario["amount_inr"],
            question=scenario["question"],
            policies=policies,
            vectorizer=vectorizer,
            policy_matrix=policy_matrix,
        )

        totals["calls"] += result["attempts"]

        if not result["failures"] and result["status"] == "ok":
            totals["first_call_valid"] += 1
        elif result["failures"] and not result["safe_default"]:
            totals["recovered"] += 1

        totals["safe_default"] += int(result["safe_default"])
        totals["overrides"] += int(result["guardrail_override"])

        note = result["case_note"]
        action = (
            note["recommended_action"] if note else "none"
        )

        print(
            f"S{index} | {scenario['txn_id']} | "
            f"status={result['status']} | "
            f"attempts={result['attempts']} | "
            f"failures={result['failures']} | "
            f"safe_default={result['safe_default']} | "
            f"action={action} | "
            f"override={result['guardrail_override']}"
        )

    print("\nTOTALS")
    for key, value in totals.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
