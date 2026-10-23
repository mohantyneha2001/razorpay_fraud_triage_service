
"""Automated smoke tests for the Razorpay Fraud Triage API."""

import json
import sys
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

passed = 0
failed = 0


def request(method, path, payload=None):
    """Send an HTTP request and return status code and response body."""
    url = BASE_URL + path
    data = None

    headers = {"Content-Type": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}

    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw_response": body}
        return error.code, parsed

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Cannot connect to the API. Make sure Uvicorn is running."
        ) from error


def check(name, condition, details=""):
    """Record and print a test result."""
    global passed, failed

    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")
        if details:
            print(f"      Details: {details}")


def sample_transaction(txn_id="SMOKE001"):
    return {
        "txn_id": txn_id,
        "merchant_category": "Electronics",
        "city": "Bengaluru",
        "payment_mode": "Card",
        "amount_inr": 95000,
        "hour_of_day": 23,
        "txn_count_last_24h": 5,
        "avg_amount_30d_inr": 12000,
        "distance_from_home_km": 850,
        "is_new_device": 1,
        "account_age_days": 3
    }


def main():
    global passed, failed

    print("=" * 55)
    print("RAZORPAY FRAUD TRIAGE — SMOKE TESTS")
    print("=" * 55)

    # Test 1: Health endpoint
    try:
        status, body = request("GET", "/health")
        check(
            "Health endpoint",
            status == 200 and body.get("status") == "ok",
            f"HTTP {status}: {body}"
        )
    except RuntimeError as error:
        print(f"ERROR: {error}")
        print("Start the API with: uvicorn app:app --reload")
        sys.exit(1)

    # Test 2: Fraud scoring endpoint
    status, body = request(
        "POST", "/score", sample_transaction("SMOKE002")
    )
    check(
        "Fraud scoring returns HTTP 200",
        status == 200,
        f"HTTP {status}: {body}"
    )

    required_score_fields = {
        "txn_id", "fraud_probability", "flagged",
        "decision", "human_review", "guardrail_override"
    }
    check(
        "Fraud scoring response has required fields",
        required_score_fields.issubset(body.keys()),
        f"Received fields: {list(body.keys())}"
    )

    if "fraud_probability" in body:
        probability = body["fraud_probability"]
        check(
            "Fraud probability is between 0 and 1",
            isinstance(probability, (int, float))
            and 0 <= probability <= 1,
            f"Probability: {probability}"
        )

    # Test 3: Case-note endpoint
    case_payload = sample_transaction("SMOKE003")
    case_payload["question"] = (
        "Should this high-value transaction from a new device "
        "be reviewed for fraud?"
    )

    status, body = request("POST", "/case", case_payload)
    check(
        "Case endpoint responds successfully",
        status == 200,
        f"HTTP {status}: {body}"
    )

    check(
        "Case response contains a decision",
        "decision" in body,
        f"Response: {body}"
    )

    # Test 4: Invalid amount should be rejected
    invalid_payload = sample_transaction("SMOKE004")
    invalid_payload["amount_inr"] = -100

    status, body = request("POST", "/score", invalid_payload)
    check(
        "Negative transaction amount is rejected",
        status == 400,
        f"HTTP {status}: {body}"
    )

    # Test 5: Invalid hour should be rejected
    invalid_hour_payload = sample_transaction("SMOKE005")
    invalid_hour_payload["hour_of_day"] = 25

    status, body = request(
        "POST", "/score", invalid_hour_payload
    )
    check(
        "Invalid transaction hour is rejected",
        status == 400,
        f"HTTP {status}: {body}"
    )

    # Test 6: Unknown merchant category should be rejected
    invalid_category_payload = sample_transaction("SMOKE006")
    invalid_category_payload["merchant_category"] = "UnknownCategory"

    status, body = request(
        "POST", "/score", invalid_category_payload
    )
    check(
        "Unknown merchant category is rejected",
        status == 400,
        f"HTTP {status}: {body}"
    )

    # Test 7: Metrics endpoint
    status, body = request("GET", "/metrics")
    check(
        "Metrics endpoint responds successfully",
        status == 200,
        f"HTTP {status}: {body}"
    )

    required_metrics = {
        "requests_total", "success_total", "error_total",
        "success_rate", "error_rate", "avg_latency_ms",
        "human_review_total"
    }
    check(
        "Metrics response contains required fields",
        required_metrics.issubset(body.keys()),
        f"Received fields: {list(body.keys())}"
    )

    # Final report
    print("\n" + "=" * 55)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 55)

    if failed:
        print("Some tests failed. Review the messages above.")
        sys.exit(1)

    print("All smoke tests passed!")


if __name__ == "__main__":
    main()
