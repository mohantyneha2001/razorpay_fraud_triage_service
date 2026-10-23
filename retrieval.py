
"""Part C: TF-IDF policy retrieval and evaluation."""

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_FILE = Path("fraud_policies.json")
QUERY_FILE = Path("retrieval_queries.json")
RESULT_FILE = Path("retrieval_results.json")

MIN_SCORE = 0.45


def load_json(path):
    """Load JSON data from a file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_index(policies):
    """Build a TF-IDF index from policy titles and text."""
    documents = [
        policy["title"] + " " + policy["text"]
        for policy in policies
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix



def expand_query(query):
    """Expand queries with relevant policy terminology."""
    text = query.lower()
    expanded = query

    # Night-time transaction far from the customer's home
    time_pattern = (
        r"\b("
        r"\d{1,2}\s*(?:am|a\.m\.)"
        r"|midnight|early morning|night|nighttime|night-time"
        r")\b"
    )

    location_terms = (
        "far away", "far from", "home city",
        "away from home", "distance", "distant",
        "km away", "kilometres away", "kilometers away",
        "usual location"
    )

    has_time = re.search(time_pattern, text) is not None
    has_location = any(term in text for term in location_terms)

    if has_time and has_location:
        expanded += (
            " night-time distance anomalies "
            "between midnight and 5 AM "
            "more than 100 km away "
            "customer home city "
            "transaction location customer care verify"
        )

    # Repeated UPI payments within a 24-hour period
    upi_terms = ("upi", "payment", "payments")
    repetition_terms = (
        "repeated", "repeat", "multiple", "many",
        "velocity", "within a day", "24 hours",
        "temporary hold"
    )

    has_upi_context = any(term in text for term in upi_terms)
    has_repetition_context = any(term in text for term in repetition_terms)

    if has_upi_context and has_repetition_context:
        expanded += (
            " UPI velocity limits more than 6 payments "
            "from one customer within 24 hours "
            "velocity alert temporary hold "
            "UPI risk team contacts customer"
        )

    return expanded


def retrieve(query, policies, vectorizer, matrix):
    """Retrieve the best matching policy if its score meets the threshold."""
    expanded_query = expand_query(query)
    query_vector = vectorizer.transform([expanded_query])

    scores = cosine_similarity(query_vector, matrix).ravel()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    if best_score < MIN_SCORE:
        return {
            "policy_id": None,
            "score": round(best_score, 4),
            "status": "no_policy_match"
        }

    return {
        "policy_id": policies[best_index]["policy_id"],
        "score": round(best_score, 4),
        "status": "policy_match"
    }


def main():
    policies = load_json(POLICY_FILE)
    queries = load_json(QUERY_FILE)

    if not policies:
        raise ValueError("fraud_policies.json contains no policies.")

    if not queries:
        raise ValueError("retrieval_queries.json contains no queries.")

    vectorizer, matrix = build_index(policies)

    results = []
    correct = 0

    for item in queries:
        result = retrieve(
            item["query"],
            policies,
            vectorizer,
            matrix
        )

        # Support either key name for expected policy.
        expected = item.get("expected", item.get("expected_policy"))
        actual = result["policy_id"]
        passed = actual == expected

        if passed:
            correct += 1

        row = {
            "query": item["query"],
            "expected": expected,
            "actual": actual,
            "score": result["score"],
            "status": result["status"],
            "passed": passed
        }

        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    total = len(queries)
    accuracy = correct / total if total else 0

    print("\nRetrieval evaluation")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")

    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"Results saved to {RESULT_FILE}")


if __name__ == "__main__":
    main()
