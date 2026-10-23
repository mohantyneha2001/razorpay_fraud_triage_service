# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview

The Razorpay Merchant Risk Desk Fraud Triage Service is a Python-based capstone project that identifies potentially fraudulent payment transactions and helps risk analysts decide which payments require manual review.

The project combines data cleaning, feature engineering, machine-learning-based fraud scoring, policy retrieval, structured case-note generation, and a FastAPI service. It uses synthetic and fictional transaction data; it does not use real Razorpay customer or payment records.

**Key components**
- **Part A — Data Foundation:** Audit, clean and engineer transaction data.
- **Part B — Fraud Model:** Compare imbalance-handling techniques, tune a Random Forest classifier and select an alert threshold.
- **Part C — Policy-Grounded Case Notes:** Retrieve relevant policies, validate structured case notes, retry failed outputs and apply safe defaults.
- **Part D — Triage API:** Expose fraud scoring and case-note generation through FastAPI with validation, guardrails, logging and metrics.

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- FastAPI, Pydantic and Uvicorn
- TF-IDF and cosine similarity for policy retrieval
- JSON for policy definitions and structured case notes
- pytest-free smoke testing using Python HTTP requests
- Git and GitHub for version control and submission

## Repository Structure

```text
Razorpay Fraud Triage/
├── generate_data.py
├── mock_llm.py
├── transactions_raw.csv
├── fraud_policies.json
├── retrieval_queries.json
├── case_scenarios.json
├── prepare.py
├── train.py
├── retrieval.py
├── casenote.py
├── app.py
├── smoke_test.py
├── requirements.txt
├── service.log
├── .gitignore
└── README.md
```

`generate_data.py` and `mock_llm.py` are the supplied project files and must remain unmodified. The mock language model operates offline, so no API key or paid external LLM service is required.

# Part A — Data Foundation: Audit, Clean and Engineer

## A.1 Data audit

The raw dataset contains 6,090 rows and 14 columns. The dataset includes transaction identifiers, merchant category, city, payment mode, transaction amount, transaction time, payment velocity, customer spending averages, distance from home, device information, account age, chargeback information and the fraud label.

| Audit measure | Result |
|---|---:|
| Raw rows | 6,090 |
| Raw columns | 14 |
| Exact duplicate rows | 90 |
| Missing `amount_inr` after deduplication | 76 |
| Missing `distance_from_home_km` after deduplication | 193 |
| Out-of-range amounts after deduplication | 5 |
| Distinct raw merchant-category spellings | 24 |
| Distinct raw city spellings | 24 |

## A.2 Data cleaning

The `clean_transactions(raw_df)` function returns a cleaned copy of the input DataFrame. It removes exact duplicate rows, standardises merchant-category and city names by stripping surrounding spaces and applying title case, removes rows with missing transaction amounts, excludes amounts below INR 1 or above INR 1,000,000, and resets the index.

| Cleaned-data measure | Result |
|---|---:|
| Rows after cleaning | 5,919 |
| Duplicate rows remaining | 0 |
| Missing `distance_from_home_km` values | 191 |
| Minimum transaction amount | INR 141.37 |
| Maximum transaction amount | INR 105,883.95 |

### Merchant-category distribution

| Merchant category | Transactions |
|---|---:|
| Grocery | 1,406 |
| Food Delivery | 1,326 |
| Fashion | 976 |
| Electronics | 845 |
| Utilities | 685 |
| Travel | 681 |
| **Total** | **5,919** |

### Payment-mode distribution

| Payment mode | Transactions |
|---|---:|
| UPI | 3,221 |
| Card | 1,450 |
| NetBanking | 731 |
| Wallet | 517 |
| **Total** | **5,919** |

Missing `distance_from_home_km` values are retained during Part A because the median must be learned from the training split only, preventing information from validation or test data from leaking into training.

## A.3 Feature engineering

The `add_features(df)` function creates three additional features:

- `amount_vs_avg_ratio`: transaction amount divided by the customer's average transaction amount over 30 days.
- `is_night`: 1 when `hour_of_day` is 5 or less; otherwise 0.
- `high_value`: 1 when the transaction amount exceeds INR 50,000; otherwise 0.

| Feature check | Result |
|---|---:|
| Sum of `is_night` | 365 |
| Sum of `high_value` | 17 |
| Maximum `amount_vs_avg_ratio` | 13.11 |

Both data-preparation functions are designed to work on a single-row DataFrame, allowing the same processing logic to be used by the API.

## A.4 Fraud prevalence and class imbalance

The cleaned dataset contains 366 fraudulent transactions out of 5,919 transactions, giving a fraud rate of approximately 6.18%.

| Measure | Result |
|---|---:|
| Total cleaned transactions | 5,919 |
| Fraudulent transactions | 366 |
| Fraud rate | 6.18% |
| Accuracy of an always-not-fraud rule | 93.82% |
| Recall of an always-not-fraud rule | 0.00% |

Accuracy alone is misleading for this dataset because most transactions are legitimate. A classifier that predicts every transaction as legitimate achieves high accuracy but fails to identify any fraudulent transaction, making fraud recall an essential evaluation metric.

## A.5 Leakage audit

The following table shows the relationship between chargebacks and fraud in the cleaned dataset.

| `chargeback_filed` | Total rows | Fraud rows |
|---|---:|---:|
| 1 | 373 | 304 |
| 0 | 5,546 | 62 |
| **Total** | **5,919** | **366** |

`chargeback_filed` is excluded from the production model because a chargeback is a post-transaction outcome that can reveal information unavailable when the original payment decision is made. The identifier columns `txn_id` and `customer_id` are also excluded because they identify records or customers rather than providing appropriate generalisable fraud signals. The target column `is_fraud` is excluded from the input features to prevent target leakage.

The model excludes `txn_id`, `customer_id`, `chargeback_filed` and `is_fraud`.

# Part B — Fraud Model, Imbalance and Alert Threshold

## B.1 Train, validation and test split

The cleaned dataset is divided using stratified splits with `random_state=42`. The split is performed in two stages to obtain approximately 60% training data, 20% validation data and 20% test data.

| Dataset split | Rows | Fraud rows |
|---|---:|---:|
| Training | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |
| **Total** | **5,919** | **366** |

Missing `distance_from_home_km` values are filled using the median calculated from the training split only.

**Training-set median distance:** 12.6 km.

## B.2 Model input features

The model uses 26 features before model fitting: ten numerical or engineered features and sixteen one-hot-encoded categorical features.

The ten non-one-hot features are:

1. `amount_inr`
2. `hour_of_day`
3. `txn_count_last_24h`
4. `avg_amount_30d_inr`
5. `distance_from_home_km`
6. `is_new_device`
7. `account_age_days`
8. `amount_vs_avg_ratio`
9. `is_night`
10. `high_value`

The categorical features are encoded for `merchant_category`, `city` and `payment_mode`. The resulting one-hot columns are:

- Merchant category: `merchant_category_Electronics`, `merchant_category_Fashion`, `merchant_category_Food Delivery`, `merchant_category_Grocery`, `merchant_category_Travel`, `merchant_category_Utilities`
- City: `city_Bengaluru`, `city_Chennai`, `city_Delhi`, `city_Kolkata`, `city_Mumbai`, `city_Pune`
- Payment mode: `payment_mode_Card`, `payment_mode_NetBanking`, `payment_mode_UPI`, `payment_mode_Wallet`

The final feature order is kept consistent between training, evaluation and API inference.

## B.3 Handling class imbalance

Three logistic-regression approaches are compared on the validation split at a probability threshold of 0.5: a plain model, a class-weighted model and a model trained using random oversampling of the fraud class.

| Model | Precision | Recall | F1 | Confusion matrix (TN, FP, FN, TP) |
|---|---:|---:|---:|---|
| Plain logistic regression | 0.625 | 0.342 | 0.442 | [FILL FROM OUTPUT] |
| Balanced logistic regression | 0.252 | 0.781 | 0.381 | [FILL FROM OUTPUT] |
| Oversampled logistic regression | 0.257 | 0.781 | 0.386 | [FILL FROM OUTPUT] |

The oversampled training dataset contains 6,662 rows, with 3,331 fraud rows and 3,331 non-fraud rows. Oversampling is performed on the training data only.

For a risk desk that prioritises catching fraud, the balanced and oversampled models offer substantially higher recall than the plain model. The trade-off is lower precision, which means more legitimate payments may be flagged and sent for manual review.

*Before submission, verify the reported metrics and confusion matrices against the actual output of `python train.py`.*

## B.4 Cross-validation and Random Forest tuning

Five-fold stratified cross-validation is performed on the training split using F1 as the scoring metric.

| Model | Mean CV F1 | Standard deviation |
|---|---:|---:|
| Balanced logistic-regression pipeline | [FILL FROM OUTPUT] | [FILL FROM OUTPUT] |
| Balanced Random Forest, 200 trees | [FILL FROM OUTPUT] | [FILL FROM OUTPUT] |
| Tuned Random Forest | [FILL FROM OUTPUT] | [FILL FROM OUTPUT] |

The Random Forest grid search evaluates 12 combinations of `max_depth` and `min_samples_leaf`, with `n_estimators=200`, `class_weight="balanced"` and `random_state=42` fixed.

**Best parameters:** [FILL FROM OUTPUT]

**Best mean CV F1:** [FILL FROM OUTPUT]

The tuned Random Forest is selected as the final model because it can capture nonlinear relationships between transaction amount, payment velocity, account age, distance and other fraud signals. Cross-validation helps assess both predictive performance and variation across folds. Compared with logistic regression, a Random Forest can be less straightforward to interpret and explain at the individual-feature level.

## B.5 Validation threshold selection

The alert threshold is selected using the validation split. A transaction is flagged when its predicted fraud probability is greater than or equal to the threshold.

The required rule is to choose the **highest threshold from 0.1 to 0.9 whose validation recall is at least 0.65**. If no threshold qualifies, use 0.1.

| Threshold | Precision | Recall | F1 | Transactions flagged |
|---:|---:|---:|---:|---:|
| 0.1 | [FILL] | [FILL] | [FILL] | [FILL] |
| 0.2 | [FILL] | [FILL] | [FILL] | [FILL] |
| 0.3 | [FILL] | [FILL] | [FILL] | [FILL] |
| 0.4 | [FILL] | [FILL] | [FILL] | [FILL] |
| 0.5 | [FILL] | [FILL] | [FILL] | [FILL] |
| 0.6 | [FILL] | [FILL] | [FILL] | [FILL] |
| 0.7 | [FILL] | [FILL] | [FILL] | [FILL] |
| 0.8 | [FILL] | [FILL] | [FILL] | [FILL] |
| 0.9 | [FILL] | [FILL] | [FILL] | [FILL] |

**Selected threshold (`t_star`):** [FILL FROM OUTPUT]

## B.6 Final test evaluation and risk-committee note

The final tuned Random Forest is evaluated on the test split once, using the threshold selected on the validation split.

| Test metric | Result |
|---|---:|
| Selected threshold | [FILL] |
| Precision | [FILL] |
| Recall | [FILL] |
| F1 | [FILL] |
| True negatives (TN) | [FILL] |
| False positives (FP) | [FILL] |
| False negatives (FN) | [FILL] |
| True positives (TP) | [FILL] |
| Total transactions flagged | [FILL] |
| Legitimate payments held per fraud caught (FP / TP) | [FILL] |

### Risk-committee note

The selected threshold is [FILL] because it is the highest tested validation threshold that achieves recall of at least 0.65. On the held-out test set, the model flagged [FILL] transactions and identified [FILL] fraudulent transactions. It missed [FILL] fraudulent transactions and generated [FILL] false positives, equivalent to approximately [FILL] legitimate payments held per fraud caught. A stricter threshold of [FILL] would change the precision-recall trade-off as shown by the validation results at thresholds [FILL] and [FILL]. The threshold should therefore be assessed in light of the risk desk's tolerance for missed fraud and the operational cost of manual review.

## B.7 Leakage experiment

A second Random Forest is trained with `chargeback_filed` added as a 27th feature. This experiment is performed only to demonstrate the effect of leakage and is not used in the production model.

| Validation evaluation at threshold 0.5 | F1 |
|---|---:|
| Clean tuned Random Forest | [FILL FROM OUTPUT] |
| Random Forest with `chargeback_filed` | [FILL FROM OUTPUT] |

The leaky model may appear to perform substantially better because chargeback information is closely associated with fraud labels. However, this information may only become available after the payment has been processed, so it cannot legitimately support the original transaction-time decision. The clean model is the appropriate model for deployment.

## B.8 Feature importance

The five most important features reported by the tuned Random Forest are:

1. `amount_vs_avg_ratio`
2. `txn_count_last_24h`
3. `amount_inr`
4. `account_age_days`
5. `distance_from_home_km`

*Verify the order against the output of `python train.py`.*

The ratio between the current amount and the customer's average spending can identify unusual spending behaviour. High transaction velocity can indicate rapid repeated payments, making both features useful signals for potential fraud.

# Part C — Policy-Grounded Case Notes with Validation

## C.1 Policy retrieval

The policy assistant uses TF-IDF vectorisation and cosine similarity to retrieve the most relevant policy from `fraud_policies.json`. The six policies cover high-value card payments, UPI velocity, night-time and distance anomalies, unauthorised payment handling, electronics merchant monitoring and low-risk automatic approval.

The `MIN_SCORE` threshold is set to **0.45**. A query below this threshold is treated as having no policy match and is not sent to the mock language model.

The retrieval test completed successfully with **10/10 queries passing**, including the in-scope policy queries and the out-of-scope refusal queries.

### Retrieval evaluation table

Populate the score column using the output of `python retrieval.py`.

| Query | Expected policy | Retrieved policy | Similarity score |
|---|---|---|---:|
| Q1 | [FROM OUTPUT] | [FROM OUTPUT] | [FILL] |
| Q2 | [FROM OUTPUT] | [FROM OUTPUT] | [FILL] |
| Q3 | [FROM OUTPUT] | [FROM OUTPUT] | [FILL] |
| Q4 | [FROM OUTPUT] | [FROM OUTPUT] | [FILL] |
| Q5 | [FROM OUTPUT] | [FROM OUTPUT] | [FILL] |
| Q6 | [FROM OUTPUT] | [FROM OUTPUT] | [FILL] |
| Q7 | [FROM OUTPUT] | [FROM OUTPUT] | [FILL] |
| Q8 | [FROM OUTPUT] | [FROM OUTPUT] | [FILL] |
| Q9 | No policy match | No policy match | [FILL] |
| Q10 | No policy match | No policy match | [FILL] |

**In-scope top-1 retrieval accuracy:** 8/8 expected policy matches.

**Out-of-scope refusal accuracy:** 2/2 queries correctly returned no policy match.

**Threshold calibration:** `MIN_SCORE = 0.45`. Record the highest measured out-of-scope score and lowest measured in-scope score from the actual retrieval output to document the separation provided by the threshold.

## C.2 Structured case-note generation

For flagged transactions, `casenote.py` retrieves a relevant policy and builds a prompt containing the transaction ID, fraud probability, amount, policy ID and policy text. The mock language model is asked to return a structured JSON case note.

The output is checked for valid JSON, required fields, permitted values and consistency with the retrieved policy. Invalid outputs are retried up to three attempts. If every attempt fails, the system returns a safe default that requires manual review. A connection failure is propagated to the API so it can return an appropriate error response.

### Case-note scenario results

The following table is from the completed `python casenote.py` run.

| Scenario | Transaction ID | Status | Attempts | Failed gates | Safe default | Final action | Override |
|---|---|---|---:|---|---|---|---|
| S1 | T900001 | ok | 1 | None | No | hold_for_review | No |
| S2 | T900002 | ok | 2 | 1 | No | hold_for_review | No |
| S3 | T900003 | ok | 2 | 2 | No | hold_for_review | No |
| S4 | T900004 | ok | 2 | 3 | No | block_and_notify | No |
| S5 | T900005 | ok | 3 | 1, 1, 1 | Yes | hold_for_review | No |
| S6 | T900006 | ok | 1 | None | No | hold_for_review | Yes |
| S7 | T900007 | ok | 2 | 3 | No | hold_for_review | No |
| S8 | T900008 | ok | 1 | None | No | block_and_notify | No |
| S9 | T900009 | ok | 3 | 1, 2 | No | hold_for_review | No |
| S10 | T900010 | ok | 2 | 3 | No | hold_for_review | No |

### Scenario totals

| Measure | Result |
|---|---:|
| Total mock LLM calls | 19 |
| First-call valid outputs | 3 |
| Recovered after an unsuccessful attempt | 6 |
| Safe defaults | 1 |
| Guardrail overrides | 1 |

The mock language model deliberately produces malformed JSON, missing fields, invalid values, incorrect policy IDs and unsafe recommendations for selected transactions. Retries allow the system to recover from temporary output-format or validation failures. Validation remains necessary because a response that looks plausible can still contain invalid fields, unsupported policy references or an unsafe recommendation.

When a transaction has a fraud probability of at least 0.5, an `approve` recommendation is overridden with `hold_for_review`. If repeated attempts fail, the system uses the safe default: *“Automated case note unavailable; manual review required.”*

# Part D — The Triage Service: Guardrails, Logging and Metrics

## D.1 API endpoints

The service is implemented using FastAPI and loads the trained fraud model, feature columns, training-set distance median and alert threshold at startup.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check whether the service is available |
| `/score` | POST | Validate and score a transaction |
| `/case` | POST | Score a transaction and generate a policy-grounded case note when appropriate |
| `/metrics` | GET | Return request counts, outcomes, human-review statistics and latency |

The API validates transaction fields such as amount, hour, merchant category and other required inputs. Invalid business inputs return HTTP 400, missing or incorrectly typed required fields are handled by FastAPI validation, and model-service connection failures return HTTP 503.

## D.2 Fraud decision and safety controls

The `/score` endpoint returns the transaction ID, fraud probability, flag status, decision, human-review requirement and any guardrail override.

The `/case` endpoint invokes the policy-grounded case-note pipeline for flagged transactions. It returns the generated case note and associated status information when a valid policy and case note are available. If no policy matches, the system does not call the language model and defaults to manual review for a flagged transaction.

The API also prevents a high-risk transaction from being approved solely because the generated case note recommends approval. These checks keep the automated model and language-model output subordinate to the service's safety rules.

## D.3 Smoke-test results

The smoke test checks the service's health, transaction scoring, case-note generation, validation failures and metrics endpoint.

**Latest reported smoke-test result: 11 passed, 0 failed.**

| Smoke-test area | Result |
|---|---|
| Health endpoint | PASS |
| Fraud scoring and response fields | PASS |
| Fraud probability range | PASS |
| Case endpoint and decision field | PASS |
| Negative transaction amount rejected | PASS |
| Invalid transaction hour rejected | PASS |
| Unknown merchant category rejected | PASS |
| Metrics endpoint and required fields | PASS |
| Overall reported result | **11 passed, 0 failed** |

Before submission, run the full smoke test from a freshly started server and use its final output to confirm the exact result. The capstone brief requires all eleven specified requests to be checked against their expected status codes and response fields.

## D.4 Service metrics

The `/metrics` endpoint exposes operational information including total requests, successful requests, errors, success and error rates, human-review counts and average latency.

Paste the actual JSON returned by `GET /metrics` after the final smoke-test run below.

```json
{
  "requests_total": "[FILL FROM OUTPUT]",
  "success_total": "[FILL FROM OUTPUT]",
  "error_total": "[FILL FROM OUTPUT]",
  "success_rate": "[FILL FROM OUTPUT]",
  "error_rate": "[FILL FROM OUTPUT]",
  "human_review_total": "[FILL FROM OUTPUT]",
  "human_review_rate": "[FILL FROM OUTPUT]",
  "avg_latency_ms": "[FILL FROM OUTPUT]"
}
```

## D.5 Logging and error handling

The service writes structured log lines containing the endpoint, response status and request latency. Informational logs record normal request processing, warnings identify validation or safety-related events, and error logs record service failures such as an unavailable mock language model.

Paste three representative lines from the final `service.log` here: one `INFO`, one `WARNING` and one `ERROR` line. Ensure the committed log contains only the intended final smoke-test run, as required by the capstone brief.

```text
[FILL WITH ONE INFO LINE]
[FILL WITH ONE WARNING LINE]
[FILL WITH ONE ERROR LINE]
```

# Reproduce

Run all commands from the repository root.

## 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Generate the synthetic data

```powershell
python generate_data.py
```

This creates the raw transaction data and the policy, retrieval-query and case-scenario JSON files.

## 4. Run data preparation and model evaluation

```powershell
python prepare.py
python train.py
```

Use the printed output to verify the data audit, feature counts, model comparison, cross-validation, threshold selection, test metrics, leakage experiment and feature importance.

## 5. Run policy retrieval and case-note scenarios

```powershell
python retrieval.py
python casenote.py
```

The retrieval script reports policy matches and similarity scores. The case-note script runs the ten scenarios, reports retry and validation behaviour, and prints the final totals.

## 6. Start the API

In one terminal:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Wait for the service to start before running the smoke test.

## 7. Run the smoke test

In a second terminal, activate the same virtual environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
python smoke_test.py
```

The test exercises the running API and prints the observed results and metrics. Restart the service before each full smoke run if required by the mock model's stateful call behaviour. Follow the capstone brief's instructions for clearing `service.log` before the final run.

# Limitations

- The transaction dataset is synthetic and does not represent actual Razorpay payments, customers, fraud patterns or internal policies.
- Model results depend on the generated data and the selected train, validation and test splits; real-world fraud performance may differ.
- Fraud probabilities are model estimates, not guarantees that a transaction is fraudulent or legitimate.
- Selecting a threshold to achieve high recall increases the likelihood that legitimate transactions will be flagged for review.
- TF-IDF retrieval relies on lexical similarity and may fail to identify the correct policy when a query uses unfamiliar terminology or lacks sufficient context.
- The supplied mock language model is an offline testing utility with deliberately injected failures. It does not measure the quality or reliability of a production LLM.
- The service is a capstone prototype, not a production payment-risk system. Production deployment would require additional security, monitoring, access control, auditability, model validation, performance testing and integration with authorised payment systems.
- Automated decisions should be monitored and supported by appropriate human review, particularly when the model is uncertain or a policy match cannot be established.

# Conclusion

The Fraud Triage Service combines a reproducible data-preparation pipeline, an imbalance-aware fraud model, policy-grounded case-note generation and a validated API. Its retry handling, safe defaults and fraud-decision guardrails demonstrate how model predictions and generated text can be incorporated into a controlled decision-support workflow. The project provides a foundation for further testing and improvement while keeping human review central to higher-risk cases.
