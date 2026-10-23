
"""Part B: Reproducible fraud-model training and evaluation."""

import json

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from prepare import clean_transactions, add_features


SEED = 42

# Exclude identifiers, the target, and post-event information
# that could cause data leakage.
EXCLUDED = [
    "txn_id",
    "customer_id",
    "chargeback_filed",
    "is_fraud",
]


def load_encoded():
    """Load, clean, engineer features, and encode the dataset."""
    raw = pd.read_csv("transactions_raw.csv")

    df = add_features(clean_transactions(raw))

    y = df["is_fraud"].astype(int)

    X = df.drop(columns=EXCLUDED)
    X = pd.get_dummies(X, dtype=int)

    return df, X, y


def split_data(X, y):
    """Create stratified 60/20/20 train/validation/test splits."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_rest,
        y_rest,
        test_size=0.5,
        stratify=y_rest,
        random_state=SEED,
    )

    # Fit imputation using training data only.
    median = float(X_train["distance_from_home_km"].median())

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    for part in (X_train, X_val, X_test):
        part["distance_from_home_km"] = (
            part["distance_from_home_km"].fillna(median)
        )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        median,
    )


def metrics_at(y_true, probabilities, threshold=0.5):
    """Calculate classification metrics at a selected threshold."""
    predictions = (
        np.asarray(probabilities) >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "flagged": int(predictions.sum()),
    }


def _model_data():
    """Load the data and return the prepared splits."""
    df, X, y = load_encoded()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        median,
    ) = split_data(X, y)

    return (
        df,
        X,
        y,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        median,
    )


def build_final_model():
    """
    Train the tuned Random Forest for use by the API.
    Select the highest threshold that achieves recall >= 0.65.
    """
    (
        _,
        X,
        y,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        median,
    ) = _model_data()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    grid = GridSearchCV(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=SEED,
        ),
        {
            "max_depth": [4, 6, 8, None],
            "min_samples_leaf": [5, 10, 20],
        },
        scoring="f1",
        cv=cv,
        n_jobs=2,
        refit=True,
    )

    grid.fit(X_train, y_train)

    validation_probabilities = (
        grid.best_estimator_.predict_proba(X_val)[:, 1]
    )

    threshold_table = [
        (
            round(float(threshold), 1),
            metrics_at(
                y_val,
                validation_probabilities,
                float(threshold),
            ),
        )
        for threshold in np.arange(0.1, 1.0, 0.1)
    ]

    eligible = [
        threshold
        for threshold, result in threshold_table
        if result["recall"] >= 0.65
    ]

    threshold = max(eligible) if eligible else 0.1

    return {
        "model": grid.best_estimator_,
        "columns": list(X.columns),
        "median": median,
        "threshold": float(round(threshold, 1)),
    }


def main():
    """Run the complete Part B evaluation."""
    (
        df,
        X,
        y,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        median,
    ) = _model_data()

    print("=" * 65)
    print("PART B: FRAUD MODEL TRAINING AND EVALUATION")
    print("=" * 65)

    print("\n1. DATA SPLITS")
    print(
        f"Training:   {len(X_train)} rows; "
        f"{int(y_train.sum())} fraud cases"
    )
    print(
        f"Validation: {len(X_val)} rows; "
        f"{int(y_val.sum())} fraud cases"
    )
    print(
        f"Test:       {len(X_test)} rows; "
        f"{int(y_test.sum())} fraud cases"
    )
    print("Training-set distance median:", median)

    print("\nModel feature columns:")
    print(json.dumps(list(X.columns), indent=2))

    # ---------------------------------------------------------
    # 2. Baseline and imbalance-handling comparisons
    # ---------------------------------------------------------
    plain = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    balanced = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    # Oversample the minority class in training data only.
    X_fraud = X_train.loc[y_train == 1]
    X_legitimate = X_train.loc[y_train == 0]

    y_fraud = y_train.loc[y_train == 1]
    y_legitimate = y_train.loc[y_train == 0]

    X_fraud_oversampled = X_fraud.sample(
        n=len(X_legitimate),
        replace=True,
        random_state=SEED,
    )

    y_fraud_oversampled = y_fraud.loc[
        X_fraud_oversampled.index
    ]

    X_oversampled = pd.concat(
        [X_legitimate, X_fraud_oversampled]
    )

    y_oversampled = pd.concat(
        [y_legitimate, y_fraud_oversampled]
    )

    # Reset indices to keep the oversampled feature and label
    # rows aligned, including duplicated oversampled examples.
    X_oversampled = X_oversampled.reset_index(drop=True)
    y_oversampled = y_oversampled.reset_index(drop=True)

    oversampled_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    models = [
        ("Plain logistic regression", plain, X_train, y_train),
        (
            "Class-weighted logistic regression",
            balanced,
            X_train,
            y_train,
        ),
        (
            "Random oversampling + logistic regression",
            oversampled_model,
            X_oversampled,
            y_oversampled,
        ),
    ]

    print("\n2. VALIDATION METRICS AT THRESHOLD 0.5")

    for name, model, features, labels in models:
        model.fit(features, labels)

        probabilities = model.predict_proba(X_val)[:, 1]
        result = metrics_at(y_val, probabilities, 0.5)

        print(f"\n{name}")
        for key, value in result.items():
            if isinstance(value, (float, np.floating)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print(
        "\nOversampled training rows:",
        len(X_oversampled),
    )
    print(
        "Oversampled fraud rows:",
        int(y_oversampled.sum()),
    )

    # ---------------------------------------------------------
    # 3. Five-fold cross-validation
    # ---------------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    print("\n3. FIVE-FOLD CROSS-VALIDATION")

    logistic_cv = cross_val_score(
        balanced,
        X_train,
        y_train,
        scoring="f1",
        cv=cv,
        n_jobs=2,
    )

    forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED,
    )

    forest_cv = cross_val_score(
        forest,
        X_train,
        y_train,
        scoring="f1",
        cv=cv,
        n_jobs=2,
    )

    print(
        "Class-weighted logistic regression: "
        f"mean F1={logistic_cv.mean():.4f}, "
        f"std={logistic_cv.std():.4f}"
    )

    print(
        "Untuned Random Forest: "
        f"mean F1={forest_cv.mean():.4f}, "
        f"std={forest_cv.std():.4f}"
    )

    # ---------------------------------------------------------
    # 4. Random Forest hyperparameter tuning
    # ---------------------------------------------------------
    print("\n4. RANDOM FOREST GRID SEARCH")

    grid = GridSearchCV(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=SEED,
        ),
        {
            "max_depth": [4, 6, 8, None],
            "min_samples_leaf": [5, 10, 20],
        },
        scoring="f1",
        cv=cv,
        n_jobs=2,
        refit=True,
    )

    grid.fit(X_train, y_train)

    print(
        "Number of parameter combinations:",
        len(grid.cv_results_["params"]),
    )
    print("Best parameters:", grid.best_params_)
    print(f"Best mean CV F1: {grid.best_score_:.4f}")

    # ---------------------------------------------------------
    # 5. Threshold selection using validation data
    # ---------------------------------------------------------
    print("\n5. VALIDATION THRESHOLD ANALYSIS")

    validation_probabilities = (
        grid.best_estimator_.predict_proba(X_val)[:, 1]
    )

    threshold_rows = []

    for threshold in np.arange(0.1, 1.0, 0.1):
        threshold = round(float(threshold), 1)

        result = metrics_at(
            y_val,
            validation_probabilities,
            threshold,
        )

        threshold_rows.append((threshold, result))

        print(
            f"Threshold={threshold:.1f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"Flagged={result['flagged']}"
        )

    # Select the highest threshold meeting the recall target.
    eligible = [
        threshold
        for threshold, result in threshold_rows
        if result["recall"] >= 0.65
    ]

    selected_threshold = max(eligible) if eligible else 0.1

    print(
        "\nSelected threshold (recall >= 0.65):",
        selected_threshold,
    )

    # ---------------------------------------------------------
    # 6. Final test evaluation
    # ---------------------------------------------------------
    print("\n6. HELD-OUT TEST RESULTS")

    test_probabilities = (
        grid.best_estimator_.predict_proba(X_test)[:, 1]
    )

    test_result = metrics_at(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    for key, value in test_result.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if test_result["TP"] > 0:
        false_positives_per_caught_fraud = (
            test_result["FP"] / test_result["TP"]
        )
    else:
        false_positives_per_caught_fraud = float("inf")

    print(
        "Legitimate payments flagged per fraud caught:",
        f"{false_positives_per_caught_fraud:.3f}",
    )
    print("Missed fraud cases:", test_result["FN"])

    # ---------------------------------------------------------
    # 7. Leakage experiment
    # ---------------------------------------------------------
    print("\n7. DATA LEAKAGE EXPERIMENT")

    leaky_X = X.copy()
    leaky_X["chargeback_filed"] = (
        df["chargeback_filed"].astype(int).to_numpy()
    )

    X_train_leaky = leaky_X.loc[X_train.index]
    X_val_leaky = leaky_X.loc[X_val.index]

    leaky_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED,
        **grid.best_params_,
    )

    leaky_model.fit(X_train_leaky, y_train)

    clean_validation_f1 = metrics_at(
        y_val,
        validation_probabilities,
        0.5,
    )["f1"]

    leaky_validation_probabilities = (
        leaky_model.predict_proba(X_val_leaky)[:, 1]
    )

    leaky_validation_f1 = metrics_at(
        y_val,
        leaky_validation_probabilities,
        0.5,
    )["f1"]

    print(
        "Clean validation F1:",
        f"{clean_validation_f1:.4f}",
    )
    print(
        "Leaky validation F1:",
        f"{leaky_validation_f1:.4f}",
    )
    print(
        "Interpretation: post-event chargeback information "
        "can artificially inflate model performance and must "
        "not be used as a production prediction feature."
    )

    # ---------------------------------------------------------
    # 8. Feature importance
    # ---------------------------------------------------------
    print("\n8. TOP FIVE FEATURE IMPORTANCES")

    importances = pd.Series(
        grid.best_estimator_.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    print(importances.head(5).to_string())

    print("\nTotal model features:", len(X.columns))
    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    main()
