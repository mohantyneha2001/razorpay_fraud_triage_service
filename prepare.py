from pathlib import Path
import pandas as pd

RAW_PATH = Path("transactions_raw.csv")


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean a copy of the raw transaction data."""
    df = raw_df.copy(deep=True)

    # Remove exact duplicate records
    df = df.drop_duplicates()

    # Standardise text fields
    for col in ("merchant_category", "city"):
        df[col] = df[col].astype("string").str.strip().str.title()

    # Validate transaction amounts
    df["amount_inr"] = pd.to_numeric(
        df["amount_inr"], errors="coerce"
    )
    df = df.dropna(subset=["amount_inr"])
    df = df.loc[
        df["amount_inr"].between(1, 1_000_000, inclusive="both")
    ]

    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived transaction risk features."""
    out = df.copy(deep=True)

    out["amount_inr"] = pd.to_numeric(
        out["amount_inr"], errors="coerce"
    )
    out["avg_amount_30d_inr"] = pd.to_numeric(
        out["avg_amount_30d_inr"], errors="coerce"
    )

    out["amount_vs_avg_ratio"] = (
        out["amount_inr"] / out["avg_amount_30d_inr"]
    )
    out["is_night"] = (
        pd.to_numeric(out["hour_of_day"]) <= 5
    ).astype(int)
    out["high_value"] = (
        out["amount_inr"] > 50_000
    ).astype(int)

    return out


def main():
    raw = pd.read_csv(RAW_PATH)
    clean = clean_transactions(raw)
    featured = add_features(clean)

    print("PART A: DATA AUDIT")
    print("Raw shape:", raw.shape)
    print("Duplicate rows:", raw.duplicated().sum())
    print("Clean shape:", clean.shape)
    print("Fraud rows:", int(clean["is_fraud"].sum()))
    print(
        "Fraud rate:",
        round(clean["is_fraud"].mean() * 100, 2),
        "%",
    )
    print("Missing distance values:",
          clean["distance_from_home_km"].isna().sum())
    print("Feature columns added:")
    print(" - amount_vs_avg_ratio")
    print(" - is_night")
    print(" - high_value")

    print("\nCategory counts:")
    print(clean["merchant_category"].value_counts())

    print("\nPayment mode counts:")
    print(clean["payment_mode"].value_counts())

    print("\nFeature preview:")
    print(featured.head())


if __name__ == "__main__":
    main()