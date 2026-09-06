import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco Customer Churn dataset."""

    df = df.copy()

    # 1. Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # 2. Remove customer ID
    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])

    # 3. Clean text values
    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in categorical_columns:
        df[col] = df[col].str.strip().str.lower()

    # 4. Convert TotalCharges to numeric
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(
            df["totalcharges"],
            errors="coerce"
        )

    # 5. Handle missing TotalCharges
    if "totalcharges" in df.columns:
        df["totalcharges"] = df["totalcharges"].fillna(0)

    # 6. Encode target
    if "churn" in df.columns:
        df["churn"] = df["churn"].map({
            "no": 0,
            "yes": 1
        })

    # 7. Reset index
    df = df.reset_index(drop=True)

    return df