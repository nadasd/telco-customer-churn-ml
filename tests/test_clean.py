import pandas as pd

from src.clean import clean_data


def test_clean_data_transforms_raw_telco_data():
    raw = pd.DataFrame(
        {
            "customerID": ["C001", "C002"],
            "gender": [" Female ", "Male"],
            "tenure": [0, 12],
            "TotalCharges": [" ", "123.45"],
            "Churn": ["No", "Yes"],
        }
    )

    cleaned = clean_data(raw)

    # customerID must not be used as a model feature
    assert "customerid" not in cleaned.columns

    # Column names are normalized
    assert set(cleaned.columns) == {
        "gender",
        "tenure",
        "totalcharges",
        "churn",
    }

    # String values are stripped/lowercased
    assert cleaned["gender"].tolist() == ["female", "male"]

    # TotalCharges becomes numeric
    assert pd.api.types.is_numeric_dtype(cleaned["totalcharges"])

    # Blank TotalCharges -> 0
    assert cleaned.loc[0, "totalcharges"] == 0

    # Valid numeric strings are preserved numerically
    assert cleaned.loc[1, "totalcharges"] == 123.45

    # Churn is encoded as binary
    assert cleaned["churn"].tolist() == [0, 1]