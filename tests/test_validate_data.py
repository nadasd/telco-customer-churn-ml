import pandas as pd

from src.validate_data import validate_data


def make_valid_dataframe():
    return pd.DataFrame(
        {
            "gender": [
                "male",
                "female",
                "male",
                "female",
            ],
            "seniorcitizen": [
                0,
                1,
                0,
                1,
            ],
            "partner": [
                "yes",
                "no",
                "yes",
                "no",
            ],
            "dependents": [
                "no",
                "yes",
                "no",
                "yes",
            ],
            "tenure": [
                0,
                12,
                36,
                72,
            ],
            "phoneservice": [
                "yes",
                "yes",
                "no",
                "yes",
            ],
            "multiplelines": [
                "no",
                "yes",
                "no phone service",
                "no",
            ],
            "internetservice": [
                "dsl",
                "fiber optic",
                "no",
                "dsl",
            ],
            "onlinesecurity": [
                "yes",
                "no",
                "no internet service",
                "yes",
            ],
            "onlinebackup": [
                "no",
                "yes",
                "no internet service",
                "yes",
            ],
            "deviceprotection": [
                "yes",
                "no",
                "no internet service",
                "yes",
            ],
            "techsupport": [
                "no",
                "yes",
                "no internet service",
                "yes",
            ],
            "streamingtv": [
                "yes",
                "no",
                "no internet service",
                "yes",
            ],
            "streamingmovies": [
                "no",
                "yes",
                "no internet service",
                "yes",
            ],
            "contract": [
                "month-to-month",
                "one year",
                "two year",
                "month-to-month",
            ],
            "paperlessbilling": [
                "yes",
                "no",
                "yes",
                "no",
            ],
            "paymentmethod": [
                "electronic check",
                "mailed check",
                "bank transfer (automatic)",
                "credit card (automatic)",
            ],
            "monthlycharges": [
                20.0,
                50.0,
                75.0,
                100.0,
            ],
            "totalcharges": [
                0.0,
                600.0,
                2700.0,
                7200.0,
            ],
            "churn": [
                0,
                1,
                0,
                1,
            ],
        }
    )


def test_validate_data_accepts_valid_dataframe(
    tmp_path,
    monkeypatch,
):
    # Isolate Great Expectations file context
    # from the real project gx/ directory.
    monkeypatch.chdir(tmp_path)

    df = make_valid_dataframe()

    result = validate_data(df)

    assert result is True


def test_validate_data_rejects_invalid_category(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    df = make_valid_dataframe()

    # Invalid value according to the current
    # gender expectation.
    df.loc[0, "gender"] = "unknown"

    result = validate_data(df)

    assert result is False


def test_validate_data_rejects_out_of_range_tenure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    df = make_valid_dataframe()

    # Current contract allows tenure from 0 to 72.
    df.loc[0, "tenure"] = 100

    result = validate_data(df)

    assert result is False


def test_validate_data_rejects_null_numeric_value(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    df = make_valid_dataframe()

    df.loc[0, "monthlycharges"] = None

    result = validate_data(df)

    assert result is False


def test_validate_data_rejects_invalid_target(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    df = make_valid_dataframe()

    df.loc[0, "churn"] = 2

    result = validate_data(df)

    assert result is False