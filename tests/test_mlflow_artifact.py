"""Integration contract for a logged Telco churn MLflow artifact."""

from __future__ import annotations

import os

import mlflow
import pandas as pd
import pytest

from src.model_artifact import (
    DECISION_COLUMN,
    PROBABILITY_COLUMN,
)


SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_logged_artifact_preserves_raw_prediction_contract() -> None:
    model_uri = os.getenv("TELCO_MODEL_URI")

    if not model_uri:
        pytest.skip(
            "TELCO_MODEL_URI must identify the artifact under test"
        )

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    metadata = loaded_model.metadata.metadata

    schema = metadata["raw_feature_schema"]
    feature_names = [
        field["name"]
        for field in schema
    ]

    raw_input = pd.DataFrame(
        [SAMPLE_CUSTOMER]
    )[feature_names]

    assert raw_input.shape == (1, 19)

    prediction_after_reload = loaded_model.predict(raw_input)

    assert list(prediction_after_reload.columns) == [
        PROBABILITY_COLUMN,
        DECISION_COLUMN,
    ]

    probability = float(
        prediction_after_reload.iloc[0][PROBABILITY_COLUMN]
    )
    decision = int(
        prediction_after_reload.iloc[0][DECISION_COLUMN]
    )

    threshold = float(
        metadata["decision_threshold"]
    )

    assert 0.0 <= probability <= 1.0
    assert decision in (0, 1)
    assert decision == int(
        probability >= threshold
    )

    assert feature_names == raw_input.columns.tolist()