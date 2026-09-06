"""Integration contract for a logged Telco churn MLflow artifact."""

from __future__ import annotations

import os

import mlflow
import pandas as pd
import pytest

from src.model_artifact import (
    DECISION_COLUMN,
    PROBABILITY_COLUMN,
    build_raw_input_example,
)


def test_logged_artifact_preserves_raw_prediction_contract() -> None:
    model_uri = os.getenv("TELCO_MODEL_URI")
    if not model_uri:
        pytest.skip("TELCO_MODEL_URI must identify the artifact under test")

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    data_path = os.getenv(
        "TELCO_TEST_DATA_PATH",
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )
    mlflow.set_tracking_uri(tracking_uri)

    raw_df = pd.read_csv(data_path)
    raw_input = build_raw_input_example(raw_df)
    assert raw_input.shape == (1, 19)

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    metadata = loaded_model.metadata.metadata
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
    threshold = float(metadata["decision_threshold"])
    reference = metadata["reference_output"]

    assert 0.0 <= probability <= 1.0
    assert decision == int(probability >= threshold)
    assert probability == pytest.approx(
        float(reference[PROBABILITY_COLUMN]),
        rel=1e-12,
        abs=1e-12,
    )
    assert decision == int(reference[DECISION_COLUMN])

    schema = metadata["raw_feature_schema"]
    assert [field["name"] for field in schema] == raw_input.columns.tolist()
