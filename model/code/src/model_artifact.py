"""Build the self-contained MLflow inference artifact."""

from __future__ import annotations

import platform
from importlib.metadata import version
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.clean import clean_data


ARTIFACT_VERSION = "1.0.0"
REGISTERED_MODEL_NAME = "TelcoChurnXGBoost"
RANDOM_SEED = 42
PROBABILITY_COLUMN = "churn_probability"
DECISION_COLUMN = "churn_prediction"


def build_raw_input_example(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return one API-shaped row with the 19 original feature names."""

    feature_frame = raw_df.drop(
        columns=["customerID", "Churn"],
        errors="raise"
    ).iloc[[0]].copy()

    # The API contract exposes TotalCharges as a number. Blank values remain
    # governed by the existing clean_data implementation used by the model.
    feature_frame["TotalCharges"] = pd.to_numeric(
        feature_frame["TotalCharges"],
        errors="coerce"
    ).fillna(0.0)

    return feature_frame.reset_index(drop=True)


def build_raw_feature_schema(
    input_example: pd.DataFrame
) -> list[dict[str, str]]:
    """Capture the ordered raw feature contract in JSON-safe form."""

    return [
        {
            "name": column,
            "pandas_dtype": str(input_example[column].dtype),
        }
        for column in input_example.columns
    ]


def build_serving_pipeline(
    preprocessor,
    classifier,
    raw_input_example: pd.DataFrame,
) -> Pipeline:
    """Combine raw-input cleaning, the fitted preprocessor and classifier."""

    cleaner = FunctionTransformer(
        clean_data,
        validate=False,
    )
    cleaner.fit(raw_input_example)

    return Pipeline([
        ("cleaner", cleaner),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


class TelcoChurnArtifact(mlflow.pyfunc.PythonModel):
    """MLflow model returning churn probabilities and threshold decisions."""

    def __init__(
        self,
        pipeline: Pipeline,
        threshold: float,
        raw_feature_schema: list[dict[str, str]],
        metadata: dict[str, Any],
    ) -> None:
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.raw_feature_schema = raw_feature_schema
        self.artifact_metadata = metadata

    @property
    def raw_feature_names(self) -> list[str]:
        return [field["name"] for field in self.raw_feature_schema]

    def _validate_and_order(self, model_input: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        expected = self.raw_feature_names
        missing = [name for name in expected if name not in model_input.columns]
        unexpected = [
            name for name in model_input.columns if name not in expected
        ]

        if missing or unexpected:
            raise ValueError(
                "Raw feature schema mismatch. "
                f"Missing columns: {missing}. "
                f"Unexpected columns: {unexpected}."
            )

        return model_input.loc[:, expected].copy()

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        del context, params

        ordered_input = self._validate_and_order(model_input)
        probabilities = self.pipeline.predict_proba(ordered_input)[:, 1]
        decisions = (probabilities >= self.threshold).astype("int64")

        return pd.DataFrame({
            PROBABILITY_COLUMN: probabilities.astype("float64"),
            DECISION_COLUMN: decisions,
        })


def build_artifact_metadata(
    raw_feature_schema: list[dict[str, str]],
    threshold: float,
    model_parameters: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Create metadata embedded both in MLmodel and the Python model."""

    return {
        "artifact_version": ARTIFACT_VERSION,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "decision_threshold": float(threshold),
        "raw_feature_schema": raw_feature_schema,
        "model_parameters": {
            key: _to_builtin(value)
            for key, value in model_parameters.items()
        },
        "test_metrics": {
            key: _to_builtin(value)
            for key, value in test_metrics.items()
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "mlflow": mlflow.__version__,
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "xgboost-cpu": version("xgboost-cpu"),
        },
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
