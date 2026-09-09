import os
from functools import lru_cache
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.api.schemas import CustomerInput


def get_model_uri() -> str:
    """Retourne le modèle choisi pour servir les prédictions."""
    return os.getenv("MODEL_URI", "models:/TelcoChurnXGBoost/2")


def get_tracking_uri() -> str:
    """Retourne l'adresse du serveur MLflow."""
    return os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


@lru_cache
def load_model(model_uri: str) -> Any:
    """Charge le modèle une seule fois par URI."""
    mlflow.set_tracking_uri(get_tracking_uri())
    return mlflow.pyfunc.load_model(model_uri)


def predict_customer(customer: CustomerInput) -> dict[str, float | int]:
    """Prédit le risque de churn pour un seul client validé."""

    input_frame = pd.DataFrame([customer.model_dump()])

    prediction_frame = load_model(get_model_uri()).predict(input_frame)

    expected_columns = {"churn_probability", "churn_prediction"}
    if not isinstance(prediction_frame, pd.DataFrame):
        raise RuntimeError("L'artefact n'a pas retourné un DataFrame.")

    if not expected_columns.issubset(prediction_frame.columns):
        raise RuntimeError("La sortie de l'artefact ne respecte pas le contrat attendu.")

    result = prediction_frame.iloc[0]

    return {
        "churn_probability": float(result["churn_probability"]),
        "churn_prediction": int(result["churn_prediction"]),
    }