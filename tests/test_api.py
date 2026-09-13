import os

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.model_artifact import DECISION_COLUMN, PROBABILITY_COLUMN

VALID_CUSTOMER = {
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


def test_health_reports_ready_model(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "models:/TelcoChurnXGBoost/2")
    monkeypatch.setattr(
        "src.api.main.load_model",
        lambda model_uri: object(),
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_uri": "models:/TelcoChurnXGBoost/2",
    }


def test_predict_returns_a_valid_prediction(monkeypatch):
    monkeypatch.setenv("MODEL_URI", "models:/TelcoChurnXGBoost/2")

    def fake_predict_customer(customer):
        assert customer.gender == "Female"

        return {
            "churn_probability": 0.7028,
            "churn_prediction": 1,
        }

    monkeypatch.setattr(
        "src.api.main.predict_customer",
        fake_predict_customer,
    )

    client = TestClient(app)
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    assert response.json() == {
        "churn_probability": 0.7028,
        "churn_prediction": 1,
        "model_uri": "models:/TelcoChurnXGBoost/2",
    }


def test_predict_rejects_an_unknown_category():
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["InternetService"] = "Satellite"

    client = TestClient(app)
    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422


def test_predict_returns_503_when_service_fails(monkeypatch):
    def unavailable_model(customer):
        raise RuntimeError("MLflow is unavailable.")

    monkeypatch.setattr(
        "src.api.main.predict_customer",
        unavailable_model,
    )

    client = TestClient(app)
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "La prédiction est temporairement indisponible.",
    }
def test_api_prediction_matches_real_mlflow_artifact(monkeypatch):
    model_uri = os.getenv("TELCO_MODEL_URI")
    if not model_uri:
        pytest.skip("TELCO_MODEL_URI must identify the artifact under test")

    monkeypatch.setenv("MODEL_URI", model_uri)

    loaded_model = mlflow.pyfunc.load_model(model_uri)

    input_frame = pd.DataFrame([VALID_CUSTOMER])

    artifact_prediction = loaded_model.predict(input_frame)

    artifact_probability = float(
        artifact_prediction.iloc[0][PROBABILITY_COLUMN]
    )
    artifact_decision = int(
        artifact_prediction.iloc[0][DECISION_COLUMN]
    )

    client = TestClient(app)

    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    api_prediction = response.json()

    assert api_prediction["churn_probability"] == pytest.approx(
        artifact_probability,
        rel=1e-12,
        abs=1e-12,
    )

    assert api_prediction["churn_prediction"] == artifact_decision
    assert api_prediction["model_uri"] == model_uri