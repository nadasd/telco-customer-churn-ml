from fastapi import FastAPI, HTTPException, status

from src.api.schemas import CustomerInput, PredictionResponse
from src.api.service import get_model_uri, load_model, predict_customer


app = FastAPI(
    title="Telco Customer Churn API",
    version="1.0.0",
    description="API de prédiction du risque de résiliation client.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Telco Customer Churn API",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_model(get_model_uri())
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas disponible.",
        ) from error

    return {
        "status": "ok",
        "model_uri": get_model_uri(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> PredictionResponse:
    try:
        prediction = predict_customer(customer)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La prédiction est temporairement indisponible.",
        ) from error

    return PredictionResponse(
        **prediction,
        model_uri=get_model_uri(),
    )