from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    """Les 19 données brutes attendues pour un client Telco."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddOn
    OnlineBackup: InternetAddOn
    DeviceProtection: InternetAddOn
    TechSupport: InternetAddOn
    StreamingTV: InternetAddOn
    StreamingMovies: InternetAddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    """Réponse renvoyée par l'API après une prédiction."""

    churn_probability: float = Field(ge=0, le=1)
    churn_prediction: Literal[0, 1]
    model_uri: str