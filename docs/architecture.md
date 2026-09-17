# Architecture

## Training and experiment tracking

Training is executed locally.

The training pipeline includes:

1. Data loading and validation
2. Train/validation/test splitting
3. Preprocessing
4. Optuna hyperparameter optimization
5. XGBoost training
6. Classification threshold selection
7. Final evaluation
8. MLflow experiment tracking
9. Model registration
10. Deployment artifact export

MLflow is used locally for experiment tracking and model registry.

## Local serving

The complete local MLOps environment is orchestrated with Docker Compose:

- MLflow
- FastAPI
- Gradio

## Cloud deployment

Railway is used only as a lightweight public demonstration environment.

The Railway deployment contains:

- FastAPI
- Gradio
- Versioned model artifact

The cloud inference service does not require a running MLflow server.

This architecture is intentional due to the zero-cost deployment constraint.