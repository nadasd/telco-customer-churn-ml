# Architecture

## Training and experiment tracking

Training runs locally or through the manually triggered `pipeline-validation.yml` workflow. The workflow downloads the reference IBM dataset at a pinned revision and verifies its SHA-256 checksum before training.

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
10. Self-contained model artifact packaging

MLflow provides experiment tracking and a model registry in the local environment or the isolated CI runner.

## Local serving

The complete local MLOps environment is orchestrated with Docker Compose:

- MLflow
- FastAPI
- Gradio

## Cloud deployment layout

The API and UI Dockerfiles accept a platform-provided `PORT`, supporting deployment on services such as Railway.

The inference deployment consists of:

- FastAPI
- Gradio
- Versioned model artifact

The cloud inference service does not require a running MLflow server.

This separation allows training and experiment tracking to remain independent of the inference services.
