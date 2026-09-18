# Telco Customer Churn — ML & MLOps

[![CI](https://github.com/nadasd/telco-customer-churn-ml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nadasd/telco-customer-churn-ml/actions/workflows/ci.yml)

An end-to-end machine-learning project for identifying telecom customers at risk of churn.

The project is designed around a practical business trade-off: missing a customer who is likely to leave can be more costly than contacting a customer who stays. The model therefore uses a recall-oriented decision threshold, while reporting the associated precision and F1 trade-offs.

> This is a portfolio project. The reported metrics are tied to the current recorded model evaluation and should not be interpreted as production performance without fresh, reproducible validation.

## Project pipeline

```text
Raw customer data
      ↓
Data cleaning and validation
      ↓
Train / validation / test split
      ↓
Fitted preprocessing pipeline
      ↓
Optuna tuning + XGBoost
      ↓
Validation-based threshold selection
      ↓
MLflow tracking and model artifact
      ↓
FastAPI prediction service
```

## Current model snapshot

| Metric | Recorded value |
|---|---:|
| Model | XGBoost |
| Model version | 2 |
| Random seed | 42 |
| Decision threshold | 0.10 |
| Recall | 0.9146 |
| Precision | 0.4054 |
| F1-score | 0.5617 |
| Accuracy | 0.6206 |

The low threshold intentionally increases the number of flagged customers. This improves recall but also increases false positives; threshold selection should ultimately be driven by campaign capacity and the business cost of false positives versus false negatives.

## Serving

The API accepts the 19 raw customer features, applies the same cleaning and preprocessing used during training, and returns a churn probability and decision.

Example response:

```json
{
  "churn_probability": 0.7028769850730896,
  "churn_prediction": 1,
  "model_uri": "models:/TelcoChurnXGBoost/2"
}
```

A live interface is available at:

[Open the demo →](https://telco-churn-ui-production.up.railway.app/)

## Main technologies

- **Python** — project implementation
- **Pandas / scikit-learn** — data processing and modeling
- **XGBoost** — classifier
- **Optuna** — hyperparameter optimization
- **Great Expectations** — data-quality checks
- **MLflow** — experiment tracking and model management
- **FastAPI / Uvicorn** — model serving
- **Pytest** — automated tests
- **Docker** — reproducible runtime
- **GitHub Actions** — continuous integration

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS/Linux
# .venv\\Scripts\\Activate.ps1            # Windows PowerShell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Place the dataset at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Then run:

```bash
python scripts/run_pipeline.py
python -m pytest -q
python -m compileall -q src tests
```

To start the API:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Limitations and next improvements

- Reproduce the final evaluation from a clean environment before treating the metrics as final.
- Add PR-AUC, calibration, confidence intervals, and cost-sensitive threshold analysis.
- Version the model artifact, input signature, feature contract, and dataset manifest together.
- Add a reproducible Docker build and smoke test to CI.
- Add structured logs and lightweight API monitoring.

## License

This project is for educational and portfolio purposes.
