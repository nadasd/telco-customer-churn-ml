# Telco Customer Churn Prediction — End-to-End ML & MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)
![MLOps](https://img.shields.io/badge/MLOps-MLflow-green)
![API](https://img.shields.io/badge/API-FastAPI-red)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)

## Overview

This project implements an end-to-end machine learning pipeline for **customer churn prediction** using the Telco Customer Churn dataset.

The objective is to predict customers with a high probability of leaving a telecom service provider and provide a reproducible ML workflow covering the complete machine learning lifecycle:

**Data Validation → Data Preparation → Model Training → Optimization → Experiment Tracking → Model Serving → Deployment**

The project follows MLOps best practices including modular code organization, experiment tracking, model packaging, API deployment, testing, and containerization.

---

# Business Problem

Customer retention is a major challenge in the telecommunications industry.

Identifying customers likely to churn allows companies to implement targeted retention strategies before losing them.

This project focuses on a binary classification problem:

- **Input:** Customer profile and service information
- **Output:** Probability of customer churn

The model prioritizes **recall** in order to minimize missed churn cases.

---

# Project Architecture

```mermaid
flowchart LR

A[Raw Customer Data] --> B[Data Cleaning]

B --> C[Data Validation - Great Expectations]

C --> D[Train Validation Test Split]

D --> E[Feature Engineering]

E --> F[XGBoost Model]

F --> G[Optuna Optimization]

G --> H[Threshold Optimization]

H --> I[Model Evaluation]

I --> J[MLflow Tracking]

J --> K[Model Artifact]

K --> L[FastAPI API]

L --> M[Gradio Interface]
Machine Learning Pipeline
Implemented workflow:
1. Data preparation
2. Data validation
3. Feature engineering
4. Model training
5. Hyperparameter optimization
6. Threshold selection
7. Model evaluation
Algorithm
- XGBoost Classifier
Optimization
- Optuna hyperparameter tuning
- Cross-validation
Model Strategy
The default classification threshold of 0.5 is not always optimal for churn prediction.
A validation-based threshold optimization strategy was implemented:
- Minimum precision constraint
- Recall maximization
- F1-score optimization
This approach improves the detection of potential churn customers.
Results
Final model performance:
Metric	Score
Accuracy	62.06%
Precision	40.54%
Recall	91.46%
F1-score	56.17%


Selected classification threshold:
0.10

MLOps Implementation
This project demonstrates a complete machine learning lifecycle.
Experiment Tracking
Implemented with:
- MLflow
Tracked information:
- Model parameters
- Training configuration
- Dataset information
- Evaluation metrics
- Model artifacts
Model Packaging
The final model artifact contains:
- Data preprocessing pipeline
- Trained XGBoost model
- Decision threshold
- Metadata
This reduces training-serving inconsistencies.
API Deployment
A FastAPI application exposes the model through REST endpoints.
Endpoint	Description
GET /	API information
GET /health	Model availability check
POST /predict	Generate churn prediction


Example:
{
  "churn_probability": 0.78,
  "churn_prediction": 1
}

Interactive Interface
A Gradio interface was developed to interact with the prediction API.
Features:
- Customer information input
- Real-time prediction
- Churn probability visualization
Technology Stack
Programming
- Python 3.12
Data Processing
- Pandas
- NumPy
Machine Learning
- Scikit-learn
- XGBoost
- Optuna
Data Validation
- Great Expectations
MLOps
- MLflow
- Model Registry
- Experiment Tracking
Backend
- FastAPI
- Pydantic
- Uvicorn
Engineering Tools
- Docker
- Docker Compose
- GitHub Actions
- Pytest
Repository Structure
telco-customer-churn-ml/

├── src/
│   ├── api/
│   ├── ui/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── model_artifact.py
│
├── scripts/
│   └── run_pipeline.py
│
├── configs/
├── tests/
├── model/
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── README.md

Installation
Clone the repository:
git clone https://github.com/nadasd/telco-customer-churn-ml.git

cd telco-customer-churn-ml

Create a virtual environment:
python -m venv .venv

Install dependencies:
pip install -r requirements.txt

Training the Model
Run the complete training pipeline:
python scripts/run_pipeline.py

The pipeline performs:
- Data preparation
- Validation
- Training
- Optimization
- Evaluation
- MLflow logging
Running the API
Start FastAPI:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

API documentation:
http://localhost:8000/docs

Running with Docker
Build and start services:
docker compose up --build

Services:
Service	Port
FastAPI	8000
Gradio	7860
MLflow	5000


Stop services:
docker compose down

Testing
Run tests:
pytest

Test coverage includes:
- Data preprocessing
- Model pipeline
- API endpoints
- Model artifact loading
- Prediction workflow
Future Improvements
- Add real-time monitoring
- Implement data drift detection
- Add automated model promotion
- Improve CI/CD deployment workflow
- Add model explainability using SHAP
- Add business cost optimization
Skills Demonstrated
- Machine Learning pipeline development
- Feature engineering
- Model optimization
- Classification threshold tuning
- MLOps workflow design
- Experiment tracking with MLflow
- Model packaging
- REST API deployment
- Docker containerization
- Testing and CI practices
Author
Nada Sadraoui
AI & Data Engineering Student
Interested in:
- Artificial Intelligence
- Machine Learning
- MLOps
- Data Analytics
GitHub:
https://github.com/nadasd

Après collage :

1. Sauvegarde `README.md`
2. Vérifie dans VS Code l'aperçu Markdown (`Ctrl + Shift + V`)
3. Puis :

```powershell
git add README.md
git commit -m "Fix README markdown formatting"
git push origin finalize-mlops