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

B --> C[Data Validation<br/>Great Expectations]

C --> D[Train Validation Test Split]

D --> E[Feature Engineering]

E --> F[XGBoost Model]

F --> G[Hyperparameter Optimization<br/>Optuna]

G --> H[Threshold Optimization]

H --> I[Model Evaluation]

I --> J[MLflow Tracking]

J --> K[MLflow Model Artifact]

K --> L[FastAPI Prediction API]

L --> M[Gradio Interface]
