# Telco Customer Churn ML
[![CI](https://github.com/nadasd/telco-customer-churn-ml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/nadasd/telco-customer-churn-ml/actions/workflows/ci.yml)

Projet de Machine Learning de prédiction du risque de résiliation des clients d’une entreprise de télécommunications.

L’objectif métier est de détecter le plus possible de clients susceptibles de résilier. Nous privilégions donc le **recall**, même si cela diminue la précision.

## Architecture

```text
Dataset brut
    ↓
Pipeline d’entraînement modulaire
    ↓
MLflow : suivi des expériences et registre des modèles
    ↓
Artefact complet versionné
    ↓
FastAPI : validation des données et prédiction
```

L’artefact utilisé par l’API contient :

```text
19 variables client brutes
    ↓
nettoyage
    ↓
préprocesseur entraîné
    ↓
XGBoost
    ↓
probabilité de churn
    ↓
seuil de décision
    ↓
prédiction : 0 ou 1
```

## Modèle actuellement utilisé

- Run MLflow : `intrigued-goose-671`
- Run ID : `8dbd192a66df423588c5390ebe5ff4cc`
- Modèle enregistré : `TelcoChurnXGBoost`
- Version déployée : `2`
- URI du modèle : `models:/TelcoChurnXGBoost/2`
- Seed : `42`
- Seuil de décision : `0.1`

Résultats du modèle sur le jeu de test :

- Recall : `0.9146`
- Precision : `0.4054`
- F1-score : `0.5617`
- Accuracy : `0.6206`

## Structure du projet

```text
.
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── scripts/
│   └── run_pipeline.py
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── service.py
│   ├── clean.py
│   ├── model_artifact.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── train.py
│   ├── tune.py
│   └── ...
├── tests/
│   ├── test_api.py
│   └── test_mlflow_artifact.py
├── requirements.txt
└── requirements-dev.txt
```

## Installation

Créer puis activer l’environnement virtuel :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## Données

Le dataset Telco Customer Churn doit être placé ici :

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Les données brutes ne sont pas versionnées avec Git.

## Entraîner le pipeline

```powershell
python .\scripts\run_pipeline.py
```

Le pipeline réalise notamment :

1. Chargement et nettoyage des données ;
2. Validation ;
3. Séparation train / validation / test ;
4. Préprocessing entraîné uniquement sur le train ;
5. Optimisation des hyperparamètres avec Optuna ;
6. Entraînement XGBoost ;
7. Sélection du seuil sur validation ;
8. Évaluation finale sur test ;
9. Enregistrement d’un artefact complet dans MLflow.

## Lancer MLflow

Dans un premier terminal :

```powershell
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri "sqlite:///mlflow.db" `
  --artifacts-destination ".\mlartifacts" `
  --host 127.0.0.1 `
  --port 5000
```

Puis ouvrir :

```text
http://127.0.0.1:5000
```

## Lancer l’API

Dans un deuxième terminal :

```powershell
.\.venv\Scripts\Activate.ps1

$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MODEL_URI="models:/TelcoChurnXGBoost/2"

python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Documentation interactive de l’API :

```text
http://127.0.0.1:8000/docs
```

Vérification de disponibilité :

```text
http://127.0.0.1:8000/health
```

## Contrat de l’API

L’endpoint `POST /predict` reçoit les 19 variables client brutes.

Exemple :

```json
{
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
  "TotalCharges": 29.85
}
```

La réponse contient :

```json
{
  "churn_probability": 0.7028769850730896,
  "churn_prediction": 1,
  "model_uri": "models:/TelcoChurnXGBoost/2"
}
```

## Tests

Tests unitaires de l’API :

```powershell
python -m pytest -q tests\test_api.py
```

Test d’intégration de l’artefact MLflow :

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:TELCO_MODEL_URI="models:/TelcoChurnXGBoost/2"

python -m pytest -q tests\test_mlflow_artifact.py
```

Tous les tests :

```powershell
python -m pytest -q
python -m compileall -q src tests
```

## Prochaines étapes

- Conteneur Docker ;
- Intégration continue avec GitHub Actions ;
- Déploiement ;
- Monitoring de l’API et des performances du modèle.
## Exécution avec Docker Compose

Docker Compose démarre deux services :

- `mlflow` : suivi des expériences et registre du modèle ;
- `api` : API FastAPI qui charge `TelcoChurnXGBoost/2`.

### Démarrer les services

```powershell
docker compose up -d --build
docker compose ps
```

Les deux services doivent afficher l’état `healthy`.

- API : http://127.0.0.1:8000/docs
- MLflow : http://127.0.0.1:5000

### Arrêter les services

```powershell
docker compose down
```

Cette commande supprime les conteneurs et le réseau Docker, mais conserve `mlflow.db` et `mlartifacts` sur la machine hôte.

### Vérifier l’API

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

L’API doit retourner `status: ok` et l’URI `models:/TelcoChurnXGBoost/2`.