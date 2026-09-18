import os
import sys

# Add project root to Python path BEFORE importing src.*
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hashlib
import subprocess
import tempfile

import mlflow
import mlflow.pyfunc
import numpy as np
import random

from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

from src.config import load_training_config
from src.load_data import load_data
from src.clean import clean_data
from src.validate_data import validate_data
from src.split_data import split_data
from src.preprocess import create_preprocessor
from src.tune import tune_model
from src.train import train_model
from src.select_threshold import select_threshold
from src.evaluate import evaluate_model
from src.model_validation import (
    create_validation_artifacts,
    evaluate_cv_stability,
)
from src.model_artifact import (
    TelcoChurnArtifact,
    build_artifact_metadata,
    build_raw_feature_schema,
    build_raw_input_example,
    build_serving_pipeline,
)

# ============================================================
# Add project root to Python path
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.config import load_training_config

config = load_training_config()
seed = config["project"]["seed"]
data_config = config["data"]
split_config = config["split"]
optuna_config = config["optuna"]
xgboost_config = config["xgboost"]
threshold_config = config["threshold"]
mlflow_config = config["mlflow"]

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    mlflow_config["tracking_uri"],
)
mlflow.set_tracking_uri(tracking_uri)

import random
import numpy as np

random.seed(seed)
np.random.seed(seed)


# ============================================================
# Imports
# ============================================================

from src.load_data import load_data
from src.clean import clean_data
from src.validate_data import validate_data
from src.split_data import split_data
from src.preprocess import create_preprocessor
from src.tune import tune_model
from src.train import train_model
from src.select_threshold import select_threshold
from src.evaluate import evaluate_model
from src.model_artifact import (
    TelcoChurnArtifact,
    build_artifact_metadata,
    build_raw_feature_schema,
    build_raw_input_example,
    build_serving_pipeline,
)

DATA_PATH = data_config["path"]
MIN_PRECISION = threshold_config["minimum_precision"]
ARTIFACT_VERSION = config["project"]["artifact_version"]
REGISTERED_MODEL_NAME = mlflow_config["registered_model_name"]

def compute_dataset_hash(df):
    csv_bytes = df.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")

    return hashlib.sha256(csv_bytes).hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
def build_mlflow_schema(df):
    dtype_mapping = {
        "object": "string",
        "int64": "long",
        "float64": "double",
        "bool": "boolean",
    }

    columns = []

    for name, dtype in df.dtypes.items():
        dtype_name = str(dtype)

        if dtype_name not in dtype_mapping:
            raise TypeError(
                f"Unsupported dtype for MLflow signature: "
                f"{name}={dtype_name}"
            )

        columns.append(
            ColSpec(
                dtype_mapping[dtype_name],
                name,
                required=True,
            )
        )

    return Schema(columns)
def main():

    # ========================================================
    # MLflow experiment
    # ========================================================

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run():

        # ====================================================
        # 1. Load data
        # ====================================================

        print("\n1. Loading data...")

        df = load_data(DATA_PATH)

        raw_input_example = build_raw_input_example(df)

        print(f"Data loaded: {df.shape}")
        dataset_rows, dataset_columns = df.shape

        mlflow.log_param("dataset_rows", dataset_rows)
        mlflow.log_param("dataset_columns", dataset_columns)
        dataset_hash = compute_dataset_hash(df)

        mlflow.set_tag("dataset_source", DATA_PATH)
        mlflow.set_tag("dataset_sha256", dataset_hash)
        git_commit = get_git_commit()

        mlflow.set_tag("git_commit", git_commit)


        # ====================================================
        # 2. Clean data
        # ====================================================

        print("\n2. Cleaning data...")

        df = clean_data(df)

        print(f"Data cleaned: {df.shape}")


        # ====================================================
        # 3. Validate data
        # ====================================================

        print("\n3. Validating data...")

        is_valid = validate_data(df)

        if not is_valid:
            raise ValueError(
                "Data validation failed. Pipeline stopped."
            )


        # ====================================================
        # 4. Split data
        # ====================================================

        print("\n4. Splitting data...")

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = split_data(
            df,
            train_fraction=split_config["train_fraction"],
            validation_fraction=split_config["validation_fraction"],
            test_fraction=split_config["test_fraction"],
            random_state=seed,
            stratify=split_config.get("stratify", True),
        )

        print(f"Train: {X_train.shape}")
        print(f"Validation: {X_val.shape}")
        print(f"Test: {X_test.shape}")


        # ====================================================
        # 5. Create preprocessor
        # ====================================================

        print("\n5. Creating preprocessor...")

        preprocessor = create_preprocessor(X_train)


        # ====================================================
        # 6. Optuna hyperparameter tuning
        # ====================================================

        print("\n6. Starting Optuna tuning...")

        best_params = tune_model(
            X_train,
            y_train,
            preprocessor,
            optuna_config=optuna_config,
            xgboost_config=xgboost_config,
            seed=seed,
        )

        print("\nBest parameters:")
        print(best_params)
        print("\nEvaluating cross-validation stability...")

        cv_results = evaluate_cv_stability(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            best_params=best_params,
            optuna_config=optuna_config,
            xgboost_config=xgboost_config,
            seed=seed,
        )

        print(
            f"CV {cv_results['scoring']} mean: "
            f"{cv_results['mean']:.4f}"
        )

        print(
            f"CV {cv_results['scoring']} std: "
            f"{cv_results['std']:.4f}"
        )

        # Log hyperparameters in MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("seed", seed)
        mlflow.log_param("artifact_version", ARTIFACT_VERSION)
        mlflow.log_param("n_trials", optuna_config["n_trials"])
        mlflow.log_param("min_precision", MIN_PRECISION)
        mlflow.log_param(
            "cv_scoring",
            cv_results["scoring"],
        )


        # ====================================================
        # 7. Train final model
        # ====================================================

        print("\n7. Training final model...")

        (
            model,
            preprocessor,
            X_train_processed,
            X_val_processed,
            X_test_processed,
            y_train,
            y_val,
            y_test
        ) = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            best_params,
            xgboost_config=xgboost_config,
            seed=seed,
        )


        # ====================================================
        # 8. Select threshold using VALIDATION
        # ====================================================

        print("\n8. Selecting threshold using validation...")

        threshold = select_threshold(
            model,
            X_val_processed,
            y_val,
            minimum=threshold_config["minimum"],
            maximum=threshold_config["maximum"],
            step=threshold_config["step"],
            min_precision=threshold_config["minimum_precision"],
        )

        print(f"\nSelected threshold: {threshold}")

        # Log threshold in MLflow
        mlflow.log_param(
            "threshold",
            threshold
        )


        # ====================================================
        # 9. Model validation
        # ====================================================

        print("\n9. Evaluating generalization...")

        train_results = evaluate_model(
            model=model,
            X_test_processed=X_train_processed,
            y_test=y_train,
            threshold=threshold,
            dataset_name="Train",
        )

        validation_results = evaluate_model(
            model=model,
            X_test_processed=X_val_processed,
            y_test=y_val,
            threshold=threshold,
            dataset_name="Validation",
        )

        test_results = evaluate_model(
            model=model,
            X_test_processed=X_test_processed,
            y_test=y_test,
            threshold=threshold,
            dataset_name="Test",
        )

        # Keep backward compatibility:
        # final "results" means test results.
        results = test_results
        generalization_gaps = {
            "f1_train_minus_validation": (
                train_results["f1_score"]
                - validation_results["f1_score"]
            ),
            "f1_train_minus_test": (
                train_results["f1_score"]
                - test_results["f1_score"]
            ),
            "recall_train_minus_validation": (
                train_results["recall"]
                - validation_results["recall"]
            ),
            "recall_train_minus_test": (
                train_results["recall"]
                - test_results["recall"]
            ),
        }

        print("\nGeneralization Gaps")
        print("-------------------")

        for name, value in generalization_gaps.items():
            print(f"{name}: {value:.4f}")


               # ====================================================
        # 10. Log validation metrics in MLflow
        # ====================================================

        print("\n10. Logging validation metrics to MLflow...")

        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
            "predicted_positive_rate",
            "tn",
            "fp",
            "fn",
            "tp",
        ]

        all_split_results = {
            "train": train_results,
            "validation": validation_results,
            "test": test_results,
        }

        mlflow_metrics = {}

        for split_name, split_results in all_split_results.items():
            for metric_name in metric_names:
                mlflow_metrics[
                    f"{split_name}_{metric_name}"
                ] = split_results[metric_name]

        mlflow_metrics.update(
            generalization_gaps
        )

        mlflow_metrics["cv_score_mean"] = (
            cv_results["mean"]
        )

        mlflow_metrics["cv_score_std"] = (
            cv_results["std"]
        )

        mlflow.log_metrics(
            mlflow_metrics
        )
                # ====================================================
        # Create model validation artifacts
        # ====================================================

        val_proba = model.predict_proba(
            X_val_processed
        )[:, 1]

        test_proba = model.predict_proba(
            X_test_processed
        )[:, 1]

        metrics_summary = {
            "train": train_results,
            "validation": validation_results,
            "test": test_results,
            "generalization_gaps": generalization_gaps,
            "cross_validation": cv_results,
        }

        with tempfile.TemporaryDirectory() as temp_dir:

            create_validation_artifacts(
                y_val=y_val,
                val_proba=val_proba,
                y_test=y_test,
                test_proba=test_proba,
                selected_threshold=threshold,
                threshold_config=threshold_config,
                metrics_summary=metrics_summary,
                output_dir=temp_dir,
            )

            mlflow.log_artifacts(
                temp_dir,
                artifact_path="model_validation",
            )
        # ====================================================
        # 11. Log model in MLflow
        # ====================================================

        print("\n11. Logging model to MLflow...")

        raw_feature_schema = build_raw_feature_schema(
            raw_input_example
        )

        serving_pipeline = build_serving_pipeline(
            preprocessor=preprocessor,
            classifier=model,
            raw_input_example=raw_input_example,
        )

        artifact_metadata = build_artifact_metadata(
            raw_feature_schema=raw_feature_schema,
            threshold=threshold,
            model_parameters=best_params,
            test_metrics=results,
            artifact_version=ARTIFACT_VERSION,
            registered_model_name=REGISTERED_MODEL_NAME,
            random_seed=seed,
        )

        artifact_model = TelcoChurnArtifact(
            pipeline=serving_pipeline,
            threshold=threshold,
            raw_feature_schema=raw_feature_schema,
            metadata=artifact_metadata,
        )

        reference_output = artifact_model.predict(
            context=None,
            model_input=raw_input_example,
        )
        artifact_metadata["reference_output"] = {
            "churn_probability": float(
                reference_output.iloc[0]["churn_probability"]
            ),
            "churn_prediction": int(
                reference_output.iloc[0]["churn_prediction"]
            ),
        }
        signature = ModelSignature(
    inputs=build_mlflow_schema(raw_input_example),
    outputs=build_mlflow_schema(reference_output),
)

        model_info = mlflow.pyfunc.log_model(
            python_model=artifact_model,
            name=mlflow_config["artifact_name"],
            code_paths=["src"],
            signature=signature,
            input_example=raw_input_example,
            metadata=artifact_metadata,
            registered_model_name=REGISTERED_MODEL_NAME,
            tags={
                "artifact_version": ARTIFACT_VERSION,
                "serving_contract": "raw-19-features",
            },
        )

        print(f"Model URI: {model_info.model_uri}")
        print(
            "Registered model version: "
            f"{model_info.registered_model_version}"
        )


        # ====================================================
        # Pipeline completed
        # ====================================================

        print("\nPipeline completed successfully.")

        return results


if __name__ == "__main__":
    main()
