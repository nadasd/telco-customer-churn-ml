import sys
import os
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature

# MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# ============================================================
# Add project root to Python path
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


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
    ARTIFACT_VERSION,
    RANDOM_SEED,
    REGISTERED_MODEL_NAME,
    TelcoChurnArtifact,
    build_artifact_metadata,
    build_raw_feature_schema,
    build_raw_input_example,
    build_serving_pipeline,
)


def main():

    # ========================================================
    # MLflow experiment
    # ========================================================

    mlflow.set_experiment("Telco-Customer-Churn")

    with mlflow.start_run():

        # ====================================================
        # 1. Load data
        # ====================================================

        print("\n1. Loading data...")

        df = load_data(
            "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )

        raw_input_example = build_raw_input_example(df)

        print(f"Data loaded: {df.shape}")


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
        ) = split_data(df)

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
            preprocessor
        )

        print("\nBest parameters:")
        print(best_params)

        # Log hyperparameters in MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("seed", RANDOM_SEED)
        mlflow.log_param("artifact_version", ARTIFACT_VERSION)


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
            best_params
        )


        # ====================================================
        # 8. Select threshold using VALIDATION
        # ====================================================

        print("\n8. Selecting threshold using validation...")

        threshold = select_threshold(
            model,
            X_val_processed,
            y_val,
            min_precision=0.40
        )

        print(f"\nSelected threshold: {threshold}")

        # Log threshold in MLflow
        mlflow.log_param(
            "threshold",
            threshold
        )


        # ====================================================
        # 9. Final evaluation using TEST
        # ====================================================

        print("\n9. Final evaluation using test...")

        results = evaluate_model(
            model=model,
            X_test_processed=X_test_processed,
            y_test=y_test,
            threshold=threshold
        )


        # ====================================================
        # 10. Log final TEST metrics in MLflow
        # ====================================================

        print("\n10. Logging metrics to MLflow...")

        mlflow.log_metrics({
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1_score": results["f1_score"]
        })


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

        signature = infer_signature(
            raw_input_example,
            reference_output,
        )

        model_info = mlflow.pyfunc.log_model(
            python_model=artifact_model,
            name="model",
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
