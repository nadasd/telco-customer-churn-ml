import sys
import os
import mlflow
import mlflow.sklearn

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

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )


        # ====================================================
        # Pipeline completed
        # ====================================================

        print("\nPipeline completed successfully.")

        return results


if __name__ == "__main__":
    main()