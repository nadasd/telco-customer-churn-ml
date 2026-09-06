import sys
import os
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================
# MLflow connection
# ============================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")


# ============================================
# Add project root to Python path
# ============================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


# ============================================
# Import project functions
# ============================================

from src.load_data import load_data
from src.clean import clean_data
from src.split_data import split_data
from src.preprocess import create_preprocessor


def evaluate_dataset(
    name,
    model,
    X_processed,
    y,
    threshold
):

    # Get churn probabilities
    y_proba = model.predict_proba(
        X_processed
    )[:, 1]

    # Apply the SAME threshold
    y_pred = (
        y_proba >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    print(f"\n{name}")
    print("-------------------------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():

    # ============================================
    # 1. Load data
    # ============================================

    df = load_data(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    print(f"Data loaded: {df.shape}")


    # ============================================
    # 2. Clean data
    # ============================================

    df = clean_data(df)

    print(f"Data cleaned: {df.shape}")


    # ============================================
    # 3. Recreate SAME split
    # ============================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(df)


    # ============================================
    # 4. Create preprocessing
    #    FIT ONLY ON TRAIN
    # ============================================

    preprocessor = create_preprocessor(
        X_train
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )


    # ============================================
    # 5. Load the SAVED model from MLflow
    # ============================================

    run_id = "8e10166621144ca59117639c6678dcb3"

    model_uri = f"runs:/{run_id}/model"

    print("\nLoading model from MLflow...")

    model = mlflow.sklearn.load_model(
    model_uri
)

    print("Model loaded successfully.")


    # ============================================
    # 6. Same threshold as final model
    # ============================================

    threshold = 0.11


    # ============================================
    # 7. Evaluate TRAIN
    # ============================================

    train_results = evaluate_dataset(
        "TRAIN",
        model,
        X_train_processed,
        y_train,
        threshold
    )


    # ============================================
    # 8. Evaluate VALIDATION
    # ============================================

    val_results = evaluate_dataset(
        "VALIDATION",
        model,
        X_val_processed,
        y_val,
        threshold
    )


    # ============================================
    # 9. Evaluate TEST
    # ============================================

    test_results = evaluate_dataset(
        "TEST",
        model,
        X_test_processed,
        y_test,
        threshold
    )


    # ============================================
    # 10. Compare F1
    # ============================================

    print("\n")
    print("OVERFITTING CHECK")
    print("=================")

    print(
        f"Train F1       : {train_results['f1']:.4f}"
    )

    print(
        f"Validation F1  : {val_results['f1']:.4f}"
    )

    print(
        f"Test F1        : {test_results['f1']:.4f}"
    )


    # Generalization gaps

    train_test_gap = (
        train_results["f1"]
        - test_results["f1"]
    )

    train_val_gap = (
        train_results["f1"]
        - val_results["f1"]
    )

    print(
        f"\nTrain-Test F1 gap : {train_test_gap:.4f}"
    )

    print(
        f"Train-Val F1 gap  : {train_val_gap:.4f}"
    )


    # ============================================
    # 11. Simple interpretation
    # ============================================

    if train_test_gap < 0.10:

        print(
            "\nConclusion:"
        )

        print(
            "No strong evidence of overfitting."
        )

    elif train_test_gap < 0.20:

        print(
            "\nConclusion:"
        )

        print(
            "There is some generalization gap, "
            "but it is not necessarily severe."
        )

    else:

        print(
            "\nConclusion:"
        )

        print(
            "There is a large generalization gap "
            "and overfitting should be investigated."
        )


if __name__ == "__main__":
    main()