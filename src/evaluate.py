from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(
    model,
    X_test_processed,
    y_test,
    threshold=0.12
):

    # Get probability of churn
    y_proba = model.predict_proba(
        X_test_processed
    )[:, 1]

    # Apply threshold
    y_pred = (
        y_proba >= threshold
    ).astype(int)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\nModel Evaluation")
    print("----------------")
    print(f"Threshold : {threshold}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "threshold": threshold
    }