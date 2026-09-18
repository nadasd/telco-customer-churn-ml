import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    y_true,
    y_proba,
    threshold: float,
):
    """
    Evaluate binary classification probabilities at a fixed threshold.

    Important:
    - threshold-dependent metrics: accuracy, precision, recall, F1
    - threshold-independent metrics: ROC-AUC, PR-AUC
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # AUC metrics require both classes to be present.
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            y_proba,
        )

        pr_auc = average_precision_score(
            y_true,
            y_proba,
        )
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    predicted_positive_rate = float(
        y_pred.mean()
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "predicted_positive_rate": predicted_positive_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def evaluate_model(
    model,
    X_test_processed,
    y_test,
    threshold: float,
    dataset_name: str = "Test",
):

    # Probability of churn = positive class
    y_proba = model.predict_proba(
        X_test_processed
    )[:, 1]

    results = evaluate_probabilities(
        y_true=y_test,
        y_proba=y_proba,
        threshold=threshold,
    )

    print(f"\n{dataset_name} Evaluation")
    print("----------------")
    print(f"Threshold : {threshold}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1_score']:.4f}")
    print(f"ROC-AUC   : {results['roc_auc']:.4f}")
    print(f"PR-AUC    : {results['pr_auc']:.4f}")
    print(
        "Confusion : "
        f"TN={results['tn']} "
        f"FP={results['fp']} "
        f"FN={results['fn']} "
        f"TP={results['tp']}"
    )

    return results