import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def select_threshold(
    model,
    X_val_processed,
    y_val,
    min_precision=0.40
):
    """
    Select the threshold using the validation set.

    We prioritize recall while requiring
    a minimum precision.
    """

    # Get churn probabilities
    y_proba = model.predict_proba(
        X_val_processed
    )[:, 1]

    thresholds = np.arange(
        0.05,
        0.51,
        0.01
    )

    results = []

    for threshold in thresholds:

        y_pred = (
            y_proba >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            y_pred,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_val,
            y_pred
        )

        if precision >= min_precision:

            results.append({
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    if not results:
        raise ValueError(
            "No threshold satisfies the minimum precision."
        )

    # Convert to DataFrame
    import pandas as pd

    results_df = pd.DataFrame(results)

    # We prioritize recall,
    # then F1 score
    best = results_df.sort_values(
        by=["recall", "f1"],
        ascending=False
    ).iloc[0]

    threshold = float(best["threshold"])

    print("\nThreshold Selection")
    print("-------------------")
    print(f"Threshold : {threshold:.2f}")
    print(f"Precision : {best['precision']:.4f}")
    print(f"Recall    : {best['recall']:.4f}")
    print(f"F1 Score  : {best['f1']:.4f}")

    return threshold