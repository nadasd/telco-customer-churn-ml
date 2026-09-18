import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
)


def save_confusion_matrix(
    y_true,
    y_proba,
    threshold,
    output_path,
):
    y_pred = (
        np.asarray(y_proba) >= threshold
    ).astype(int)

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=[
            "No Churn",
            "Churn",
        ],
        ax=ax,
    )

    ax.set_title(
        f"Test Confusion Matrix — Threshold {threshold:.2f}"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_roc_curve(
    y_true,
    y_proba,
    output_path,
):
    fig, ax = plt.subplots()

    RocCurveDisplay.from_predictions(
        y_true,
        y_proba,
        ax=ax,
    )

    ax.set_title("Test ROC Curve")

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_precision_recall_curve(
    y_true,
    y_proba,
    output_path,
):
    fig, ax = plt.subplots()

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_proba,
        ax=ax,
    )

    ax.set_title(
        "Test Precision-Recall Curve"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_threshold_curve(
    y_true,
    y_proba,
    selected_threshold,
    minimum,
    maximum,
    step,
    output_path,
):
    thresholds = np.arange(
        minimum,
        maximum + step / 2,
        step,
    )

    precision_values = []
    recall_values = []
    f1_values = []

    for threshold in thresholds:
        y_pred = (
            np.asarray(y_proba) >= threshold
        ).astype(int)

        precision_values.append(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        recall_values.append(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        f1_values.append(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

    fig, ax = plt.subplots()

    ax.plot(
        thresholds,
        precision_values,
        label="Precision",
    )

    ax.plot(
        thresholds,
        recall_values,
        label="Recall",
    )

    ax.plot(
        thresholds,
        f1_values,
        label="F1",
    )

    ax.axvline(
        selected_threshold,
        linestyle="--",
        label=f"Selected threshold = {selected_threshold:.2f}",
    )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Validation Threshold Analysis"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_metrics_summary(
    metrics,
    output_path,
):
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            allow_nan=True,
        )


def create_validation_artifacts(
    y_val,
    val_proba,
    y_test,
    test_proba,
    selected_threshold,
    threshold_config,
    metrics_summary,
    output_dir,
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_confusion_matrix(
        y_true=y_test,
        y_proba=test_proba,
        threshold=selected_threshold,
        output_path=(
            output_dir
            / "confusion_matrix_test.png"
        ),
    )

    save_roc_curve(
        y_true=y_test,
        y_proba=test_proba,
        output_path=(
            output_dir
            / "roc_curve_test.png"
        ),
    )

    save_precision_recall_curve(
        y_true=y_test,
        y_proba=test_proba,
        output_path=(
            output_dir
            / "precision_recall_curve_test.png"
        ),
    )

    # IMPORTANT:
    # Threshold was selected on VALIDATION,
    # therefore threshold diagnostics must also
    # use validation data, not test data.
    save_threshold_curve(
        y_true=y_val,
        y_proba=val_proba,
        selected_threshold=selected_threshold,
        minimum=threshold_config["minimum"],
        maximum=threshold_config["maximum"],
        step=threshold_config["step"],
        output_path=(
            output_dir
            / "threshold_curve_validation.png"
        ),
    )

    save_metrics_summary(
        metrics=metrics_summary,
        output_path=(
            output_dir
            / "metrics_summary.json"
        ),
    )
def evaluate_cv_stability(
    X_train,
    y_train,
    preprocessor,
    best_params,
    optuna_config,
    xgboost_config,
    seed,
):
    model = Pipeline([
        ("preprocessing", preprocessor),
        (
            "model",
            XGBClassifier(
                **best_params,
                random_state=seed,
                n_jobs=xgboost_config["n_jobs"],
                eval_metric=xgboost_config["eval_metric"],
                objective=xgboost_config["objective"],
            ),
        ),
    ])

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=optuna_config["cv_folds"],
        scoring=optuna_config["scoring"],
        n_jobs=optuna_config["n_jobs"],
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    return {
        "scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scoring": optuna_config["scoring"],
    }