import numpy as np
import pytest

from src.evaluate import evaluate_model


class RecordingModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.received_input = None

    def predict_proba(self, X):
        self.received_input = X
        return self.probabilities


def test_evaluate_model_uses_positive_class_threshold_and_expected_metrics():
    X_test_processed = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y_test = np.array([
        0,
        1,
        1,
        0,
    ])

    model = RecordingModel(
        probabilities=np.array([
            [0.90, 0.10],
            [0.20, 0.80],
            [0.50, 0.50],
            [0.40, 0.60],
        ])
    )

    results = evaluate_model(
        model=model,
        X_test_processed=X_test_processed,
        y_test=y_test,
        threshold=0.50,
    )

    assert model.received_input is X_test_processed

    assert results["accuracy"] == pytest.approx(
        0.75
    )

    assert results["precision"] == pytest.approx(
        2 / 3
    )

    assert results["recall"] == pytest.approx(
        1.0
    )

    assert results["f1_score"] == pytest.approx(
        0.8
    )

    assert results["threshold"] == pytest.approx(
        0.50
    )

    assert results["tn"] == 1
    assert results["fp"] == 1
    assert results["fn"] == 0
    assert results["tp"] == 2

    assert 0 <= results["roc_auc"] <= 1
    assert 0 <= results["pr_auc"] <= 1