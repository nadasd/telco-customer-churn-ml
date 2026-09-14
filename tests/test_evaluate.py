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
    y_test = np.array([0, 1, 1, 0])
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
    assert results == pytest.approx({
        "accuracy": 0.75,
        "precision": 2 / 3,
        "recall": 1.0,
        "f1_score": 0.8,
        "threshold": 0.50,
    })
