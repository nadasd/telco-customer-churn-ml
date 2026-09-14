import numpy as np
import pytest

from src.select_threshold import select_threshold


class DummyModel:
    def __init__(self, positive_probabilities):
        self.positive_probabilities = np.asarray(
            positive_probabilities,
            dtype=float,
        )

    def predict_proba(self, X):
        negative_probabilities = 1 - self.positive_probabilities

        return np.column_stack(
            [
                negative_probabilities,
                self.positive_probabilities,
            ]
        )


def test_select_threshold_prioritizes_recall_then_f1():
    # At threshold 0.31:
    #
    # probabilities = [0.90, 0.31, 0.30]
    # labels        = [1,    1,    0]
    #
    # predictions = [1, 1, 0]
    #
    # precision = 1.0
    # recall    = 1.0
    # F1        = 1.0
    #
    # Lower thresholds include the false positive.
    # Higher thresholds lose one true positive.

    probabilities = [0.90, 0.31, 0.30]
    y_val = np.array([1, 1, 0])

    model = DummyModel(probabilities)

    X_val_processed = np.zeros((3, 2))

    threshold = select_threshold(
        model=model,
        X_val_processed=X_val_processed,
        y_val=y_val,
        min_precision=0.40,
    )

    assert threshold == pytest.approx(
        0.31,
        abs=1e-8,
    )


def test_select_threshold_uses_positive_class_probability():
    probabilities = [0.80, 0.70, 0.20]
    y_val = np.array([1, 1, 0])

    model = DummyModel(probabilities)

    X_val_processed = np.zeros((3, 1))

    threshold = select_threshold(
        model,
        X_val_processed,
        y_val,
        min_precision=0.50,
    )

    assert 0.05 <= threshold <= 0.50


def test_select_threshold_raises_when_precision_requirement_is_impossible():
    probabilities = [0.9, 0.8, 0.7]
    y_val = np.array([1, 0, 0])

    model = DummyModel(probabilities)

    X_val_processed = np.zeros((3, 1))

    with pytest.raises(
        ValueError,
        match="No threshold satisfies",
    ):
        select_threshold(
            model=model,
            X_val_processed=X_val_processed,
            y_val=y_val,
            min_precision=1.01,
        )