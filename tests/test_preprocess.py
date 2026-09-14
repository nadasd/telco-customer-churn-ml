import numpy as np
import pandas as pd

from src.preprocess import create_preprocessor


def _as_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return np.asarray(matrix)


def test_preprocessor_handles_numeric_and_categorical_features():
    X_train = pd.DataFrame(
        {
            "tenure": [1, 2, 3, 4],
            "monthlycharges": [20.0, 40.0, 60.0, 80.0],
            "contract": [
                "month-to-month",
                "one year",
                "two year",
                "month-to-month",
            ],
        }
    )

    preprocessor = create_preprocessor(X_train)

    transformed = preprocessor.fit_transform(X_train)
    transformed = _as_dense(transformed)

    assert transformed.shape[0] == len(X_train)
    assert transformed.shape[1] > X_train.shape[1]

    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unknown_categories():
    X_train = pd.DataFrame(
        {
            "tenure": [1, 2, 3],
            "monthlycharges": [20.0, 40.0, 60.0],
            "contract": [
                "month-to-month",
                "one year",
                "two year",
            ],
        }
    )

    X_new = pd.DataFrame(
        {
            "tenure": [10],
            "monthlycharges": [75.0],
            # Category never seen during fit
            "contract": ["unknown-contract"],
        }
    )

    preprocessor = create_preprocessor(X_train)

    train_transformed = preprocessor.fit_transform(X_train)
    new_transformed = preprocessor.transform(X_new)

    train_transformed = _as_dense(train_transformed)
    new_transformed = _as_dense(new_transformed)

    # Same feature space must be preserved
    assert new_transformed.shape[1] == train_transformed.shape[1]

    # Unknown category must not crash or produce invalid values
    assert np.isfinite(new_transformed).all()


def test_preprocessor_fits_scaler_only_on_training_data():
    X_train = pd.DataFrame(
        {
            "tenure": [1.0, 3.0],
            "contract": ["one year", "two year"],
        }
    )

    preprocessor = create_preprocessor(X_train)

    preprocessor.fit(X_train)

    numeric_transformer = preprocessor.named_transformers_["num"]

    # Mean of training tenure: (1 + 3) / 2 = 2
    assert numeric_transformer.mean_[0] == 2.0