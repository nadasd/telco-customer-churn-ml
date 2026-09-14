import pandas as pd

from src.split_data import split_data


def make_dataset(n_rows=100):
    return pd.DataFrame(
        {
            "feature_num": range(n_rows),
            "feature_cat": ["a", "b"] * (n_rows // 2),
            "churn": [0, 1] * (n_rows // 2),
        }
    )


def test_split_data_sizes_and_no_overlap():
    df = make_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    # Expected 70 / 15 / 15 split
    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15

    assert len(y_train) == 70
    assert len(y_val) == 15
    assert len(y_test) == 15

    # Target must not remain in the feature matrices
    assert "churn" not in X_train.columns
    assert "churn" not in X_val.columns
    assert "churn" not in X_test.columns

    # No row may appear in more than one split
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    test_indices = set(X_test.index)

    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)

    # All original rows must be preserved
    all_indices = train_indices | val_indices | test_indices
    assert all_indices == set(df.index)


def test_split_data_is_deterministic():
    df = make_dataset()

    split_1 = split_data(df)
    split_2 = split_data(df)

    X_train_1, X_val_1, X_test_1, *_ = split_1
    X_train_2, X_val_2, X_test_2, *_ = split_2

    assert X_train_1.index.tolist() == X_train_2.index.tolist()
    assert X_val_1.index.tolist() == X_val_2.index.tolist()
    assert X_test_1.index.tolist() == X_test_2.index.tolist()


def test_split_data_preserves_class_distribution():
    df = make_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df)

    original_rate = df["churn"].mean()

    assert abs(y_train.mean() - original_rate) <= 0.05
    assert abs(y_val.mean() - original_rate) <= 0.05
    assert abs(y_test.mean() - original_rate) <= 0.05