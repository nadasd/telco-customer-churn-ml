import pandas as pd

from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_state: int,
    stratify: bool,
):
    if not abs(train_fraction + validation_fraction + test_fraction - 1.0) < 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Split fractions must be positive.")


    # Separate features and target
    X = df.drop(columns=["churn"])
    y = df["churn"]

    temporary_fraction = validation_fraction + test_fraction
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temporary_fraction,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    test_fraction_of_temporary = test_fraction / temporary_fraction
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction_of_temporary,
        random_state=random_state,
        stratify=y_temp if stratify else None,
    )

    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
