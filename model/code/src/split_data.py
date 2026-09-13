import pandas as pd

from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame):

    # Separate features and target
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # First split: 70% train, 30% temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Second split: 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
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