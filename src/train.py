import pandas as pd

from xgboost import XGBClassifier

from src.preprocess import create_preprocessor


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: dict
):

    # 1. Create preprocessor using TRAIN only
    preprocessor = create_preprocessor(X_train)

    # 2. Fit preprocessing on TRAIN
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # 3. Transform VALIDATION
    X_val_processed = preprocessor.transform(
        X_val
    )

    # 4. Transform TEST
    X_test_processed = preprocessor.transform(
        X_test
    )

    # 5. Create model with Optuna's best parameters
    model = XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    # 6. Train only on TRAIN
    model.fit(
        X_train_processed,
        y_train
    )

    return (
        model,
        preprocessor,
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train,
        y_val,
        y_test
    )
