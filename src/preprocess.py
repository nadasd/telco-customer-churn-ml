import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def create_preprocessor(X_train: pd.DataFrame):

    # Identify numerical features
    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns

    # Identify categorical features
    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    return preprocessor