import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains an XGBoost model using a train/validation/test split, selects a probability
    threshold on the validation set, evaluates on the test set, and logs everything
    to MLflow. Also saves the final model and the chosen threshold to disk.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
    """
    import os
    import joblib
    import numpy as np
    from sklearn.metrics import precision_score, f1_score

    # Features / target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 70% train, 15% val, 15% test (via two-step split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    )

    with mlflow.start_run():
        # Train on the training split
        model.fit(X_train, y_train)

        # ============================================================
        # 1) Choose threshold on VALIDATION set
        # ============================================================
        y_val_proba = model.predict_proba(X_val)[:, 1]

        thresholds = np.arange(0.10, 0.51, 0.01)
        validation_results = []

        for thr in thresholds:
            y_val_pred = (y_val_proba >= thr).astype(int)

            validation_results.append({
                "threshold": float(thr),
                "precision": precision_score(y_val, y_val_pred, zero_division=0),
                "recall": recall_score(y_val, y_val_pred, zero_division=0),
                "f1": f1_score(y_val, y_val_pred, zero_division=0),
            })

        validation_results = pd.DataFrame(validation_results)

        # Keep thresholds that meet a minimum precision requirement
        candidates = validation_results[validation_results["precision"] >= 0.40]

        if candidates.empty:
            # fallback: pick threshold that maximizes recall then f1
            best_threshold = float(
                validation_results.sort_values(by=["recall", "f1"], ascending=False).iloc[0]["threshold"]
            )
        else:
            best_threshold = float(
                candidates.sort_values(by=["recall", "f1"], ascending=False).iloc[0]["threshold"]
            )

        # ============================================================
        # 2) Evaluate on TEST one single time with the chosen threshold
        # ============================================================
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= best_threshold).astype(int)

        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)

        # Log params, threshold, metrics, and model
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("threshold", float(best_threshold))

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("f1_score", float(f1))

        mlflow.xgboost.log_model(model, "model")

        # Save final model object together with threshold for easy loading later
        os.makedirs("models", exist_ok=True)
        final_model = {"model": model, "threshold": float(best_threshold)}
        joblib.dump(final_model, os.path.join("models", "final_xgboost_model.pkl"))

        # 🔑 Log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print("Seuil sélectionné sur validation :", best_threshold)
        print(f"Résultats finaux sur le TEST — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
