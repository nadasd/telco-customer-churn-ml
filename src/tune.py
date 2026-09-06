def tune_model(X_train, y_train, preprocessor):

    import optuna
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 500
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 6
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 10
            ),
            "gamma": trial.suggest_float(
                "gamma", 0, 5
            )
        }

        model = Pipeline([
            ("preprocessing", preprocessor),
            ("model", XGBClassifier(
                **params,
                random_state=42,
                eval_metric="logloss"
            ))
        ])

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="recall",
            n_jobs=-1
        )

        return scores.mean()

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        objective,
        n_trials=100
    )

    print("\nBest Optuna parameters:")
    print(study.best_params)

    print(f"\nBest CV Recall: {study.best_value:.4f}")

    return study.best_params