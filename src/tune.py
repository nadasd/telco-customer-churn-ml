def tune_model(
    X_train,
    y_train,
    preprocessor,
    optuna_config,
    xgboost_config,
    seed,
):

    import optuna
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    search_space = optuna_config["search_space"]
    n_trials = optuna_config["n_trials"]
    cv_folds = optuna_config["cv_folds"]
    scoring = optuna_config["scoring"]

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )
    cv_n_jobs = optuna_config["n_jobs"]
    xgb_n_jobs = xgboost_config["n_jobs"]
    eval_metric = xgboost_config["eval_metric"]
    objective_name = xgboost_config["objective"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                search_space["n_estimators_min"],
                search_space["n_estimators_max"],
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                search_space["max_depth_min"],
                search_space["max_depth_max"],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                search_space["learning_rate_min"],
                search_space["learning_rate_max"],
                log=search_space["learning_rate_log"],
            ),
            "subsample": trial.suggest_float(
                "subsample",
                search_space["subsample_min"],
                search_space["subsample_max"],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                search_space["colsample_bytree_min"],
                search_space["colsample_bytree_max"],
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                search_space["min_child_weight_min"],
                search_space["min_child_weight_max"],
            ),
            "gamma": trial.suggest_float(
                "gamma",
                search_space["gamma_min"],
                search_space["gamma_max"],
            ),
        }

        model = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        **params,
                        random_state=seed,
                        n_jobs=xgb_n_jobs,
                        eval_metric=eval_metric,
                        objective=objective_name,
                    ),
                ),
            ]
        )

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=cv_n_jobs,
        )
        return scores.mean()

    study = optuna.create_study(
    direction=optuna_config["direction"],
    sampler=optuna.samplers.TPESampler(
        seed=optuna_config["seed"]
    )
)

    study.optimize(objective, n_trials=n_trials)

    print("\nBest Optuna parameters:")
    print(study.best_params)

    print(f"\nBest CV Recall: {study.best_value:.4f}")

    return study.best_params, study.best_value
