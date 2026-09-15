import os
from types import SimpleNamespace

import mlflow
import pandas as pd

from src.load_data import load_data as real_load_data
from src.validate_data import validate_data as real_validate_data


def make_raw_telco_dataframe(n_rows=240):
    rows = []

    payment_methods = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    for i in range(n_rows):
        churn = i % 3 == 0

        internet_service = [
            "DSL",
            "Fiber optic",
            "No",
        ][i % 3]

        phone_service = (
            "No"
            if i % 5 == 0
            else "Yes"
        )

        if phone_service == "No":
            multiple_lines = "No phone service"
        else:
            multiple_lines = (
                "Yes"
                if i % 2
                else "No"
            )

        if internet_service == "No":
            internet_option = "No internet service"
        else:
            internet_option = (
                "Yes"
                if i % 2
                else "No"
            )

        contract = (
            "Month-to-month"
            if churn
            else (
                "One year"
                if i % 2
                else "Two year"
            )
        )

        tenure = (i % 72) + 1

        monthly_charges = (
            85.0 + (i % 20)
            if churn
            else 25.0 + (i % 20)
        )

        total_charges = (
            tenure * monthly_charges
        )

        rows.append(
            {
                "customerID": f"SMOKE-{i:05d}",
                "gender": (
                    "Male"
                    if i % 2
                    else "Female"
                ),
                "SeniorCitizen": i % 2,
                "Partner": (
                    "Yes"
                    if i % 2
                    else "No"
                ),
                "Dependents": (
                    "Yes"
                    if i % 4 == 0
                    else "No"
                ),
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": internet_option,
                "OnlineBackup": internet_option,
                "DeviceProtection": internet_option,
                "TechSupport": internet_option,
                "StreamingTV": internet_option,
                "StreamingMovies": internet_option,
                "Contract": contract,
                "PaperlessBilling": (
                    "Yes"
                    if i % 2
                    else "No"
                ),
                "PaymentMethod": (
                    payment_methods[
                        i % len(payment_methods)
                    ]
                ),
                "MonthlyCharges": monthly_charges,
                "TotalCharges": f"{total_charges:.2f}",
                "Churn": (
                    "Yes"
                    if churn
                    else "No"
                ),
            }
        )

    return pd.DataFrame(rows)


def test_training_pipeline_smoke(
    tmp_path,
    monkeypatch,
):
    # Import here because run_pipeline.py sets the
    # MLflow URI when imported.
    import scripts.run_pipeline as pipeline

    # --------------------------------------------------
    # 1. Create temporary raw Telco dataset
    # --------------------------------------------------

    raw_df = make_raw_telco_dataframe()

    data_path = tmp_path / "telco_smoke.csv"

    raw_df.to_csv(
        data_path,
        index=False,
    )

    # run_pipeline.py normally requests its hard-coded
    # raw CSV path. During this test only, redirect that
    # request to our temporary CSV while still using the
    # real load_data() implementation.
    def smoke_load_data(_):
        return real_load_data(
            str(data_path)
        )

    monkeypatch.setattr(
        pipeline,
        "load_data",
        smoke_load_data,
    )

    # --------------------------------------------------
    # 2. Isolate Great Expectations
    # --------------------------------------------------

    # validate_data() uses a file-based GX context.
    # Run that context inside tmp_path so the smoke test
    # cannot modify the repository's real gx/ directory.
    def isolated_validate_data(df):
        original_cwd = os.getcwd()

        try:
            os.chdir(tmp_path)
            return real_validate_data(df)
        finally:
            os.chdir(original_cwd)

    monkeypatch.setattr(
        pipeline,
        "validate_data",
        isolated_validate_data,
    )

    # --------------------------------------------------
    # 3. Use temporary MLflow tracking
    # --------------------------------------------------

    previous_tracking_uri = (
        mlflow.get_tracking_uri()
    )

    mlflow_db = (
        tmp_path
        / "mlflow_smoke.db"
    )

    smoke_tracking_uri = (
        f"sqlite:///{mlflow_db.as_posix()}"
    )

    mlflow.set_tracking_uri(
        smoke_tracking_uri
    )
    # --------------------------------------------------
    # 4. Reduce Optuna from 100 trials to 1
    # --------------------------------------------------

    import src.tune as tune_module

    monkeypatch.setattr(
        tune_module,
        "N_TRIALS",
        1,
    )

    monkeypatch.setattr(
        pipeline,
        "N_TRIALS",
        1,
    )

    # --------------------------------------------------
    # 5. Prevent Registry modification
    # --------------------------------------------------

    # The artifact itself is still really serialized by
    # MLflow. We remove only registered_model_name so no
    # Registry version can be created.
    original_log_model = (
        mlflow.pyfunc.log_model
    )

    captured = {}

    def log_model_without_registry(
        *args,
        **kwargs,
    ):
        kwargs.pop(
            "registered_model_name",
            None,
        )

        model_info = original_log_model(
            *args,
            **kwargs,
        )

        captured["model_uri"] = (
            model_info.model_uri
        )

        # run_pipeline.py expects this attribute because
        # production runs normally register the model.
        return SimpleNamespace(
            model_uri=model_info.model_uri,
            registered_model_version=None,
        )

    monkeypatch.setattr(
        pipeline.mlflow.pyfunc,
        "log_model",
        log_model_without_registry,
    )

    try:
        # --------------------------------------------------
        # 6. Execute the REAL orchestration
        # --------------------------------------------------

        results = pipeline.main()

        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1
        assert 0 <= results["f1_score"] <= 1

        assert "model_uri" in captured
                # --------------------------------------------------
        # 7. Verify MLflow provenance
        # --------------------------------------------------

        runs = mlflow.search_runs(
            experiment_names=["Telco-Customer-Churn"],
            order_by=["start_time DESC"],
            max_results=1,
        )

        assert len(runs) == 1

        run = runs.iloc[0]

        assert run["params.n_trials"] == "1"
        assert run["params.min_precision"] == "0.4"
        assert run["params.dataset_rows"] == "240"
        assert run["params.dataset_columns"] == "21"

        assert run["tags.dataset_source"]
        assert len(run["tags.dataset_sha256"]) == 64
        assert run["tags.git_commit"]

        # --------------------------------------------------
        # 8. Reload the artifact actually created
        # --------------------------------------------------

        loaded_model = (
            mlflow.pyfunc.load_model(
                captured["model_uri"]
            )
        )

        raw_example = (
            pipeline.build_raw_input_example(
                raw_df
            )
        )

        predictions = (
            loaded_model.predict(
                raw_example
            )
        )

        assert len(predictions) > 0

        assert {
            "churn_probability",
            "churn_prediction",
        }.issubset(
            predictions.columns
        )

        assert (
            predictions[
                "churn_probability"
            ]
            .between(0, 1)
            .all()
        )

        assert set(
            predictions[
                "churn_prediction"
            ].unique()
        ).issubset({0, 1})

    finally:
        mlflow.set_tracking_uri(
            previous_tracking_uri
        )