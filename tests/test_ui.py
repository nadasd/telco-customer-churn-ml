from src.ui import app


def test_ui_prediction_calls_api(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "churn_probability": 0.645886242389679,
                "churn_prediction": 1,
                "model_uri": "/app/model",
            }

    def fake_post(url, json, timeout):
        assert url.endswith("/predict")
        assert timeout == 30
        assert len(json) == 19
        assert json["gender"] == "Female"
        return FakeResponse()

    monkeypatch.setattr(app.requests, "post", fake_post)

    probability, decision = app.predict_churn(
        "Female",
        0,
        "Yes",
        "No",
        12,
        "Yes",
        "No",
        "Fiber optic",
        "No",
        "Yes",
        "No",
        "No",
        "Yes",
        "Yes",
        "Month-to-month",
        "Yes",
        "Electronic check",
        85.50,
        1026.00,
    )

    assert probability == "64.59%"
    assert "at risk" in decision