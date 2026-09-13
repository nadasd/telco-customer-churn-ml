import os

import gradio as gr
import requests


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30,
        )

        response.raise_for_status()
        result = response.json()

        probability = float(result["churn_probability"])
        prediction = int(result["churn_prediction"])

        probability_text = f"{probability * 100:.2f}%"

        if prediction == 1:
            decision = "⚠️ Customer at risk of churn"
        else:
            decision = "✅ Customer likely to stay"

        return probability_text, decision

    except requests.RequestException as exc:
        raise gr.Error(f"Prediction API unavailable: {exc}")


with gr.Blocks(
    title="Telco Customer Churn Predictor"
) as demo:

    gr.Markdown(
        """
        # Telco Customer Churn Predictor

        Predict whether a telecom customer is likely to churn.
        """
    )

    with gr.Row():

        with gr.Column():

            gender = gr.Dropdown(
                ["Female", "Male"],
                value="Female",
                label="Gender",
            )

            senior_citizen = gr.Dropdown(
                [0, 1],
                value=0,
                label="Senior Citizen",
            )

            partner = gr.Dropdown(
                ["Yes", "No"],
                value="Yes",
                label="Partner",
            )

            dependents = gr.Dropdown(
                ["Yes", "No"],
                value="No",
                label="Dependents",
            )

            tenure = gr.Slider(
                0,
                72,
                value=12,
                step=1,
                label="Tenure (months)",
            )

            phone_service = gr.Dropdown(
                ["Yes", "No"],
                value="Yes",
                label="Phone Service",
            )

            multiple_lines = gr.Dropdown(
                ["Yes", "No", "No phone service"],
                value="No",
                label="Multiple Lines",
            )

            internet_service = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                value="Fiber optic",
                label="Internet Service",
            )

            online_security = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="No",
                label="Online Security",
            )

            online_backup = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="Yes",
                label="Online Backup",
            )

        with gr.Column():

            device_protection = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="No",
                label="Device Protection",
            )

            tech_support = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="No",
                label="Tech Support",
            )

            streaming_tv = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="Yes",
                label="Streaming TV",
            )

            streaming_movies = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="Yes",
                label="Streaming Movies",
            )

            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
                label="Contract",
            )

            paperless_billing = gr.Dropdown(
                ["Yes", "No"],
                value="Yes",
                label="Paperless Billing",
            )

            payment_method = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                value="Electronic check",
                label="Payment Method",
            )

            monthly_charges = gr.Number(
                value=85.50,
                label="Monthly Charges",
            )

            total_charges = gr.Number(
                value=1026.00,
                label="Total Charges",
            )

    predict_button = gr.Button(
        "Predict Churn",
        variant="primary",
    )

    with gr.Row():

        churn_probability = gr.Textbox(
            label="Churn Probability",
            interactive=False,
        )

        churn_prediction = gr.Textbox(
            label="Prediction",
            interactive=False,
        )

    predict_button.click(
        fn=predict_churn,
        inputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges,
        ],
        outputs=[
            churn_probability,
            churn_prediction,
        ],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )