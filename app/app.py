from flask import Flask, render_template, request
import joblib
import pandas as pd

from app.services.risk_engine import calculate_risk_score
from app.services.evidence import generate_evidence
from app.agents.risk_agent import build_investigation

app = Flask(__name__)


# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

MODEL_PATH = "models/risk_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        # ------------------------------------------
        # Read transaction data from form
        # ------------------------------------------

        amount = float(request.form["amount"])
        transaction_hour = int(request.form["transaction_hour"])
        location_change = int(request.form["location_change"])
        device_change = int(request.form["device_change"])
        transaction_frequency = int(
            request.form["transaction_frequency"]
        )
        average_customer_amount = float(
            request.form["average_customer_amount"]
        )
        previous_risk_events = int(
            request.form["previous_risk_events"]
        )
        merchant_category = request.form["merchant_category"]

        if merchant_category == "other":
            merchant_category = request.form["other_category"]

        # ------------------------------------------
        # Create transaction dataframe
        # ------------------------------------------

        transaction = pd.DataFrame([
            {
                "amount": amount,
                "transaction_hour": transaction_hour,
                "location_change": location_change,
                "device_change": device_change,
                "transaction_frequency": transaction_frequency,
                "average_customer_amount":
                    average_customer_amount,
                "merchant_category":
                    merchant_category,
                "previous_risk_events":
                    previous_risk_events,
            }
        ])

        
        # ------------------------------------------
        # ML prediction
        # ------------------------------------------

        fraud_probability = model.predict_proba(
            transaction
        )[0][1]
        # ------------------------------------------
        # Risk Engine
        # ------------------------------------------

        risk_result = calculate_risk_score(
            fraud_probability=fraud_probability,
            amount=amount,
            average_customer_amount=
                average_customer_amount,
            transaction_hour=transaction_hour,
            location_change=location_change,
            device_change=device_change,
            transaction_frequency=
                transaction_frequency,
            previous_risk_events=
                previous_risk_events,
        )

        # ------------------------------------------
        # Evidence generation
        # ------------------------------------------

        evidence = generate_evidence(
            amount=amount,
            average_customer_amount=
                average_customer_amount,
            transaction_hour=transaction_hour,
            location_change=location_change,
            device_change=device_change,
            transaction_frequency=
                transaction_frequency,
            previous_risk_events=
                previous_risk_events,
        )

        # ------------------------------------------
        # Investigation Agent
        # ------------------------------------------

        investigation = build_investigation(
            risk_score=
                risk_result["risk_score"],
            risk_level=
                risk_result["risk_level"],
            evidence=evidence,
        )

        result = {
            "fraud_probability":
                round(fraud_probability * 100, 2),

            "risk_score":
                risk_result["risk_score"],

            "risk_level":
                risk_result["risk_level"],

            "amount_deviation":
                risk_result["amount_deviation"],

            "evidence":
                investigation["evidence"],

            "summary":
                investigation["summary"],

            "recommended_action":
                investigation["recommended_action"],
        }

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(
        debug=True,
        port=5001
    )