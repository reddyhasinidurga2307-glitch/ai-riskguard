from flask import Flask, render_template, request
import joblib
import pandas as pd

from app.services.risk_engine import calculate_risk_score
from app.services.evidence import generate_evidence
from app.agents.risk_agent import build_investigation
from app.services.transaction_monitor import (
    monitor_transactions,
    generate_monitoring_summary
)


app = Flask(__name__)


# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

MODEL_PATH = "models/risk_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Automatic monitoring cache
# --------------------------------------------------

_monitoring_results = None


def get_monitoring_results():
    """
    Run automatic monitoring only once.

    The results are reused by /monitor and
    /investigate/<transaction_id>.
    """

    global _monitoring_results

    if _monitoring_results is None:

        print()
        print("Running automatic transaction monitoring...")
        print("Processing transactions for the first time...")
        print()

        _monitoring_results = monitor_transactions()

        print()
        print(
            f"Monitoring complete: "
            f"{len(_monitoring_results)} transactions processed."
        )
        print()

    return _monitoring_results


# --------------------------------------------------
# Home page - Manual Transaction Assessment
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        amount = float(
            request.form["amount"]
        )

        transaction_hour = int(
            request.form["transaction_hour"]
        )

        location_change = int(
            request.form["location_change"]
        )

        device_change = int(
            request.form["device_change"]
        )

        transaction_frequency = int(
            request.form["transaction_frequency"]
        )

        average_customer_amount = float(
            request.form["average_customer_amount"]
        )

        previous_risk_events = int(
            request.form["previous_risk_events"]
        )

        merchant_category = request.form[
            "merchant_category"
        ]


        # ------------------------------------------
        # Create ML input
        # ------------------------------------------

        transaction = pd.DataFrame([
            {
                "amount": amount,

                "transaction_hour":
                    transaction_hour,

                "location_change":
                    location_change,

                "device_change":
                    device_change,

                "transaction_frequency":
                    transaction_frequency,

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

            transaction_hour=
                transaction_hour,

            location_change=
                location_change,

            device_change=
                device_change,

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

            transaction_hour=
                transaction_hour,

            location_change=
                location_change,

            device_change=
                device_change,

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


        # ------------------------------------------
        # Final result
        # ------------------------------------------

        result = {

            "fraud_probability":
                round(
                    fraud_probability * 100,
                    2
                ),

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


# --------------------------------------------------
# Automatic Transaction Monitoring
# --------------------------------------------------

@app.route("/monitor")
def monitor():

    # Get cached monitoring results

    results = get_monitoring_results()


    # Generate dashboard statistics

    summary = generate_monitoring_summary(
        results
    )


    # Sort by highest risk score

    sorted_results = sorted(
        results,
        key=lambda x: x["risk_score"],
        reverse=True
    )


    # Display only the 20 highest-risk transactions

    top_results = sorted_results[:20]


    return render_template(
        "monitor.html",
        results=top_results,
        summary=summary
    )


# --------------------------------------------------
# Transaction Investigation
# --------------------------------------------------

@app.route("/investigate/<transaction_id>")
def investigate(transaction_id):

    # Reuse already processed transactions.
    #
    # IMPORTANT:
    # This does NOT run the ML model again.

    results = get_monitoring_results()


    # Find requested transaction

    transaction = next(
        (
            result
            for result in results
            if result["transaction_id"]
            == transaction_id
        ),
        None
    )


    # Transaction does not exist

    if transaction is None:

        return "Transaction not found", 404


    # Display complete investigation

    return render_template(
        "investigate.html",
        transaction=transaction
    )


# --------------------------------------------------
# Start Flask application
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )