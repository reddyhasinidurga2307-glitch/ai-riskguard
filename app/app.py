from flask import Flask, render_template, request
import joblib
import pandas as pd

from app.services.risk_engine import calculate_risk_score
from app.services.evidence import generate_evidence
from app.agents.risk_agent import build_investigation
from app.services.transaction_monitor import (
    monitor_transactions,
    monitor_transaction_batch,
    generate_monitoring_summary
)


app = Flask(__name__)


# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

MODEL_PATH = "models/risk_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Automatic monitoring state
# --------------------------------------------------

_monitoring_results = []

_next_transaction_index = 0

BATCH_SIZE = 100


# --------------------------------------------------
# Get current monitoring results
# --------------------------------------------------

def get_monitoring_results():

    global _monitoring_results
    global _next_transaction_index

    # If no transactions have been processed yet,
    # automatically process the first batch.

    if not _monitoring_results:

        print()
        print("Running automatic transaction monitoring...")
        print("Processing first transaction batch...")
        print()

        _monitoring_results = monitor_transaction_batch(
            batch_size=BATCH_SIZE,
            start_index=0
        )

        # The next batch must start after the
        # transactions we just processed.

        _next_transaction_index = len(
            _monitoring_results
        )

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

        # ------------------------------------------
        # Read transaction data
        # ------------------------------------------

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

    results = get_monitoring_results()


    # ----------------------------------------------
    # Generate dashboard statistics
    # ----------------------------------------------

    summary = generate_monitoring_summary(
        results
    )


    # ----------------------------------------------
    # Add progress information
    # ----------------------------------------------

    try:

        total_available = len(
            pd.read_csv(
                "data/transactions.csv"
            )
        )

    except Exception:

        total_available = len(
            results
        )


    processed_count = len(
        results
    )


    if total_available > 0:

        progress = round(
            (
                processed_count /
                total_available
            ) * 100,
            1
        )

    else:

        progress = 0


    # Add progress to summary

    summary["progress"] = progress

    summary["total_available"] = (
        total_available
    )

    summary["processed_count"] = (
        processed_count
    )


    # ----------------------------------------------
    # Sort by highest risk score
    # ----------------------------------------------

    sorted_results = sorted(
        results,
        key=lambda x: x["risk_score"],
        reverse=True
    )


    # Display only top 20

    top_results = sorted_results[:20]


    return render_template(
        "monitor.html",
        results=top_results,
        summary=summary
    )


# --------------------------------------------------
# Process Next Transaction Batch
# --------------------------------------------------

@app.route("/process-batch")
def process_batch():

    global _monitoring_results
    global _next_transaction_index


    # ----------------------------------------------
    # Determine total available transactions
    # ----------------------------------------------

    try:

        total_available = len(
            pd.read_csv(
                "data/transactions.csv"
            )
        )

    except Exception:

        total_available = 0


    # ----------------------------------------------
    # Check whether all transactions are processed
    # ----------------------------------------------

    if (
        _next_transaction_index
        >= total_available
    ):

        return monitor()


    # ----------------------------------------------
    # Process next batch
    # ----------------------------------------------

    print()
    print(
        "Processing next transaction batch..."
    )

    print(
        f"Starting from transaction index: "
        f"{_next_transaction_index}"
    )

    print()


    batch_results = monitor_transaction_batch(
        batch_size=BATCH_SIZE,
        start_index=_next_transaction_index
    )


    # ----------------------------------------------
    # Add new results
    # ----------------------------------------------

    _monitoring_results.extend(
        batch_results
    )


    # ----------------------------------------------
    # Move pointer forward
    # ----------------------------------------------

    _next_transaction_index += len(
        batch_results
    )


    print(
        f"Batch processed: "
        f"{len(batch_results)} transactions"
    )

    print(
        f"Total processed: "
        f"{len(_monitoring_results)}"
    )

    print()


    # ----------------------------------------------
    # Generate updated statistics
    # ----------------------------------------------

    summary = generate_monitoring_summary(
        _monitoring_results
    )


    # ----------------------------------------------
    # Calculate progress
    # ----------------------------------------------

    processed_count = len(
        _monitoring_results
    )


    if total_available > 0:

        progress = round(
            (
                processed_count /
                total_available
            ) * 100,
            1
        )

    else:

        progress = 0


    summary["progress"] = progress

    summary["total_available"] = (
        total_available
    )

    summary["processed_count"] = (
        processed_count
    )


    # ----------------------------------------------
    # Sort highest-risk transactions first
    # ----------------------------------------------

    sorted_results = sorted(
        _monitoring_results,
        key=lambda x: x["risk_score"],
        reverse=True
    )


    top_results = sorted_results[:20]


    return render_template(
        "monitor.html",
        results=top_results,
        summary=summary
    )


# --------------------------------------------------
# Transaction Investigation
# --------------------------------------------------

@app.route(
    "/investigate/<transaction_id>"
)
def investigate(transaction_id):

    # Reuse already processed transactions.
    #
    # IMPORTANT:
    # This does NOT run the ML model again.

    results = get_monitoring_results()


    # ----------------------------------------------
    # Find requested transaction
    # ----------------------------------------------

    transaction = next(
        (
            result
            for result in results
            if result["transaction_id"]
            == transaction_id
        ),
        None
    )


    # ----------------------------------------------
    # Transaction does not exist
    # ----------------------------------------------

    if transaction is None:

        return (
            "Transaction not found",
            404
        )


    # ----------------------------------------------
    # Display complete investigation
    # ----------------------------------------------

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