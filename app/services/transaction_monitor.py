from pathlib import Path

import joblib
import pandas as pd

from app.services.risk_engine import calculate_risk_score
from app.services.evidence import generate_evidence
from app.agents.risk_agent import build_investigation


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "risk_model.joblib"
DATA_PATH = BASE_DIR / "data" / "transactions.csv"


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Process one transaction
# --------------------------------------------------

def process_transaction(transaction):

    amount = float(transaction["amount"])

    transaction_hour = int(
        transaction["transaction_hour"]
    )

    location_change = int(
        transaction["location_change"]
    )

    device_change = int(
        transaction["device_change"]
    )

    transaction_frequency = int(
        transaction["transaction_frequency"]
    )

    average_customer_amount = float(
        transaction["average_customer_amount"]
    )

    merchant_category = transaction[
        "merchant_category"
    ]

    previous_risk_events = int(
        transaction["previous_risk_events"]
    )


    # ----------------------------------------------
    # Create ML input
    # ----------------------------------------------

    model_input = pd.DataFrame([
        {
            "amount": amount,
            "transaction_hour": transaction_hour,
            "location_change": location_change,
            "device_change": device_change,
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


    # ----------------------------------------------
    # ML prediction
    # ----------------------------------------------

    fraud_probability = model.predict_proba(
        model_input
    )[0][1]


    # ----------------------------------------------
    # Risk Engine
    # ----------------------------------------------

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


    # ----------------------------------------------
    # Evidence generation
    # ----------------------------------------------

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


    # ----------------------------------------------
    # Investigation Agent
    # ----------------------------------------------

    investigation = build_investigation(
        risk_score=
            risk_result["risk_score"],
        risk_level=
            risk_result["risk_level"],
        evidence=evidence,
    )


    # ----------------------------------------------
    # Complete result
    # ----------------------------------------------

    return {

        "transaction_id":
            transaction["transaction_id"],

        "customer_id":
            transaction["customer_id"],

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


# --------------------------------------------------
# Process transactions
# --------------------------------------------------

def monitor_transactions(limit=None):

    df = pd.read_csv(DATA_PATH)

    if limit is not None:
        df = df.head(limit)

    results = []

    for _, transaction in df.iterrows():

        result = process_transaction(
            transaction
        )

        results.append(result)

    return results


# --------------------------------------------------
# Generate dashboard statistics
# --------------------------------------------------

def generate_monitoring_summary(results):

    total_transactions = len(results)

    low_risk = sum(
        1
        for result in results
        if result["risk_level"] == "LOW"
    )

    medium_risk = sum(
        1
        for result in results
        if result["risk_level"] == "MEDIUM"
    )

    high_risk = sum(
        1
        for result in results
        if result["risk_level"] == "HIGH"
    )

    fraud_detected = sum(
        1
        for result in results
        if result["fraud_probability"] >= 50
    )

    escalation_count = sum(
        1
        for result in results
        if result["recommended_action"]
        == "ESCALATE_FOR_REVIEW"
    )


    return {

        "total_transactions":
            total_transactions,

        "low_risk":
            low_risk,

        "medium_risk":
            medium_risk,

        "high_risk":
            high_risk,

        "fraud_detected":
            fraud_detected,

        "escalation_count":
            escalation_count,
    }


# --------------------------------------------------
# Test automatic monitoring
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("========================================")
    print("       RISKGUARD TRANSACTION MONITOR")
    print("========================================")
    print()

    results = monitor_transactions(
        limit=10
    )

    summary = generate_monitoring_summary(
        results
    )


    print(
        f"Transactions processed: "
        f"{summary['total_transactions']}"
    )

    print(
        f"Low risk: "
        f"{summary['low_risk']}"
    )

    print(
        f"Medium risk: "
        f"{summary['medium_risk']}"
    )

    print(
        f"High risk: "
        f"{summary['high_risk']}"
    )

    print(
        f"Fraud probability >= 50%: "
        f"{summary['fraud_detected']}"
    )

    print(
        f"Escalations: "
        f"{summary['escalation_count']}"
    )

    print()

    for result in results:

        print(
            result["transaction_id"],
            "|",
            result["risk_level"],
            "|",
            result["fraud_probability"],
            "%"
        )