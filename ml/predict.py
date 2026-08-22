from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"


def load_model():
    """Load the trained RiskGuard model."""

    return joblib.load(MODEL_PATH)


def predict_risk(transaction):
    """
    Predict fraud probability for a single transaction.

    Parameters
    ----------
    transaction : dict
        Transaction features.

    Returns
    -------
    float
        Fraud probability between 0 and 1.
    """

    model = load_model()

    df = pd.DataFrame([transaction])

    probability = model.predict_proba(df)[0][1]

    return float(probability)


if __name__ == "__main__":

    sample_transaction = {
        "amount": 85000,
        "transaction_hour": 2,
        "location_change": 1,
        "device_change": 1,
        "transaction_frequency": 6,
        "average_customer_amount": 12000,
        "merchant_category": "electronics",
        "previous_risk_events": 2,
    }

    probability = predict_risk(sample_transaction)

    print(f"Fraud probability: {probability:.4f}")
    print(f"Fraud probability: {probability * 100:.2f}%")