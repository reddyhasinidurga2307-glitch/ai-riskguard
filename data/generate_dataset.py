import random
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducibility
random.seed(42)
np.random.seed(42)

NUM_TRANSACTIONS = 10000


def generate_transaction(transaction_number):
    customer_id = f"C{random.randint(1000, 1999)}"

    # Normal transaction amount for the customer
    average_customer_amount = round(
        np.random.lognormal(mean=8.0, sigma=0.7), 2
    )

    # Most transactions are close to the customer's normal spending
    amount = round(
        np.random.lognormal(
            mean=np.log(max(average_customer_amount, 1)),
            sigma=0.45
        ),
        2
    )

    transaction_hour = random.randint(0, 23)

    location_change = np.random.choice(
        [0, 1],
        p=[0.90, 0.10]
    )

    device_change = np.random.choice(
        [0, 1],
        p=[0.93, 0.07]
    )

    transaction_frequency = max(
        1,
        int(np.random.poisson(lam=2))
    )

    merchant_category = random.choice([
        "grocery",
        "electronics",
        "travel",
        "restaurant",
        "online_services",
        "jewelry",
        "cash_withdrawal"
    ])

    previous_risk_events = np.random.choice(
        [0, 1, 2, 3],
        p=[0.75, 0.15, 0.07, 0.03]
    )

    # Behavioral risk signals
    amount_deviation = amount / max(average_customer_amount, 1)

    risk_points = 0

    if amount_deviation > 4:
        risk_points += 3
    elif amount_deviation > 2.5:
        risk_points += 1

    if transaction_hour < 5:
        risk_points += 2

    if location_change == 1:
        risk_points += 2

    if device_change == 1:
        risk_points += 2

    if transaction_frequency >= 5:
        risk_points += 3
    elif transaction_frequency >= 3:
        risk_points += 1

    if previous_risk_events >= 2:
        risk_points += 2
    elif previous_risk_events == 1:
        risk_points += 1

    if merchant_category == "jewelry" and amount > 50000:
        risk_points += 2

    # Convert risk signals into a fraud probability
    fraud_probability = 0.02 + (risk_points * 0.08)

    fraud_probability = min(fraud_probability, 0.95)

    is_fraud = np.random.binomial(
        1,
        fraud_probability
    )

    return {
        "transaction_id": f"TXN{100000 + transaction_number}",
        "customer_id": customer_id,
        "amount": amount,
        "transaction_hour": transaction_hour,
        "location_change": location_change,
        "device_change": device_change,
        "transaction_frequency": transaction_frequency,
        "average_customer_amount": average_customer_amount,
        "merchant_category": merchant_category,
        "previous_risk_events": previous_risk_events,
        "is_fraud": is_fraud
    }


def main():
    transactions = [
        generate_transaction(i)
        for i in range(1, NUM_TRANSACTIONS + 1)
    ]

    df = pd.DataFrame(transactions)

    output_path = Path(__file__).resolve().parent / "transactions.csv"

    df.to_csv(output_path, index=False)

    print("Dataset generated successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {output_path}")
    print("\nFraud distribution:")
    print(df["is_fraud"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()