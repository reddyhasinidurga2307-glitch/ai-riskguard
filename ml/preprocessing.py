from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"


def load_data():
    """Load the transaction dataset."""

    df = pd.read_csv(DATA_PATH)

    return df


def prepare_data(df):
    """Prepare features and target for model training."""

    # Target
    y = df["is_fraud"]

    # Remove identifiers and target from model features
    X = df.drop(
        columns=[
            "transaction_id",
            "customer_id",
            "is_fraud",
        ]
    )

    # Categorical and numerical features
    categorical_features = [
        "merchant_category"
    ]

    numerical_features = [
        "amount",
        "transaction_hour",
        "location_change",
        "device_change",
        "transaction_frequency",
        "average_customer_amount",
        "previous_risk_events",
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    # Stratified split keeps the fraud/legitimate ratio similar
    # in both training and testing datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":
    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = prepare_data(df)

    print("Data preprocessing successful.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training fraud cases: {y_train.sum()}")
    print(f"Testing fraud cases: {y_test.sum()}")