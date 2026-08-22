from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import load_data, prepare_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "risk_model.joblib"


def train_model():
    """Train and evaluate the baseline risk model."""

    # Load dataset
    df = load_data()

    # Prepare training and testing data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = prepare_data(df)

    # Build complete ML pipeline
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # Train
    print("Training RiskGuard baseline model...")

    model.fit(
        X_train,
        y_train,
    )

    # Predictions
    y_pred = model.predict(X_test)

    # Probability of fraud
    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Fraud",
            ],
        )
    )

    print("===== CONFUSION MATRIX =====")

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    print("\n===== ROC-AUC =====")

    auc = roc_auc_score(
        y_test,
        y_probability,
    )

    print(f"ROC-AUC: {auc:.4f}")

    # Save trained pipeline
    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\n===== MODEL SAVED =====")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_model()