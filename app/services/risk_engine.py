def calculate_risk_score(
    fraud_probability,
    amount,
    average_customer_amount,
    transaction_hour,
    location_change,
    device_change,
    transaction_frequency,
    previous_risk_events,
):
    """
    Calculate a 0-100 RiskGuard risk score.

    The score combines the ML model probability
    with interpretable behavioral and security signals.
    """

    score = 0.0

    # --------------------------------------------------
    # 1. ML MODEL SIGNAL — maximum 50 points
    # --------------------------------------------------

    ml_score = fraud_probability * 50

    score += ml_score

    # --------------------------------------------------
    # 2. AMOUNT DEVIATION — maximum 15 points
    # --------------------------------------------------

    amount_deviation = (
        amount / max(average_customer_amount, 1)
    )

    if amount_deviation >= 5:
        score += 15
    elif amount_deviation >= 3:
        score += 10
    elif amount_deviation >= 2:
        score += 5

    # --------------------------------------------------
    # 3. TRANSACTION VELOCITY — maximum 10 points
    # --------------------------------------------------

    if transaction_frequency >= 7:
        score += 10
    elif transaction_frequency >= 5:
        score += 7
    elif transaction_frequency >= 3:
        score += 3

    # --------------------------------------------------
    # 4. LOCATION CHANGE — maximum 8 points
    # --------------------------------------------------

    if location_change == 1:
        score += 8

    # --------------------------------------------------
    # 5. DEVICE CHANGE — maximum 8 points
    # --------------------------------------------------

    if device_change == 1:
        score += 8

    # --------------------------------------------------
    # 6. PREVIOUS RISK EVENTS — maximum 6 points
    # --------------------------------------------------

    if previous_risk_events >= 3:
        score += 6
    elif previous_risk_events == 2:
        score += 4
    elif previous_risk_events == 1:
        score += 2

    # --------------------------------------------------
    # 7. UNUSUAL TRANSACTION TIME — maximum 3 points
    # --------------------------------------------------

    if transaction_hour < 5:
        score += 3

    # Keep score inside 0-100
    score = min(round(score, 2), 100.0)

    # Risk classification
    if score >= 70:
        risk_level = "HIGH"
    elif score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "amount_deviation": round(amount_deviation, 2),
    }


if __name__ == "__main__":

    result = calculate_risk_score(
        fraud_probability=0.9978,
        amount=85000,
        average_customer_amount=12000,
        transaction_hour=2,
        location_change=1,
        device_change=1,
        transaction_frequency=6,
        previous_risk_events=2,
    )

    print("RiskGuard Risk Assessment")
    print("-------------------------")
    print(f"Risk Score: {result['risk_score']}/100")
    print(f"Risk Level: {result['risk_level']}")
    print(
        f"Amount Deviation: "
        f"{result['amount_deviation']}x"
    )