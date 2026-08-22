def generate_evidence(
    amount,
    average_customer_amount,
    transaction_hour,
    location_change,
    device_change,
    transaction_frequency,
    previous_risk_events,
):
    """
    Generate deterministic risk evidence from transaction signals.

    The evidence produced here is based only on known
    transaction data. The AI agent will later explain
    this evidence without inventing new facts.
    """

    evidence = []

    # Amount deviation
    amount_deviation = (
        amount / max(average_customer_amount, 1)
    )

    if amount_deviation >= 5:
        evidence.append(
            f"Transaction amount is "
            f"{amount_deviation:.2f}x the customer's average."
        )
    elif amount_deviation >= 2:
        evidence.append(
            f"Transaction amount is "
            f"{amount_deviation:.2f}x the customer's average."
        )

    # Unusual transaction time
    if transaction_hour < 5:
        evidence.append(
            f"Transaction occurred at {transaction_hour:02d}:00, "
            "which is an unusual overnight transaction time."
        )

    # Location change
    if location_change == 1:
        evidence.append(
            "A new or unusual transaction location was detected."
        )

    # Device change
    if device_change == 1:
        evidence.append(
            "A new or unrecognized device was detected."
        )

    # Transaction velocity
    if transaction_frequency >= 7:
        evidence.append(
            f"High transaction velocity detected: "
            f"{transaction_frequency} recent transactions."
        )
    elif transaction_frequency >= 5:
        evidence.append(
            f"Elevated transaction velocity detected: "
            f"{transaction_frequency} recent transactions."
        )

    # Previous risk events
    if previous_risk_events >= 3:
        evidence.append(
            f"{previous_risk_events} previous risk events "
            "are associated with this customer."
        )
    elif previous_risk_events > 0:
        evidence.append(
            f"{previous_risk_events} previous risk event(s) "
            "are associated with this customer."
        )

    return evidence


if __name__ == "__main__":

    evidence = generate_evidence(
        amount=85000,
        average_customer_amount=12000,
        transaction_hour=2,
        location_change=1,
        device_change=1,
        transaction_frequency=6,
        previous_risk_events=2,
    )

    print("RiskGuard Evidence")
    print("------------------")

    for number, item in enumerate(evidence, start=1):
        print(f"{number}. {item}")