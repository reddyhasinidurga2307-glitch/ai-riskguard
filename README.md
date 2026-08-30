# AI RiskGuard
🚀 **[Live Demo](https://ai-riskguard.onrender.com)**

## Intelligent Transaction Risk Management & Investigation System

AI RiskGuard is a machine-learning-powered financial transaction risk management system that analyzes transactions, estimates fraud probability, calculates an overall risk score, generates risk evidence, and supports investigation decisions through a unified monitoring workflow.

The system combines **Machine Learning, Risk Scoring, Evidence Generation, Transaction Monitoring, and Investigation** into one application.

---

## Problem Statement

Financial transaction systems need to identify potentially suspicious transactions quickly and consistently.

Manual transaction review can become difficult when the number of transactions increases. RiskGuard provides an automated decision-support workflow that helps identify higher-risk transactions and prioritize them for investigation.

---

## Solution

AI RiskGuard analyzes transaction characteristics using a trained machine learning model and a dedicated risk engine.

For each analyzed transaction, the system can produce:

- ML-based fraud probability
- Risk score out of 100
- Risk classification
- Risk evidence
- Investigation summary
- Recommended action
- Transaction monitoring information
- Batch transaction processing
- Risk monitoring dashboard
- Transaction investigation details

---

## Key Features

### 1. Machine Learning Fraud Detection

The trained machine learning model analyzes transaction characteristics and produces a fraud probability.

The transaction features used by the system include:

- Transaction amount
- Transaction hour
- Location change
- Device change
- Transaction frequency
- Average customer transaction amount
- Merchant category
- Previous risk events

---

### 2. Risk Scoring Engine

The ML fraud probability is combined with transaction-level risk factors through the Risk Engine.

The system produces:

```text
Fraud Probability
Risk Score
Risk Level
Risk Factors
```
---

## Installation & Running

### 1. Clone the repository

```bash
git clone https://github.com/reddyhasinidurga2307-glitch/ai-riskguard.git
cd ai-riskguard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m app.app
```

The application will start at:

```text
http://127.0.0.1:5001
```

Open that address in your browser.

---

## System Workflow

```text
Transaction Data
       |
Machine Learning Model
       |
Fraud Probability
       |
Risk Scoring Engine
       |
Risk Score + Risk Level
       |
Risk Evidence
       |
Monitoring Dashboard
       |
Transaction Investigation
       |
Recommended Action
+```
---

## Intended Users

AI RiskGuard is designed primarily for **financial risk analysts and fraud investigation teams**.

The system helps analysts:

- Monitor processed transactions
- Identify high-risk transactions
- Understand why a transaction was flagged
- Investigate suspicious transactions
- Prioritize transactions for review
- Support risk-based decision making

---

## Important Note

AI RiskGuard is a **decision-support system**. It assists analysts by identifying potentially risky transactions and providing supporting information. Final decisions remain with the responsible financial institution or analyst.
