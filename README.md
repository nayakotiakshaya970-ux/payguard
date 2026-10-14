# PayGuard — Invoice Payment Risk Predictor

## Problem
Businesses lose cash-flow predictability because they only discover a
payment problem after an invoice is already overdue, not before.

## Solution
A machine learning tool that scores every outstanding invoice by its
probability of late payment, so finance teams can follow up early.

## Features
- Single-invoice risk check with a suggested action (High / Medium / Low)
- Bulk CSV upload that ranks all invoices from riskiest to safest
- "Money at risk" total for high-risk invoices
- Skips bad rows with a warning instead of crashing
- Downloadable risk report as CSV

## Approach
- Simulated a realistic dataset of 2,000 invoices (no public dataset exists)
- Trained a Random Forest classifier with class balancing
- F1-score on unseen test data: 0.71 (recall 0.84)
- Top risk factors: past late payments, invoice amount, client rating

## How to run
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py

## Tech stack
Python, Pandas, scikit-learn, Streamlit, Faker

## Live Demo
https://payguard-akshaya.streamlit.app/

## Future improvements
- Connect to real accounting data (QuickBooks / Xero exports)
- Automatic email reminders for high-risk invoices
- Track prediction accuracy over time