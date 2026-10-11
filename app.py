import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PayGuard", layout="centered")
st.title("💰 PayGuard — Invoice Payment Risk Predictor")
st.write("Flag invoices likely to be paid late, before the due date.")

# Load the saved model files from train_model.py
model = joblib.load("models/payguard_model.pkl")
le_industry = joblib.load("models/industry_encoder.pkl")
feature_names = joblib.load("models/feature_names.pkl")
INDUSTRIES = list(le_industry.classes_)

REQUIRED_COLUMNS = [
    "client_age_months",
    "past_late_payments",
    "invoice_amount",
    "payment_terms_days",
    "industry",
    "client_rating",
]


def risk_label(probability):
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


tab1, tab2 = st.tabs(["Single invoice", "Bulk upload (CSV)"])

# ---------- TAB 1: one invoice ----------
with tab1:
    st.subheader("Enter invoice details")
    client_age = st.slider("Client relationship age (months)", 1, 60, 12)
    past_late = st.number_input("Past late payments by this client", 0, 20, 0)
    amount = st.number_input("Invoice amount ($)", 100.0, 20000.0, 2000.0)
    terms = st.selectbox("Payment terms (days)", [15, 30, 45, 60])
    industry = st.selectbox("Industry", INDUSTRIES)
    rating = st.slider("Client reliability rating (1-5)", 1.0, 5.0, 3.5)

    if st.button("Predict payment risk"):
        row = pd.DataFrame(
            [[client_age, past_late, amount, terms,
              le_industry.transform([industry])[0], rating]],
            columns=feature_names,
        )
        probability = model.predict_proba(row)[0][1]
        level = risk_label(probability)

        if level == "High":
            st.error(f"⚠️ High risk — {probability:.1%} chance of late payment")
            st.write("**Suggested action:** follow up before the due date and consider shorter payment terms.")
        elif level == "Medium":
            st.warning(f"🟡 Medium risk — {probability:.1%} chance of late payment")
            st.write("**Suggested action:** send a friendly reminder a few days before the due date.")
        else:
            st.success(f"✅ Low risk — {probability:.1%} chance of late payment")

# ---------- TAB 2: many invoices ----------
with tab2:
    st.subheader("Score a whole list of invoices")
    st.write("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    st.caption("Tip: to test it, upload data/invoices.csv from your project folder.")

    uploaded = st.file_uploader("Upload invoices CSV", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)

        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Your file is missing these columns: {missing}")
        else:
            # Skip rows with blanks or an unknown industry instead of crashing
            valid = data.dropna(subset=REQUIRED_COLUMNS)
            valid = valid[valid["industry"].isin(INDUSTRIES)].copy()
            skipped = len(data) - len(valid)
            if skipped > 0:
                st.warning(f"Skipped {skipped} rows with missing or unrecognised values.")

            X = valid[REQUIRED_COLUMNS].copy()
            X["industry_encoded"] = le_industry.transform(X["industry"])
            X = X[feature_names]

            valid["late_risk"] = model.predict_proba(X)[:, 1].round(3)
            valid["risk_level"] = valid["late_risk"].apply(risk_label)
            valid = valid.sort_values("late_risk", ascending=False)

            high = valid[valid["risk_level"] == "High"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Invoices scored", len(valid))
            col2.metric("High-risk invoices", len(high))
            col3.metric("Money at risk", f"${high['invoice_amount'].sum():,.0f}")

            st.subheader("Highest-risk invoices first")
            st.dataframe(valid, width="stretch")

            st.download_button(
                label="📥 Download risk report as CSV",
                data=valid.to_csv(index=False),
                file_name="payguard_risk_report.csv",
                mime="text/csv",
            )