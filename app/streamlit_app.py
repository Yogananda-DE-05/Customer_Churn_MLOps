import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.title("Customer Churn Prediction")

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    value=24
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=85.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=2040.0
)

if st.button("Predict"):

    sample = pd.DataFrame(
        [[0] * len(feature_names)],
        columns=feature_names
    )

    sample["SeniorCitizen"] = senior
    sample["tenure"] = tenure
    sample["MonthlyCharges"] = monthly
    sample["TotalCharges"] = total

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("Customer Will Churn")
    else:
        st.success("Customer Will Not Churn")