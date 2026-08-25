import os
import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("📊 Customer Churn Prediction")

st.info(
    """
    This application predicts whether a telecom customer is likely to churn
    based on customer demographics, billing information, and contract details.
    """
)

# =========================
# MODEL PERFORMANCE
# =========================

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", "79.35%")
    st.metric("Precision", "65.65%")

with col2:
    st.metric("Recall", "46.11%")
    st.metric("F1 Score", "54.17%")

# =========================
# FEATURE IMPORTANCE
# =========================

st.subheader("Top Important Features")

image_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "feature_importance.png"
)

if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning("Feature importance chart not found.")

# =========================
# CUSTOMER INPUTS
# =========================

st.subheader("Customer Details")

senior_option = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

senior = 1 if senior_option == "Yes" else 0

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

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

partner = st.selectbox(
    "Has Partner",
    ["No", "Yes"]
)

fiber = st.selectbox(
    "Fiber Internet",
    ["No", "Yes"]
)

contract = st.selectbox(
    "Two Year Contract",
    ["No", "Yes"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

# =========================
# CONVERT TO MODEL VALUES
# =========================

gender_male = 1 if gender == "Male" else 0
partner_yes = 1 if partner == "Yes" else 0
internet_fiber = 1 if fiber == "Yes" else 0
contract_two_year = 1 if contract == "Yes" else 0
paperless_billing = 1 if paperless == "Yes" else 0

# =========================
# PREDICT BUTTON
# =========================

predict_btn = st.button(
    "🔍 Predict Churn",
    use_container_width=True
)

if predict_btn:

    sample = pd.DataFrame(
        [[0] * len(feature_names)],
        columns=feature_names
    )

    # Core Features
    if "SeniorCitizen" in feature_names:
        sample["SeniorCitizen"] = senior

    if "tenure" in feature_names:
        sample["tenure"] = tenure

    if "MonthlyCharges" in feature_names:
        sample["MonthlyCharges"] = monthly

    if "TotalCharges" in feature_names:
        sample["TotalCharges"] = total

    # Encoded Features
    if "gender_Male" in feature_names:
        sample["gender_Male"] = gender_male

    if "Partner_Yes" in feature_names:
        sample["Partner_Yes"] = partner_yes

    if "InternetService_Fiber optic" in feature_names:
        sample["InternetService_Fiber optic"] = internet_fiber

    if "Contract_Two year" in feature_names:
        sample["Contract_Two year"] = contract_two_year

    if "PaperlessBilling_Yes" in feature_names:
        sample["PaperlessBilling_Yes"] = paperless_billing

    # Prediction
    prediction = model.predict(sample)

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer Will Churn")
    else:
        st.success("✅ Customer Will Stay")

    if probability is not None:
        st.metric(
    "Churn Probability",
    f"{probability:.2%}"
)

        if probability >= 0.70:
            st.error("High Risk Customer")
        elif probability >= 0.40:
             st.warning("Medium Risk Customer")
        else:
             st.success("Low Risk Customer")

    with st.expander("View Input Data Used"):
        st.dataframe(sample.T)

        st.markdown("---")
st.caption(
    "Built using Python, Scikit-Learn, Pandas and Streamlit"
)