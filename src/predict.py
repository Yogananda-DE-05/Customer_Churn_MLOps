import joblib
import pandas as pd

# Load model
model = joblib.load("../models/churn_model.pkl")

# Load feature names
feature_names = joblib.load("../models/feature_names.pkl")

print("Model Loaded Successfully")
print("Total Features:", len(feature_names))

# Empty dataframe with all training columns
sample_data = pd.DataFrame(
    [[0] * len(feature_names)],
    columns=feature_names
)

print(sample_data.head())

# Sample Customer Values

sample_data["SeniorCitizen"] = 0
sample_data["tenure"] = 24
sample_data["MonthlyCharges"] = 85
sample_data["TotalCharges"] = 2040

# Prediction
prediction = model.predict(sample_data)

print("Prediction:", prediction)

if prediction[0] == 1:
    print("Customer Will Churn")
else:
    print("Customer Will Not Churn")