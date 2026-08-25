import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
feature_path = os.path.join(BASE_DIR, "models", "feature_names.pkl")

model = joblib.load(model_path)
feature_names = joblib.load(feature_path)

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and feature names
model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")


@app.route("/")
def home():
    return """
    <h2>Customer Churn Prediction API</h2>
    <p>API is running successfully.</p>
    <p>Use POST /predict to get predictions.</p>
    <p>Use GET /test for quick testing.</p>
    """


@app.route("/test")
def test():

    sample = pd.DataFrame(
        [[0] * len(feature_names)],
        columns=feature_names
    )

    sample["SeniorCitizen"] = 0
    sample["tenure"] = 24
    sample["MonthlyCharges"] = 85
    sample["TotalCharges"] = 2040

    prediction = model.predict(sample)

    result = "Churn" if prediction[0] == 1 else "No Churn"

    return jsonify({
        "prediction": result
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        sample = pd.DataFrame(
            [[0] * len(feature_names)],
            columns=feature_names
        )

        # Numerical Features
        sample["SeniorCitizen"] = data.get("SeniorCitizen", 0)
        sample["tenure"] = data.get("tenure", 0)
        sample["MonthlyCharges"] = data.get("MonthlyCharges", 0)
        sample["TotalCharges"] = data.get("TotalCharges", 0)

        prediction = model.predict(sample)

        result = "Churn" if prediction[0] == 1 else "No Churn"

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)