import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Resolve absolute paths relative to app.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

# Load model and scaler on startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading artifacts from {BASE_DIR}: {e}")

# Expected model feature inputs
REQUIRED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid or missing JSON payload."}), 400

    # Support single JSON object or list of JSON objects
    is_batch = isinstance(data, list)
    records = data if is_batch else [data]

    # Validate incoming features
    missing_info = []
    for idx, record in enumerate(records):
        missing = [feat for feat in REQUIRED_FEATURES if feat not in record]
        if missing:
            missing_info.append({"index": idx, "missing_features": missing})

    if missing_info:
        return jsonify({
            "error": "Missing required input features",
            "details": missing_info,
            "expected_features": REQUIRED_FEATURES
        }), 400

    try:
        # 1. Convert payload to DataFrame with proper feature ordering
        input_df = pd.DataFrame(records)[REQUIRED_FEATURES]

        # 2. Scale features using scaler.joblib
        scaled_features = scaler.transform(input_df)

        # 3. Predict using scaled data
        predictions = model.predict(scaled_features).tolist()
        probabilities = model.predict_proba(scaled_features).tolist()

        formatted_results = [
            {
                "prediction": int(pred),
                "probability": {
                    "class_0": round(float(prob[0]), 4),
                    "class_1": round(float(prob[1]), 4)
                }
            }
            for pred, prob in zip(predictions, probabilities)
        ]

        return jsonify({
            "status": "success",
            "results": formatted_results if is_batch else formatted_results[0]
        }), 200

    except Exception as err:
        return jsonify({"error": f"Prediction failed: {str(err)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)