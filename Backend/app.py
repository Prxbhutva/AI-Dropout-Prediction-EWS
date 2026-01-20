from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ Import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)   # ✅ Allow all cross-origin requests (needed for frontend on another PC)

# ----------------------------
# Load model & scaler safely
# ----------------------------
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("❌ model.pkl or scaler.pkl not found! Please train and save them first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON from frontend
        data = request.json
        attendance = float(data["Attendance"])
        cgpa = float(data["CGPA"])
        backlogs = int(data["Backlogs"])

        # Preprocess input
        features = np.array([[attendance, cgpa, backlogs]])
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Convert numeric prediction to label
        if prediction == 0:
            risk_label = "Low Risk"
        elif prediction == 1:
            risk_label = "Medium Risk"
        else:
            risk_label = "High Risk"

        return jsonify({
            "prediction": int(prediction),
            "risk_label": risk_label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Run Flask server
# ----------------------------
if __name__ == "__main__":
    # Run on all network interfaces so frontend PC can access it
    app.run(host="0.0.0.0", port=5000, debug=True)
