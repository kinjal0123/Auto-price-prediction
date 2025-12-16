from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

# -----------------------------
# Load Saved Model & Columns
# -----------------------------
model = pickle.load(open("Model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

# Optional: load scaler if used during training
# scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Preprocess Input
# -----------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # -----------------------------
    # Label Encoding (safe mapping)
    # -----------------------------
    df['fuel-type'] = df['fuel-type'].map({'diesel': 0, 'gas': 1}).fillna(0)
    df['aspiration'] = df['aspiration'].map({'std': 1, 'turbo': 0}).fillna(0)

    # -----------------------------
    # Ordinal Encoding for Cylinders
    # -----------------------------
    cyl_map = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'eight': 8, 'twelve': 12
    }
    df['num-of-cylinders'] = df['num-of-cylinders'].map(cyl_map).fillna(4)  # default 4 cylinders

    # -----------------------------
    # One-Hot Encode categorical columns
    # -----------------------------
    df = pd.get_dummies(df)

    # -----------------------------
    # Add missing columns & align order
    # -----------------------------
    df = df.reindex(columns=model_columns, fill_value=0)

    return df

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.json
        processed = preprocess_input(data)

        # Optional: scale input if scaler used
        # processed_scaled = scaler.transform(processed)
        # prediction = model.predict(processed_scaled)[0]

        prediction = model.predict(processed)[0]

        return jsonify({
            "status": "success",
            "predicted_price": float(prediction)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# -----------------------------
# Home Route
# -----------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
