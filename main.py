from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
import pickle

# -----------------------------
# Load Saved Random Forest Model
# -----------------------------
model = pickle.load(open("Model.pkl", "rb"))

# Flask App
app = Flask(__name__)

# --------------------------------------
# Required columns (after preprocessing)
# --------------------------------------
all_features = [
    'fuel-type', 'aspiration', 'wheel-base', 'length', 'width', 'height',
    'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio',
    'horsepower', 'peak-rpm', 'city-mpg', 'num-of-cylinders',

    # One-Hot Encoded Columns ↓↓↓
    'make_audi', 'make_bmw', 'make_chevrolet', 'make_dodge', 'make_honda',
    'make_isuzu', 'make_jaguar', 'make_mazda', 'make_mercedes-benz',
    'make_mercury', 'make_mitsubishi', 'make_nissan', 'make_peugeot',
    'make_plymouth', 'make_porsche', 'make_renault', 'make_saab',
    'make_subaru', 'make_toyota', 'make_volkswagen', 'make_volvo',

    'body-style_convertible', 'body-style_hardtop', 'body-style_hatchback',
    'body-style_sedan', 'body-style_wagon',

    'drive-wheels_4wd', 'drive-wheels_fwd', 'drive-wheels_rwd',

    'engine-type_dohc', 'engine-type_dohcv', 'engine-type_l',
    'engine-type_ohc', 'engine-type_ohcf', 'engine-type_ohcv',
    'engine-type_rotor'
]

# ------------------------------------------------------
# Utility function: Preprocess single JSON input
# ------------------------------------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Label Encoding (same as training)
    df['fuel-type'] = df['fuel-type'].map({'diesel': 0, 'gas': 1})
    df['aspiration'] = df['aspiration'].map({'std': 1, 'turbo': 0})

    # Ordinal encoding for cylinders
    cyl_map = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'eight': 8, 'twelve': 12
    }
    df['num-of-cylinders'] = df['num-of-cylinders'].map(cyl_map)

    # One-Hot Encode categorical columns
    df = pd.get_dummies(df)

    # Add missing one-hot columns
    for col in all_features:
        if col not in df.columns:
            df[col] = 0

    # Ensure exact column order
    df = df[all_features]

    return df


# -----------------------
# Predict Endpoint
# -----------------------
@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.json

    try:
        processed = preprocess_input(data)
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


# -----------------------
# Home Route
# -----------------------
@app.route('/', methods=['GET'])
def home():
    return render_template{'index.html'}




# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
