# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)


# Load artifacts
preprocessor = joblib.load('model/preprocessor.pkl')
rf = joblib.load('model/rf_model.pkl')
iso = joblib.load('model/iso_model.pkl')


# Define expected schema (same order used in training)
CATEGORICAL = ['merchant_id', 'device_type']
NUMERIC = [c for c in [f'feat_{i}' for i in range(12)] + ['transaction_amount', 'day']]
ALL_COLS = CATEGORICAL + NUMERIC


@app.route('/predict', methods=['POST'])
def predict():
payload = request.get_json()
if not payload:
return jsonify({'error': 'Expected JSON payload'}), 400


# Accept either a dict for single record or a list for batch
if isinstance(payload, dict):
df = pd.DataFrame([payload])
elif isinstance(payload, list):
df = pd.DataFrame(payload)
else:
return jsonify({'error': 'Invalid payload format'}), 400


# Ensure missing cols are added with NaNs
for c in ALL_COLS:
if c not in df.columns:
df[c] = np.nan


X_pre = preprocessor.transform(df[CATEGORICAL + NUMERIC])


# Supervised prediction
pred_proba = rf.predict_proba(df[CATEGORICAL + NUMERIC]) if hasattr(rf, 'predict_proba') else rf.predict(df[CATEGORICAL + NUMERIC])
pred = rf.predict(df[CATEGORICAL + NUMERIC])


# Isolation Forest anomaly score (negative -> more anomalous)
iso_score = iso.decision_function(X_pre)


# Build response
results = []
for i in range(len(df)):
p = float(pred[i])
prob = float(pred_proba[i][1]) if hasattr(rf, 'predict_proba') else None
app.run(debug=True, port=5000)