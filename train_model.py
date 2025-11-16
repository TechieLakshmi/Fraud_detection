# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_fscore_support
import joblib
import os


from data_gen import generate_synthetic_fraud


MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)


# 1) Generate dataset
print('Generating dataset')
df = generate_synthetic_fraud(n_samples=20000, n_features=12, fraud_ratio=0.02)


# 2) Feature selection
TARGET = 'is_fraud'
categorical_cols = ['merchant_id', 'device_type']
numeric_cols = [c for c in df.columns if c.startswith('feat_') or c in ['transaction_amount', 'day']]


# 3) Train/test split
X = df[categorical_cols + numeric_cols]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# 4) Preprocessing pipeline
preprocessor = ColumnTransformer(
transformers=[
('num', StandardScaler(), numeric_cols),
('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols),
]
)


# 5) Supervised classifier pipeline (RandomForest)
clf_pipeline = Pipeline(steps=[
('pre', preprocessor),
('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))
])


print('Training supervised model')
clf_pipeline.fit(X_train, y_train)


# Evaluate
print('Evaluating')
print('Saved models to', MODEL_DIR)