# 🛡️ Fraud Detection System

## 🚀 Project Overview

This repository contains a modular and production-structured fraud detection pipeline. The system simulates fraud data, preprocesses features, trains both **supervised** and **unsupervised models**, and exposes a **Flask REST API** for real-time predictions.

## 🧠 Key Features

* ✔ **Synthetic Fraud Dataset Generator** (no external dataset needed)
* ✔ **Multiple Modeling Approaches**:

* RandomForestClassifier (supervised)
* IsolationForest (unsupervised anomaly detection)
  * ✔ **Feature Engineering + Preprocessing Pipeline** using ColumnTransformer
  * ✔ **REST API for predictions** using Flask
  * ✔ **Docker-ready deployment** with Dockerfile + docker-compose.yml
  * ✔ Clean, modular, interview-ready code base

---

## 🧪 How to Run the Project

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Generate synthetic data

```
python data_gen.py
```

This produces `synthetic_fraud.csv`.

### 3️⃣ Train the models

```
python train_model.py
```

This creates the `model/` folder with all trained artifacts.

### 4️⃣ Run the API

```
python app.py
```

API runs at: **[http://localhost:5000/predict](http://localhost:5000/predict)**

## 📊 Model Architecture

* **RandomForestClassifier** detects fraud using supervised learning.
* **IsolationForest** detects anomalies where fraud might be unknown.
* **Hybrid scoring**: outputs label, probability, and anomaly score.
