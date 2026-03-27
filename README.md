# 🚀 AI-Based Predictive Maintenance System

An end-to-end Machine Learning system designed to predict the **Remaining Useful Life (RUL)** of industrial engines and enable **proactive maintenance decisions**.
---

## 📌 Problem Statement
Unexpected machine failures lead to high downtime and financial loss in industries.  
This project aims to predict equipment failure in advance using sensor data, enabling **predictive maintenance instead of reactive maintenance**.
---

## 🧠 Solution Overview
The system uses a trained **Random Forest Regression model** to estimate the Remaining Useful Life (RUL) of an engine based on operational sensor data.

Based on predicted RUL, the system classifies engine health into:
- ✅ Good
- ⚠️ Maintainance required
- ❌ Critical
---

## ⚙️ System Architecture
User Input → FastAPI → ML Model → RUL Prediction → Health Classification → Response
---

## 🛠️ Tech Stack
- **Python**
- **Scikit-learn** (Random Forest)
- **FastAPI** (API deployment)
- **Docker** (containerization)
- **Pandas / NumPy** (data processing)
---
## 📊 Model Details
- Algorithm: Random Forest Regressor  
- Input: Sensor data (temperature, pressure, RPM, etc.)  
- Output: Remaining Useful Life (RUL)  
---

## 📥 API Usage
### 🔹 Endpoint:http://localhost:8080/predict

### 🔹 Request: Provide 12 correlative sensor values
json
{
  data:[25.25,35.24,....]
} 

### 🔹 Response:
{
  "Predicted_RUL": 115.94,
  "Engine_Status": "Good"
}
