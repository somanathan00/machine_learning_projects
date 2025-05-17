import os
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Base directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "customer_churn_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "encoder.pkl")

# Load model and encoders
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features_names"]

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

# Input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    for column, encoder in encoders.items():
        if column == "Churn":
            continue
        input_df[column] = encoder.transform(input_df[column])

    prediction = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][0]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "probability": round(pred_prob, 4)
    }
