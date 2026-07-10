# importing all the libraries and dependencies
import os
import pickle as pk
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Defining the input data structure using Pydantic
# This ensures incoming data is automatically validated and strictly typed.
class PatientData(BaseModel):
    age: int = Field(..., description = "Age of the patient")
    sex: int = Field(..., description = "1 = female; 0 = male")
    cp: int = Field(..., description = "Chest pain type (0-3)")
    trestbps: int = Field(..., description = "Resting Blood Pressure in mm Hg")
    chol: int = Field(..., description = "Serum cholesterol in mg/dl")
    fbs: int = Field(..., description = "Fasting blood sugar > 120 mg/dl (1 = True; 0 = False)")
    restecg: int = Field(..., description = "Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., description = "Maximum heart rate achieved")
    exang: int = Field(..., description = "Exercise induced angina (1 = yes; 0 = no)")
    oldpeak: float = Field(..., description = "ST depression induced by exercise relative to rest")
    slope: int = Field(..., description = "The slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., description = "Number of major vessels coloured by flourosopy (0-3)")
    thal: int = Field(..., description = "Thalassemia type (0-3)")

# Initializing the FastAPI Application
app = FastAPI(
    title = "Heart Attack Possibility API",
    description = "An API that uses a calibrated SVM classifier to predict heart attack risks.",
    version = "1.0.0"
)

# Load the model artifacts globally on startup
# Update this path to match exactly where your .pkl file is saved
ARTIFACTS_PATH = "heart_attack_model_artifacts.pkl"
if not os.path.exists(ARTIFACTS_PATH):
    raise FileNotFoundError(f"could not find model artifacts at: {ARTIFACTS_PATH}. Please run your training script first")

with open(ARTIFACTS_PATH, "rb") as file:
    artifacts = pk.load(file)

scaler = artifacts["scaler"]
model = artifacts["model"]

# Define API Endpoints
@app.get("/")
def home():
    """Root endpoint to verify the API status"""
    return{
        "status": "Healthy",
        "message": "Heart Attack Possibility Predictor API is running successfully."
    }
@app.post("/predict")
def predict_heart_attack(patient: PatientData):
    """
    Endpint to recieve raw paitent health parameters,
    preprocess them using the saved scaler, and return predictions
    """
    try:
        # # Converting incoming Pydantic object values directly into a 2D array for scikit-learn
        raw_features = [
            [
            patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
            patient.fbs, patient.restecg, patient.thalach, patient.exang,
            patient.oldpeak, patient.slope, patient.ca, patient.thal
            ]
        ]

        # Scaling the data using your saved training rules
        scaled_features = scaler.transform(raw_features)

        # Generating hard classification prediction (0 or 1) 
        prediction = int(model.predict(scaled_features)[0])

        # Generateing prediction probabilities 
        probabilities = model.predict_proba(scaled_features)[0]
        # probabilities of class 1 (high risk)
        risk_probabilities = float(probabilities[1])

        # building response dictionary
        result = "High Risk of Heart Attack" if prediction == 1 else "Low Risk of Heart Attack"

        return{
            "prediction_code": prediction,
            "prediction_label": result,
            "confidence_score": round(risk_probabilities * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Prediction error occured: {str(e)}")
    