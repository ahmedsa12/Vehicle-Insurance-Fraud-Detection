"""
=============================================================
  Vehicle Insurance Fraud Detection - FastAPI Backend
=============================================================
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated
import pickle
import numpy as np
import os

app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Predicts if an insurance claim is fraudulent",
    version="1.0.0"
)

# ─── Load saved model artifacts ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)


# ─── Input Schema ────────────────────────────────────────────
class ClaimInput(BaseModel):
    policy_state: str = Field(..., example="CA")
    policy_deductible: int = Field(..., example=500)
    policy_annual_premium: float = Field(..., example=1200.0)
    insured_age: int = Field(..., gt=0, example=35)
    insured_sex: str = Field(..., example="MALE")
    insured_education_level: str = Field(..., example="College")
    insured_occupation: str = Field(..., example="Manager")
    insured_hobbies: str = Field(..., example="reading")
    incident_type: str = Field(..., example="Single Vehicle Collision")
    collision_type: str = Field(..., example="Front")
    incident_severity: str = Field(..., example="Major Damage")
    authorities_contacted: str = Field(..., example="Police")
    incident_state: str = Field(..., example="CA")
    incident_city: str = Field(..., example="Charlesville")
    incident_hour_of_the_day: Annotated[int, Field(ge=0, le=23)] = Field(..., example=14)
    number_of_vehicles_involved: int = Field(..., ge=1, example=1)
    bodily_injuries: int = Field(..., ge=0, example=0)
    witnesses: int = Field(..., ge=0, example=2)
    police_report_available: str = Field(..., example="Yes")
    claim_amount: float = Field(..., gt=0, example=8000.0)
    total_claim_amount: float = Field(..., gt=0, example=9500.0)


# ─── Helper: encode + engineer ────────────────────────────────
def preprocess_input(data: ClaimInput) -> np.ndarray:
    cat_cols = [
        'policy_state', 'insured_sex', 'insured_education_level',
        'insured_occupation', 'insured_hobbies', 'incident_type',
        'collision_type', 'incident_severity', 'authorities_contacted',
        'incident_state', 'incident_city', 'police_report_available'
    ]

    row = {
        'policy_state': data.policy_state,
        'policy_deductible': data.policy_deductible,
        'policy_annual_premium': data.policy_annual_premium,
        'insured_age': data.insured_age,
        'insured_sex': data.insured_sex,
        'insured_education_level': data.insured_education_level,
        'insured_occupation': data.insured_occupation,
        'insured_hobbies': data.insured_hobbies,
        'incident_type': data.incident_type,
        'collision_type': data.collision_type,
        'incident_severity': data.incident_severity,
        'authorities_contacted': data.authorities_contacted,
        'incident_state': data.incident_state,
        'incident_city': data.incident_city,
        'incident_hour_of_the_day': data.incident_hour_of_the_day,
        'number_of_vehicles_involved': data.number_of_vehicles_involved,
        'bodily_injuries': data.bodily_injuries,
        'witnesses': data.witnesses,
        'police_report_available': data.police_report_available,
        'claim_amount': data.claim_amount,
        'total_claim_amount': data.total_claim_amount,
    }

    # Encode categorical columns
    for col in cat_cols:
        le = label_encoders.get(col)
        if le is None:
            raise HTTPException(status_code=400, detail=f"No encoder for column: {col}")
        try:
            row[col] = int(le.transform([row[col]])[0])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value '{row[col]}' for '{col}'. Valid values: {list(le.classes_)}"
            )

    # Feature engineering (same as training)
    row['claim_to_premium_ratio'] = row['total_claim_amount'] / (row['policy_annual_premium'] + 1)
    row['high_claim_flag'] = 1 if row['total_claim_amount'] > 14500 else 0
    h = data.incident_hour_of_the_day
    row['night_incident'] = 1 if (h >= 20 or h <= 5) else 0
    row['many_vehicles'] = 1 if data.number_of_vehicles_involved > 2 else 0
    no_idx = int(label_encoders['police_report_available'].transform(['No'])[0])
    row['no_police_report'] = 1 if row['police_report_available'] == no_idx else 0

    # Build feature vector in correct order
    features = [row[feat] for feat in feature_names]
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    return X_scaled


# ─── Endpoints ───────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Insurance Fraud Detection API is running!",
        "docs": "/docs"
    }


@app.post("/predict")
def predict(data: ClaimInput):
    try:
        X_scaled = preprocess_input(data)
        prediction = int(model.predict(X_scaled)[0])
        proba = model.predict_proba(X_scaled)[0]
        fraud_prob = round(float(proba[1]) * 100, 2)
        not_fraud_prob = round(float(proba[0]) * 100, 2)

        return {
            "prediction": prediction,
            "label": "FRAUD" if prediction == 1 else "NOT FRAUD",
            "fraud_probability": f"{fraud_prob}%",
            "not_fraud_probability": f"{not_fraud_prob}%",
            "model_used": type(model).__name__
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/valid-values")
def get_valid_values():
    """Returns valid values for all categorical fields"""
    return {
        col: list(le.classes_)
        for col, le in label_encoders.items()
        if col != 'incident_city'  # too many cities
    }
