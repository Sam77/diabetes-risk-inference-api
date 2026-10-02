"""FastAPI inference server for the diabetes risk model."""
import json
from contextlib import asynccontextmanager
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "model/model.joblib"
METADATA_PATH = "model/metadata.json"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    state["model"] = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        state["metadata"] = json.load(f)
    yield
    state.clear()


app = FastAPI(title="Diabetes Risk Inference API", version="1.0.0", lifespan=lifespan)


class Patient(BaseModel):
    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(ge=0, le=300, description="Plasma glucose (mg/dL)")
    blood_pressure: float = Field(ge=0, le=200, description="Diastolic (mm Hg)")
    skin_thickness: float = Field(ge=0, le=100, description="Triceps skinfold (mm)")
    insulin: float = Field(ge=0, le=900, description="2-hour serum insulin (mu U/ml)")
    bmi: float = Field(ge=0, le=80)
    diabetes_pedigree: float = Field(ge=0, le=3)
    age: int = Field(ge=18, le=120)


class Prediction(BaseModel):
    risk_probability: float
    high_risk: bool


def score(patients: List[Patient]) -> List[Prediction]:
    features = state["metadata"]["features"]
    df = pd.DataFrame([p.model_dump() for p in patients])[features]
    probs = state["model"].predict_proba(df)[:, 1]
    return [Prediction(risk_probability=round(float(p), 4), high_risk=bool(p >= 0.5))
            for p in probs]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    return state["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    return score([patient])[0]


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(patients: List[Patient]):
    if not 1 <= len(patients) <= 1000:
        raise HTTPException(status_code=422, detail="Send between 1 and 1000 records")
    return score(patients)
