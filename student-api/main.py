from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import joblib, pandas as pd
from pathlib import Path

app = FastAPI(title="Student Performance API", version="1.0.0")

# Allow your WordPress site to call this API
ALLOWED_ORIGINS = [
    "https://movidya.online",
    "http://movidya.online",
    "http://localhost:3000",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model files shipped with this repo
PIPE = joblib.load(Path("student_model.joblib"))
RANGES_PATH = Path("feature_ranges.joblib")
RANGES = joblib.load(RANGES_PATH) if RANGES_PATH.exists() else None

class PredictIn(BaseModel):
    Hours_Studied: float
    Attendance_Percentage: float
    Previous_Scores: float
    Anxiety_Level: int
    Extracurricular_Activities: str  # "Yes" or "No"

    @field_validator("Extracurricular_Activities")
    def chk_extra(cls, v):
        v = v.strip()
        if v not in {"Yes", "No"}:
            raise ValueError("Extracurricular_Activities must be 'Yes' or 'No'")
        return v

    @field_validator("Anxiety_Level")
    def clamp_anx(cls, v):
        return 1 if v < 1 else 5 if v > 5 else v

def clamp_with_ranges(row: dict):
    if not RANGES: return row
    for col, (lo, hi) in RANGES.items():
        if col in row:
            if row[col] < lo: row[col] = lo
            if row[col] > hi: row[col] = hi
    return row

@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /predict"}

@app.post("/predict")
def predict(payload: PredictIn):
    data = clamp_with_ranges(payload.model_dump())
    df = pd.DataFrame([data])
    pred = float(PIPE.predict(df)[0])
    return {"input_used": data, "predicted_score": round(pred, 2), "pass": pred >= 50}
