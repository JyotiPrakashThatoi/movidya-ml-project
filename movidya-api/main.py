# movidya-api/main.py
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# Paths & Settings
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
# Detect path for Render environment
if not os.path.exists(ART_DIR):
    ART_DIR = "/app/movidya-api/artifacts"

STUDENT_MODEL_PATH = os.path.join(ART_DIR, "student_model.joblib")
EFFICIENCY_MODEL_PATH = os.path.join(ART_DIR, "sleep_efficiency_model.pkl")

# =========================
# App & CORS
# =========================
app = FastAPI(
    title="Movidya API",
    version="1.0.1",
    description="Unified API for Movidya projects"
)

# For local testing you may add "http://127.0.0.1:10000" temporarily
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://movidya.online",
        "https://www.movidya.online",
        # "http://127.0.0.1:10000",  # uncomment only for local browser tests
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Model Loading Helpers
# =========================
def load_joblib(path: str):
    if not os.path.exists(path):
        print(f"[WARN] Model file not found: {path}")
        return None
    try:
        model = joblib.load(path)
        print(f"[OK] Loaded model: {path}")
        return model
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None

student_model = load_joblib(STUDENT_MODEL_PATH)
efficiency_model = load_joblib(EFFICIENCY_MODEL_PATH)

# =========================
# Root & Health
# =========================
@app.get("/")
def root():
    return {
        "service": "movidya-api",
        "docs": "/docs",
        "models": {
            "student_loaded": student_model is not None,
            "efficiency_loaded": efficiency_model is not None
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "message": "Movidya API running smoothly"}

# =========================
# 1) Student Performance (Project 1)
# =========================
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

class StudentInput(BaseModel):
    Hours_Studied: float = Field(..., ge=0, le=16)
    Attendance_Percentage: float = Field(..., ge=0, le=100)
    Previous_Scores: float = Field(..., ge=0, le=100)
    Anxiety_Level: int = Field(..., ge=1, le=5)
    Extracurricular_Activities: str = Field(..., examples=["Yes", "No"])

def to_student_df(p: StudentInput) -> pd.DataFrame:
    cols = [
        "Hours_Studied",
        "Attendance_Percentage",
        "Previous_Scores",
        "Anxiety_Level",
        "Extracurricular_Activities",
    ]
    row = {
        "Hours_Studied": p.Hours_Studied,
        "Attendance_Percentage": p.Attendance_Percentage,
        "Previous_Scores": p.Previous_Scores,
        "Anxiety_Level": p.Anxiety_Level,
        "Extracurricular_Activities": p.Extracurricular_Activities,
    }
    return pd.DataFrame([row], columns=cols)

@app.post("/student/predict")
def student_predict(payload: StudentInput):
    if student_model is None:
        return {"error": "student model not loaded"}
    try:
        X = to_student_df(payload)
        pred = float(student_model.predict(X)[0])
        score = round(pred, 2)

        # Assign performance level
        if score < 40:
            grade = "Fail"
        elif score < 60:
            grade = "Pass"
        elif score < 80:
            grade = "Good"
        else:
            grade = "Excellent"

        # Motivational message
        messages = {
            "Fail": "Don't give up — review your study habits and try again! 💪",
            "Pass": "You passed! Keep improving — small efforts daily count. 🌱",
            "Good": "Nice work! You’re doing well — stay consistent! 💡",
            "Excellent": "Outstanding! You’re at the top of your game! 🚀"
        }

        return {
            "final_score": score,
            "grade": grade,
            "message": messages[grade]
        }
    except Exception as e:
        return {"error": f"student prediction failed: {e}"}


# =========================
# 2) Study Efficiency (Project 2)
#    Predicts Quality of Sleep (1–10) → Efficiency Score (0–100)
# =========================
class EfficiencyInput(BaseModel):
    gender: str = Field(..., examples=["Male", "Female"])
    age: int = Field(..., ge=10, le=100)
    occupation: str = Field(..., examples=["Student", "Doctor", "Engineer", "Teacher"])
    sleep_duration: float = Field(..., ge=0, le=24, description="Hours slept (e.g., 7.5)")
    physical_activity_level: int = Field(..., ge=0, le=100, description="0–100 index")
    stress_level: int = Field(..., ge=0, le=10, description="0–10")
    bmi_category: str = Field(..., examples=["Underweight", "Normal", "Overweight", "Obese", "Normal Weight"])
    heart_rate: int = Field(..., ge=30, le=220)
    daily_steps: int = Field(..., ge=0, le=100000)
    sleep_disorder: str = Field("None", examples=["None", "Insomnia", "Sleep Apnea"])

TRAIN_COLS_EFF = [
    "Gender","Age","Occupation","Sleep Duration","Physical Activity Level",
    "Stress Level","BMI Category","Heart Rate","Daily Steps","Sleep Disorder",
]

def to_efficiency_df(p: EfficiencyInput) -> pd.DataFrame:
    row = {
        "Gender": p.gender,
        "Age": p.age,
        "Occupation": p.occupation,
        "Sleep Duration": p.sleep_duration,
        "Physical Activity Level": p.physical_activity_level,
        "Stress Level": p.stress_level,
        "BMI Category": p.bmi_category,
        "Heart Rate": p.heart_rate,
        "Daily Steps": p.daily_steps,
        "Sleep Disorder": p.sleep_disorder or "None",
    }
    return pd.DataFrame([row], columns=TRAIN_COLS_EFF)

def efficiency_fallback(p: EfficiencyInput) -> float:
    q = 0.9*min(max(p.sleep_duration,0),9) \
        + 0.03*p.physical_activity_level \
        - 0.5*p.stress_level \
        - 0.02*p.heart_rate \
        + 0.0002*p.daily_steps
    return float(np.clip(q, 1.0, 10.0))

def efficiency_tips(p: EfficiencyInput):
    tips = []
    if p.sleep_duration < 6.5:
        tips.append("Increase sleep toward 7–8 hours; it’s the biggest efficiency driver.")
    if p.stress_level >= 6:
        tips.append("High stress detected—try 25–30 min focus blocks with 5-min breaks.")
    if p.daily_steps < 6000:
        tips.append("Aim for at least 6,000+ steps; light movement improves focus.")
    if p.physical_activity_level < 50:
        tips.append("Add a short workout or walk; activity improves sleep quality.")
    if p.heart_rate > 80:
        tips.append("Elevated resting heart rate—hydrate and wind down before bed.")
    if not tips:
        tips.append("Great routine! Maintain consistency and track weekly trends.")
    return tips

@app.post("/efficiency/predict")
def efficiency_predict(payload: EfficiencyInput):
    if efficiency_model is None:
        quality = efficiency_fallback(payload)
    else:
        try:
            X = to_efficiency_df(payload)
            quality = float(efficiency_model.predict(X)[0])  # 1–10
        except Exception:
            quality = efficiency_fallback(payload)

    efficiency = float(np.clip(quality * 10.0, 0, 100))      # 0–100
    level = "High" if efficiency >= 80 else "Moderate" if efficiency >= 60 else "Low"
    return {
        "predicted_sleep_quality": round(quality, 2),
        "efficiency_score": round(efficiency, 1),
        "level": level,
        "tips": efficiency_tips(payload),
    }

