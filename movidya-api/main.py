# movidya-api/main.py
from __future__ import annotations

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import json

# optional tree export for explanation
try:
    from sklearn.tree import export_text
    SKLEARN_TREE_AVAILABLE = True
except Exception:
    SKLEARN_TREE_AVAILABLE = False

# =========================
# Paths & Settings (define early)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
# Detect path for Render environment fallback
if not os.path.exists(ART_DIR):
    ART_DIR = "/app/movidya-api/artifacts"

STUDENT_MODEL_PATH = os.path.join(ART_DIR, "student_model.joblib")
EFFICIENCY_MODEL_PATH = os.path.join(ART_DIR, "sleep_efficiency_model.pkl")

# New Project 3 artifact paths
LEARNING_PROD_MODEL_PATH = os.path.join(ART_DIR, "production_model.joblib")
LEARNING_EXPLAIN_MODEL_PATH = os.path.join(ART_DIR, "explainable_model.joblib")
LEARNING_ENCODER_PATH = os.path.join(ART_DIR, "learning_style_label_encoder.joblib")

# =========================
# App & CORS (create app early)
# =========================
app = FastAPI(
    title="Movidya API",
    version="1.0.2",
    description="Unified API for Movidya projects (Student perf, Efficiency, Learning Style)",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
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
# Feedback logging (now that BASE_DIR & app exist)
# =========================
FEEDBACK_LOG = Path(BASE_DIR) / "logs" / "learning_feedback.jsonl"
FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)

class Feedback(BaseModel):
    input: dict
    predicted: str
    correct: bool
    corrected_label: Optional[str] = None
    comment: Optional[str] = None

@app.post("/learning-style/feedback")
def learning_style_feedback(f: Feedback):
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "input": f.input,
        "predicted": f.predicted,
        "correct": f.correct,
        "corrected_label": f.corrected_label,
        "comment": f.comment
    }
    try:
        with open(FEEDBACK_LOG, "a", encoding="utf8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")

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
learning_production_model = load_joblib(LEARNING_PROD_MODEL_PATH)
learning_explain_model = load_joblib(LEARNING_EXPLAIN_MODEL_PATH)
learning_label_encoder = load_joblib(LEARNING_ENCODER_PATH)

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
            "efficiency_loaded": efficiency_model is not None,
            "learning_production_loaded": learning_production_model is not None,
            "learning_explainable_loaded": learning_explain_model is not None,
            "learning_encoder_loaded": learning_label_encoder is not None
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Movidya API running smoothly",
        "models": {
            "student": student_model is not None,
            "efficiency": efficiency_model is not None,
            "learning_production": learning_production_model is not None
        }
    }

# =========================
# 1) Student Performance (Project 1)
# =========================
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
    # Predict Quality of Sleep (1–10) internally
    if efficiency_model is None:
        quality = efficiency_fallback(payload)
    else:
        try:
            X = to_efficiency_df(payload)
            quality = float(efficiency_model.predict(X)[0])
        except Exception:
            quality = efficiency_fallback(payload)

    # Convert to 0–100 efficiency score
    efficiency = float(np.clip(quality * 10.0, 0, 100))

    # 4-tier mapping focused on efficiency
    if efficiency < 60:
        grade = "Low"
        message = "Needs attention—sleep more consistently and lower stress for better focus."
    elif efficiency < 75:
        grade = "Moderate"
        message = "Decent routine—small tweaks in rest and movement will boost your study focus."
    elif efficiency < 90:
        grade = "High"
        message = "Great job—your current routine supports strong study sessions."
    else:
        grade = "Peak"
        message = "Outstanding—your habits are optimized for deep, focused learning."

    # Efficiency-first response (no explicit sleep metric)
    return {
        "efficiency_score": round(efficiency, 1),   # 0–100
        "grade": grade,                              # Low/Moderate/High/Peak
        "message": message,
        "tips": efficiency_tips(payload),            # actionables already phrased as study improvements
        # Optional: expose drivers if you want, with efficiency language
        "drivers": {
            "hours_of_recovery": payload.sleep_duration,   # kept generic
            "stress_level": payload.stress_level,
            "activity_index": payload.physical_activity_level,
            "daily_steps": payload.daily_steps,
            "resting_heart_rate": payload.heart_rate,
        }
    }

# =========================
# 3) Learning Style Predictor (Project 3)
# =========================
class LearningStyleInput(BaseModel):
    age: Optional[int] = Field(17, ge=10, le=100)
    sex: Optional[str] = Field("F", examples=["M", "F"])
    studytime: Optional[int] = Field(2, ge=1, le=4, description="1= <2h, 2=2-5h, 3=5-10h, 4=10+h")
    failures: Optional[int] = Field(0, ge=0, le=10)
    goout: Optional[int] = Field(2, ge=1, le=5)
    activities: Optional[str] = Field("no", examples=["yes", "no"])
    health: Optional[int] = Field(3, ge=1, le=5)
    freetime: Optional[int] = Field(3, ge=1, le=5)
    absences: Optional[int] = Field(0, ge=0, le=200)

def to_learning_df(p: LearningStyleInput) -> pd.DataFrame:
    cols = ["age","sex","studytime","failures","goout","activities","health","freetime","absences"]
    row = {
        "age": p.age,
        "sex": p.sex,
        "studytime": p.studytime,
        "failures": p.failures,
        "goout": p.goout,
        "activities": p.activities,
        "health": p.health,
        "freetime": p.freetime,
        "absences": p.absences
    }
    return pd.DataFrame([row], columns=cols)

RECOMMENDATION_MAP = {
    "Visual": "You learn best with diagrams, charts and visual summaries. Try mind-maps and videos.",
    "Auditory": "You learn best by listening and discussing. Try audio summaries, group discussions, and recorded lectures.",
    "Kinesthetic": "You learn by doing. Try hands-on projects, practice tests, and real exercises."
}

@app.post("/learning-style/predict")
def learning_style_predict(payload: LearningStyleInput):
    if learning_production_model is None:
        raise HTTPException(status_code=503, detail="Learning style production model not loaded.")

    # Convert to DataFrame
    X = to_learning_df(payload)

    try:
        # Predict (model pipeline should handle preprocessing)
        pred_enc = learning_production_model.predict(X)
        # If label encoder exists, decode to string labels
        if learning_label_encoder is not None:
            try:
                pred_label = learning_label_encoder.inverse_transform(pred_enc)[0]
            except Exception:
                # Sometimes model returns strings already
                pred_label = pred_enc[0] if isinstance(pred_enc, (list, np.ndarray)) else str(pred_enc)
        else:
            pred_label = pred_enc[0] if isinstance(pred_enc, (list, np.ndarray)) else str(pred_enc)

        # get probabilities if available
        probs = None
        classes = None
        try:
            proba = learning_production_model.predict_proba(X)[0]
            probs = proba.tolist()
            # attempt to get classes — prefer encoder.classes_ if available
            if learning_label_encoder is not None:
                classes = learning_label_encoder.classes_.tolist()
            else:
                # fallback to classifier classes if pipeline provides it
                if hasattr(learning_production_model, "named_steps") and "classifier" in learning_production_model.named_steps:
                    clf = learning_production_model.named_steps["classifier"]
                    if hasattr(clf, "classes_"):
                        classes = clf.classes_.tolist()
        except Exception:
            probs = None
            classes = None

        return {
            "learning_style": pred_label,
            "recommendation": RECOMMENDATION_MAP.get(pred_label, ""),
            "confidence_scores": {"classes": classes, "probs": probs}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning style prediction failed: {e}")

@app.get("/learning-style/explain")
def learning_style_explain():
    """
    Return a simple textual rule explanation from the DecisionTree explainable model, if available.
    """
    if learning_explain_model is None:
        raise HTTPException(status_code=404, detail="Explainable learning model not available.")

    if not SKLEARN_TREE_AVAILABLE:
        raise HTTPException(status_code=500, detail="sklearn.tree.export_text not available in this environment.")

    try:
        # Expect explainable model to be a pipeline with a 'classifier' that is a DecisionTreeClassifier
        if hasattr(learning_explain_model, "named_steps") and "classifier" in learning_explain_model.named_steps:
            clf = learning_explain_model.named_steps["classifier"]
            # Attempt to get feature names from preprocessor if possible
            feature_text = "Decision tree rules unavailable (failed to fetch feature names)."
            try:
                # If the pipeline has a preprocessor, try to infer transformed feature names (best-effort)
                preproc = learning_explain_model.named_steps.get("preprocessor", None)
                if preproc is not None and hasattr(preproc, "transformers_"):
                    # Best-effort: collect numeric feature names + onehot categories
                    feature_names = []
                    for name, transformer, cols in preproc.transformers_:
                        if name == "num":
                            feature_names.extend(cols if isinstance(cols, (list, tuple)) else [cols])
                        elif name == "cat":
                            # OneHotEncoder categories are inside transformer[1].named_steps... it's complex;
                            # fallback to original column names for readability.
                            feature_names.extend(cols if isinstance(cols, (list, tuple)) else [cols])
                    feature_text = export_text(clf, feature_names=feature_names)
                else:
                    feature_text = export_text(clf)
            except Exception:
                try:
                    feature_text = export_text(clf)
                except Exception as e:
                    feature_text = f"Could not export tree: {e}"

            return {"explanation_text": feature_text}
        else:
            raise HTTPException(status_code=500, detail="Explainable model pipeline missing classifier step.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {e}")

# =========================
# End of file
# =========================
