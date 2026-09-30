"""
main_v3.py
==========
FastAPI backend — Squat Analyzer v3
Model  : MultiOutputClassifier (knee_label, hip_label, back_label)
Input  : knee_angle, hip_angle, back_angle
Output : per-joint labels + specific feedback
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle, os
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "squat_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

FEEDBACK = {
    'knee': {
        0: None,
        1: "Squat deeper",
        2: "Too deep — stop at 90°",
        3: "Knee error — stop now!",
    },
    'hip': {
        0: None,
        1: "Open your hips more",
        2: "Lean forward more",
        3: "Hip error — stop now!",
    },
    'back': {
        0: None,
        1: "Lean back slightly",
        2: "Lean forward slightly",
        3: "Back error — stop now!",
    },
}

OVERALL = {
    0: "Correct form",
    1: "Minor adjustment needed",
    2: "Minor adjustment needed",
    3: "Incorrect — fix your form!",
}

class SquatInput(BaseModel):
    knee_angle: float
    hip_angle:  float
    back_angle: float

@app.get("/")
def root():
    return {"status": "ok", "model": "squat_v3_multioutput",
            "features": ["knee_angle", "hip_angle", "back_angle"]}

@app.post("/predict")
def predict(data: SquatInput):
    X = pd.DataFrame([{
        "knee_angle": data.knee_angle,
        "hip_angle":  data.hip_angle,
        "back_angle": data.back_angle,
    }])

    pred = model.predict(X)[0]
    knee_l, hip_l, back_l = int(pred[0]), int(pred[1]), int(pred[2])

    # Build feedback messages
    msgs = [
        FEEDBACK['knee'][knee_l],
        FEEDBACK['hip'][hip_l],
        FEEDBACK['back'][back_l],
    ]
    feedback_parts = [m for m in msgs if m]
    feedback = " | ".join(feedback_parts) if feedback_parts else "Correct form ✓"

    # Overall severity: worst joint
    overall = max(knee_l, hip_l, back_l)

    return {
        "label":       overall,
        "feedback":    feedback,
        "joint_labels": {
            "knee": knee_l,
            "hip":  hip_l,
            "back": back_l,
        },
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
