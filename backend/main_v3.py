"""
main_v3.py
==========
FastAPI backend — Squat Analyzer v3
Model  : MultiOutputClassifier (knee_label, hip_label, back_label)
Input  : knee_angle, hip_angle, back_angle, phase_id
         phase_id: 0 standing | 1 descent | 2 bottom | 3 ascent
Output : per-joint labels + specific feedback
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
try:   # model chosen with: python train_experiment.py --use exp_...
    with open(os.path.join(HERE, "current_model.txt")) as f:
        MODEL_PATH = os.path.join(HERE, f.read().strip(), "model.pkl")
except FileNotFoundError:
    MODEL_PATH = os.path.join(HERE, "squat_model.pkl")
print("Using model:", MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Label meaning (same as squat_dataset_phases.csv):
#   0 = correct | 1 = above correct range | 2 = below correct range | 3 = far outside range
FEEDBACK = {
    'knee': {
        0: None,
        1: "Squat deeper",                 # knee angle too big → not deep enough
        2: "Too deep — stop at 90°",       # knee angle too small → over-flexion
        3: "Knee error — stop now!",
    },
    'hip': {
        0: None,
        1: "Push your hips back",          # hip angle too open → not hinging enough
        2: "Don't fold forward too much",  # hip angle too closed
        3: "Hip error — stop now!",
    },
    'back': {
        0: None,
        1: "Raise your chest — back too low",  # too much forward lean
        2: "Lean forward slightly",            # torso too upright at the bottom
        3: "Back error — raise your chest now!",
    },
}

OVERALL = {
    0: "Correct form",
    1: "Minor adjustment needed",
    2: "Minor adjustment needed",
    3: "Incorrect — fix your form!",
}

FEATURES = ["knee_angle", "hip_angle", "back_angle", "phase_id"]

class SquatInput(BaseModel):
    knee_angle: float
    hip_angle:  float
    back_angle: float
    phase_id:   int     # 0 standing | 1 descent | 2 bottom | 3 ascent

@app.get("/")
def root():
    return {"status": "ok", "model": "squat_v3_multioutput_phases",
            "features": FEATURES}

@app.post("/predict")
def predict(data: SquatInput):
    X = pd.DataFrame([{
        "knee_angle": data.knee_angle,
        "hip_angle":  data.hip_angle,
        "back_angle": data.back_angle,
        "phase_id":   data.phase_id,
    }])[FEATURES]

    pred = model.predict(X)[0]
    knee_l, hip_l, back_l = int(pred[0]), int(pred[1]), int(pred[2])

    # Build feedback messages — back first (safety), then knee, then hip
    msgs = [
        FEEDBACK['back'][back_l],
        FEEDBACK['knee'][knee_l],
        FEEDBACK['hip'][hip_l],
    ]
    feedback_parts = [m for m in msgs if m]
    feedback = " | ".join(feedback_parts) if feedback_parts else "Correct form ✓"

    # Overall severity: worst joint
    overall = max(knee_l, hip_l, back_l)

    return {
        "label":       overall,
        "feedback":    feedback,
        "phase_id":    data.phase_id,
        "joint_labels": {
            "knee": knee_l,
            "hip":  hip_l,
            "back": back_l,
        },
    }

@app.get("/health")
def health():
    return {"status": "healthy"}