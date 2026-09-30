"""
FastAPI backend — Squat Analyzer v2 (angles only)
Features: knee_angle, hip_angle, back_angle
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, os, numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "squat_model_v2.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

FEEDBACK = {
    0: "Correct form ✓",
    1: "Squat deeper — knee slightly too high",
    2: "You're going too deep — stop at 90°",
    3: "INCORRECT — fix your form now!",
}
ARABIC_FEEDBACK = {
    0: "شكل صحيح ✓",
    1: "انزل أكثر — الركبة ما وصلت 90°",
    2: "عمق زيادة — قف عند 90°",
    3: "خطأ شديد — صحح وضعيتك الآن!",
}

class SquatInput(BaseModel):
    knee_angle: float
    hip_angle: float
    back_angle: float

@app.get("/")
def root():
    return {"status": "ok", "model": "squat_v2_angles_only",
            "features": ["knee_angle", "hip_angle", "back_angle"]}

@app.post("/predict")
def predict(data: SquatInput):
    X = np.array([[data.knee_angle, data.hip_angle, data.back_angle]])
    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return {
        "label":    label,
        "feedback": FEEDBACK[label],
        "feedback_ar": ARABIC_FEEDBACK[label],
        "confidence": round(max(proba), 3),
        "probabilities": {str(i): round(p, 3) for i, p in enumerate(proba)},
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
