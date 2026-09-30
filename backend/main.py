import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open('squat_model.pkl', 'rb') as f:
    model = pickle.load(f)

class SquatData(BaseModel):
    knee_angle: float
    knee_cross: float
    hip_angle: float
    hip_cross: float
    back_angle: float

FEEDBACK = {
    0: 'Correct form',
    1: 'Adjust: knee caving in / too shallow / hip too open',
    2: 'Adjust: knee bowing out / too deep / too upright',
    3: 'Incorrect: severe error — show demo video',
}

@app.post('/predict')
def predict(data: SquatData):
    x = [[data.knee_angle, data.knee_cross,
          data.hip_angle, data.hip_cross, data.back_angle]]
    label = int(model.predict(x)[0])
    return {'label': label, 'feedback': FEEDBACK[label]}