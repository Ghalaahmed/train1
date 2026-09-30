
# train1 – Squat Form Analysis (v3)

Real-time squat form analysis using **MediaPipe Pose** and a **Random Forest** model.
The camera detects the body, calculates joint angles, sends them to a FastAPI backend, and shows instant feedback with a color bar and a rep counter.

Part of the **AI-Powered Smart Fitness Application** graduation project.

---

## How it works

```
Camera → MediaPipe (33 landmarks) → joint angles → FastAPI /predict → ML model → feedback + color
```

1. **MediaPipe** detects the body landmarks from the camera (side view).
2. Three angles are calculated using the **dot product**:
   - **Knee angle:** hip – knee – ankle
   - **Hip angle:** shoulder – hip – knee
   - **Back angle:** torso lean from vertical
3. The angles are sent to the backend, and the model predicts a label for **each joint** (knee, hip, back).
4. The screen shows the feedback, and a rep is counted **only** if the user reached real depth **and** had no form errors during the whole rep.

### Labels

| Label | Meaning | Color |
|:----:|---------|-------|
| 0 | Correct form | 🟢 Green |
| 1 | Minor adjustment (type 1) | 🟡 Yellow |
| 2 | Minor adjustment (type 2) | 🟠 Orange |
| 3 | Incorrect – stop and fix | 🔴 Red |

---

## Files

### `backend/` (v3 – latest)

| File | Description |
|------|-------------|
| `squat_camera_v3.py` | Opens the camera, runs MediaPipe, calculates the angles, calls the API, and draws the feedback bar, rep counter, and movement state on screen. |
| `main_v3.py` | FastAPI backend. Loads `squat_model.pkl` and exposes `POST /predict`, which takes the 3 angles and returns a label + feedback message for each joint. |
| `train_squat3.py` | Trains the model on `squat_dataset_angles_only.csv` and saves it as `squat_model.pkl`. Prints the accuracy for each joint. |
| `squat_model.pkl` | The trained model: `MultiOutputClassifier` with `RandomForestClassifier` (300 trees, max depth 15). |
| `squat_dataset_angles_only.csv` | Training data: 6,000 rows. Inputs: `knee_angle`, `hip_angle`, `back_angle`. Outputs: `knee_label`, `hip_label`, `back_label`, `overall_label`. |
| `pose_landmarker_full.task` | MediaPipe pose detection model (required by the camera script). |

### Version 1 (old – for reference only)

| File | Description |
|------|-------------|
| `backend/main.py` | v1 backend. Used 5 features (including cross-product values) and one overall label. |
| `squat_train.py` | v1 training script. |
| `squat_dataset.csv` | v1 dataset. |

> ⚠️ v1 files are **not compatible** with the current `squat_model.pkl`.

---

## Rep counting logic

The camera script tracks the movement with a state machine:

```
STANDING → DESCENDING → ASCENDING → STANDING
```

| Setting | Value | Meaning |
|---------|:-----:|---------|
| `DESCENT_TRIGGER` | 140° | Knee below this → the squat started |
| `VALID_DEPTH_MAX` | 115° | Must reach this depth for the rep to count |
| `ERROR_ZONE_MAX` | 120° | Form errors are only checked near the bottom |
| `STAND_THRESHOLD` | 155° | Knee above this → back to standing |

If there are **3 errors in a row**, a warning appears at the bottom of the screen.

---

## Environment

Developed using **Anaconda** with a dedicated conda environment:

- **Environment name:** `squat_env`
- **Python version:** 3.11
- **Libraries:** MediaPipe, OpenCV, NumPy, Pandas, scikit-learn, FastAPI, Uvicorn, Requests

### Setup

1. Install [Anaconda](https://www.anaconda.com/download)
2. Create and activate the environment:

```bash
conda create -n squat_env python=3.11
conda activate squat_env
pip install mediapipe opencv-python numpy pandas scikit-learn fastapi uvicorn requests
```

> Make sure `squat_env` is activated before running any script.
> You should see `(squat_env)` at the start of your terminal line.

---

## Run

You need **two terminals**, both inside `backend/` with `squat_env` activated.

**Terminal 1 – start the backend:**

```bash
cd backend
uvicorn main_v3:app --port 8000
```

**Terminal 2 – start the camera:**

```bash
cd backend
python squat_camera_v3.py
```

- Stand **sideways** to the camera, with your full body visible.
- Press **Q** to quit.
- If the bar shows `API unavailable`, the backend in Terminal 1 is not running.

---

## Retrain the model

```bash
cd backend
python train_squat3.py
```

This overwrites `squat_model.pkl` with the new model.
