# Pace – Squat Form Analysis

Real-time squat form analysis with **MediaPipe Pose** and a machine-learning model, plus session reports and measurable goals (closed-loop coaching).
Part of the **AI-Powered Smart Fitness Application** graduation project.

---

## How it works

```
Camera → MediaPipe (pose landmarks) → joint angles + movement phase → model → stable feedback + rep counting
                                                                                  ↓
                                              session log → session report → goal check → next goal
```

1. **MediaPipe** detects the body from a **side view** camera.
2. Three angles are calculated with the **dot product**:
   - **Knee:** hip – knee – ankle
   - **Hip:** shoulder – hip – knee
   - **Back:** torso lean from vertical
3. The **movement phase** is detected: `0 standing · 1 descent · 2 bottom · 3 ascent`.
4. The model predicts a label for **each joint**:

| Label | Meaning | Color |
|:----:|---------|-------|
| 0 | Correct | 🟢 Green |
| 1 | Above the correct range | 🟡 Yellow |
| 2 | Below the correct range | 🟠 Orange |
| 3 | Far outside the range | 🔴 Red |

5. A rep is counted **only** if the knee reached depth, the hips really went down, and there was no lasting form error.
6. After the session, a **report** is created with scores, warnings, and a **measurable goal** for the next session.

---

## Folder structure

```
backend/
├── main_v3.py                 FastAPI server – loads the model chosen in current_model.txt
├── squat_camera_v4.py         Camera app (latest)
├── pose_landmarker_full.task  MediaPipe pose model
│
├── train_experiment.py        Train / register models – each one saved separately
├── evaluate_model.py          Detailed evaluation of a model
├── squat_dataset_phases.csv   Training data (40,581 frames · 424 reps · 40 people · synthetic)
├── current_model.txt          Which experiment the app uses
├── experiments/
│   ├── experiments.xlsx       Experiment tracking sheet (Training · Live tests · How to use)
│   └── exp_NNN_<name>/        model.pkl + info.json (settings, data, scores)
│
├── pace_report.py             Session metrics → report → goal check → next goal
├── plot_session.py            Plots a session (angles over time, reps, errors)
│
├── sessions/                  (local only) per-frame + per-rep logs of each session
├── profiles/                  (local only) user profile, goals, session reports
│
└── old: main.py, squat_camera_v3.py, train_squat3.py, squat_model.pkl, squat_dataset_angles_only.csv
```

`sessions/` and `profiles/` contain personal training data and are **not** uploaded to GitHub.

---

## Environment

Developed with **Anaconda**:

- **Environment:** `squat_env`
- **Python:** 3.11

```bash
conda create -n squat_env python=3.11
conda activate squat_env
pip install mediapipe opencv-python numpy pandas scikit-learn fastapi uvicorn requests matplotlib openpyxl
```

> Always check you see `(squat_env)` at the start of the terminal line.

---

## Run the app

Two terminals, both inside `backend/` with `squat_env` active.

**Terminal 1 – server**

```bash
uvicorn main_v3:app --port 8000
```

You should see `Using model: .../experiments/exp_.../model.pkl`.

**Terminal 2 – camera**

```bash
python squat_camera_v4.py
```

- Stand **sideways**, full body visible, straight for 2 seconds → counting starts.
- Leaving the frame pauses counting.
- Press **Q** to finish → summary + session report are created.

Then:

```bash
python plot_session.py
```

Settings (timings, thresholds, `USER_ID`) are at the top of `squat_camera_v4.py`.

---

## Experiments (models are never overwritten)

```bash
# keep an existing model as an experiment (copied, not retrained)
python train_experiment.py --register squat_model.pkl --name original_v3 --notes "first phases model"

# train a new experiment
python train_experiment.py --name small_rf --trees 50 --depth 12 --notes "smaller + faster"
python train_experiment.py --name tree8 --model tree --depth 8 --notes "single tree for the phone"

# choose the model the app uses (then restart the server)
python train_experiment.py --use exp_002_small_rf

# list all experiments
python train_experiment.py --list
```

Every experiment is scored the **same way**, so results can be compared:

- split **by person** (test people are never used for training)
- macro F1 + balanced accuracy per joint
- rep accuracy (whole rep judged correct / incorrect)
- speed (ms per frame) and size (MB)

A row is added to `experiments/experiments.xlsx` automatically (close Excel first).
After every camera test, fill one row in the **Live tests** tab.

> Change **one thing per experiment** and write it in `--notes`.

---

## Session report & goals

`pace_report.py` runs automatically at the end of each session.

| Metric | Definition |
|--------|-----------|
| Range of Motion | (180 − lowest knee angle) / 90 → 90° = 100% |
| Knee / Hip / Back | % of rep frames where the joint was correct |
| Movement Control | lowering time / 1 s → ≥ 1 s = 100% |
| Warnings | number of reps with a lasting error, per joint |
| Form Score | average of the scores above |

- **Main focus** = joint with the most warnings.
- **Goal** = warnings N → 60% of N (e.g. 5 → 3 → 1).
- At ≤ 1 warning the weakness is resolved and the focus moves to the next one.
- Goals are only compared when the **same model** was used.

---

## Current status & limitations

- ✅ Real-time feedback, stable messages, false-rep protection, session reports, goal loop, experiment tracking.
- ⚠️ The dataset is **synthetic** — high training scores do not prove it works on real people. Live tests and real recordings are needed.
- ⚠️ Side view measures depth, hip and back. **Knee alignment (knee caving in) is not measured** — it needs a front view.
- 🔜 Planned: model on the phone (Flutter), session reports + Pace Coach (RAG) on the server.