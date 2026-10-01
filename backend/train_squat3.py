"""
train_squat.py
==============
Trains a multi-output squat model.
Input  : knee_angle, hip_angle, back_angle
Outputs: knee_label, hip_label, back_label  (one prediction per joint)
Saved  : squat_model.pkl
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ── 1. Load dataset ───────────────────────────────────────────────
df = pd.read_csv("squat_dataset_phases.csv")

FEATURES = ['knee_angle', 'hip_angle', 'back_angle', 'phase_id']
TARGETS  = ['knee_label', 'hip_label', 'back_label']

X = df[FEATURES]
y = df[TARGETS]

print(f"Dataset : {len(df)} rows")
print(f"Knee labels : {y['knee_label'].value_counts().sort_index().to_dict()}")
print(f"Hip  labels : {y['hip_label'].value_counts().sort_index().to_dict()}")
print(f"Back labels : {y['back_label'].value_counts().sort_index().to_dict()}")

# ── 2. Train / test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ── 3. Train ──────────────────────────────────────────────────────
base = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
model = MultiOutputClassifier(base)

print("\nTraining...")
model.fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test)
LABEL_NAMES = ['Correct', 'Adj1', 'Adj2', 'Incorrect']

for i, joint in enumerate(TARGETS):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"\n{joint}  —  Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i],
          target_names=LABEL_NAMES, zero_division=0))

# ── 5. Save ───────────────────────────────────────────────────────
with open("squat_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved: squat_model.pkl")

# ── 6. Quick test ─────────────────────────────────────────────────
FEEDBACK = {
    'knee': {0: None, 1: "Squat deeper",      2: "Too deep — stop at 90°", 3: "Knee error!"},
    'hip' : {0: None, 1: "Open your hips",    2: "Lean forward more",      3: "Hip error!"},
    'back': {0: None, 1: "Lean back slightly", 2: "Lean forward slightly",  3: "Back error!"},
}

print("\n--- Quick test ---")
tests = [
    ([92, 97, 35],  "Good squat"),
    ([108, 112, 35], "Shallow squat"),
    ([80, 97, 35],  "Too deep"),
    ([92, 97, 60],  "Too much lean"),
]
for vals, desc in tests:
    pred = model.predict([vals])[0]
    kl, hl, bl = pred
    msgs = [FEEDBACK['knee'][kl], FEEDBACK['hip'][hl], FEEDBACK['back'][bl]]
    feedback = " | ".join(m for m in msgs if m) or "Correct form ✓"
    print(f"  {desc}: {feedback}")
