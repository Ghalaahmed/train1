"""
train_squat.py
==============
Squat model training script — run once to generate squat_model.pkl
Features : knee_angle, hip_angle, back_angle
Labels   : 0=Correct  1=Adj1(too shallow)  2=Adj2(too deep)  3=Incorrect
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ── 1. Load dataset ───────────────────────────────────────────────
df = pd.read_csv("squat_dataset_angles_only.csv")

FEATURES = ['knee_angle', 'hip_angle', 'back_angle']
TARGET   = 'overall_label'

X = df[FEATURES]
y = df[TARGET]

print(f"Dataset : {len(df)} rows")
print(f"Labels  : {y.value_counts().sort_index().to_dict()}")

# ── 2. Train / test split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ── 3. Train ──────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("\nTraining...")
model.fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Correct', 'Adj1:Too_shallow', 'Adj2:Too_deep', 'Incorrect']
))

# ── 5. Save model ─────────────────────────────────────────────────
with open("squat_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved: squat_model.pkl")
