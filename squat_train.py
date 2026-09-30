"""
Squat Form Classifier — Training Script
========================================
Dataset  : squat_dataset.csv  (6000 rows, 4 balanced classes)
Model    : RandomForestClassifier
Output   : squat_model.pkl

Classes:
    0 = Correct
    1 = Adjust Type-1  (knee valgus / too shallow / hip open / lean forward)
    2 = Adjust Type-2  (knee varus  / too deep   / hip closed / too upright)
    3 = Incorrect      (severe error — show demo video)

Features (5):
    knee_angle   — interior knee angle in degrees  (hip–knee–ankle)
    knee_cross   — cross-product z-value (+valgus / -varus)
    hip_angle    — interior hip angle in degrees   (shoulder–hip–knee)
    hip_cross    — lateral hip deviation (cross-product)
    back_angle   — torso lean from vertical in degrees

Usage:
    pip install scikit-learn pandas
    python squat_train.py
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. Load dataset ──────────────────────────────────────────────────────────

df = pd.read_csv('squat_dataset.csv')

print("=" * 55)
print("  SQUAT FORM CLASSIFIER — TRAINING")
print("=" * 55)
print(f"\nDataset  : {len(df)} rows,  {df.shape[1]} columns")
print(f"Classes  :\n{df['overall_label'].value_counts().sort_index().to_string()}\n")

# ── 2. Features & target ─────────────────────────────────────────────────────

FEATURES = ['knee_angle', 'knee_cross', 'hip_angle', 'hip_cross', 'back_angle']

X = df[FEATURES]
y = df['overall_label']

# ── 3. Train / test split (80 / 20, stratified) ──────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train set : {len(X_train)} rows")
print(f"Test  set : {len(X_test)} rows\n")

# ── 4. Train ─────────────────────────────────────────────────────────────────

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
print("Training complete.\n")

# ── 5. Evaluate ──────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)

CLASS_NAMES = ['Correct', 'Adjust-Type1', 'Adjust-Type2', 'Incorrect']

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

print("Confusion Matrix  (rows = actual, cols = predicted):")
cm = confusion_matrix(y_test, y_pred)
print(f"{'':16}" + "".join(f"{n:>14}" for n in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"  {CLASS_NAMES[i]:14}" + "".join(f"{v:>14}" for v in row))
print()

# ── 6. Feature importance ────────────────────────────────────────────────────

print("Feature Importances:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:15}  {imp:.4f}  {bar}")
print()

# ── 7. Save model ────────────────────────────────────────────────────────────

with open('squat_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved  →  squat_model.pkl\n")

# ── 8. Real-time inference function (copy this into your backend) ─────────────

FEEDBACK = {
    0: 'Correct form',
    1: 'Adjust: knee caving in / squat too shallow / hip too open / leaning too far forward',
    2: 'Adjust: knee bowing out / squat too deep / hip too closed / torso too upright',
    3: 'Incorrect: severe error detected — trigger demo video',
}

def predict_squat(knee_angle, knee_cross, hip_angle, hip_cross, back_angle):
    """
    Run inference on a single MediaPipe frame.

    Parameters
    ----------
    knee_angle  : float  — hip–knee–ankle interior angle (degrees)
    knee_cross  : float  — cross-product z-value (+inward / -outward)
    hip_angle   : float  — shoulder–hip–knee interior angle (degrees)
    hip_cross   : float  — hip lateral deviation cross-product
    back_angle  : float  — torso lean from vertical (degrees)

    Returns
    -------
    label    : int   — 0 / 1 / 2 / 3
    feedback : str   — human-readable correction message
    """
    x = [[knee_angle, knee_cross, hip_angle, hip_cross, back_angle]]
    label = int(model.predict(x)[0])
    return label, FEEDBACK[label]


# ── 9. Quick smoke test ───────────────────────────────────────────────────────

print("-" * 55)
print("INFERENCE SMOKE TEST")
print("-" * 55)

tests = [
    (93.0,  0.10, 88.0, 0.05, 42.0, "Correct form"),
    (108.0, 0.10, 88.0, 0.05, 42.0, "Too shallow  → Adj-1"),
    (93.0,  0.40, 88.0, 0.05, 42.0, "Valgus       → Adj-1"),
    (78.0,  0.10, 88.0, 0.05, 42.0, "Too deep     → Adj-2"),
    (93.0, -0.40, 88.0, 0.05, 42.0, "Varus        → Adj-2"),
    (60.0,  0.75, 50.0, 0.05, 75.0, "Multi severe → Incorrect"),
]

for ka, kc, ha, hc, ba, note in tests:
    label, msg = predict_squat(ka, kc, ha, hc, ba)
    status = "✓" if str(label) in note or "Correct" in note and label == 0 else "→"
    print(f"\n  {note}")
    print(f"    knee={ka}°  cross={kc:+.2f}  hip={ha}°  back={ba}°")
    print(f"    [{label}] {msg}")
