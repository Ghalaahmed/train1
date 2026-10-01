"""
evaluate_model.py
=================
Benchmark & evaluation for the squat phases model.

Run from backend/ with squat_env activated:
    python evaluate_model.py

What it measures
----------------
1. Fair split      : train/test split BY PERSON (test people never seen in training)
                     vs. the random split used in train_squat3.py (shows data leakage)
2. Per-joint scores: accuracy, balanced accuracy, macro F1, recall per class, confusion matrix
3. Baseline        : a "dumb" model that always predicts 'Correct' - our model must beat it
4. Rep-level       : does the system judge the WHOLE REP correctly? (same rule as the camera:
                     a rep fails if an error lasts >= 5 consecutive frames)
5. Speed           : prediction time of squat_model.pkl (camera waits max 150 ms)
"""

import time, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, confusion_matrix)

DATA      = "squat_dataset_phases.csv"
MODEL     = "squat_model.pkl"
FEATURES  = ['knee_angle', 'hip_angle', 'back_angle', 'phase_id']
TARGETS   = ['knee_label', 'hip_label', 'back_label']
LABELS    = [0, 1, 2, 3]
NAMES     = ['Correct', 'Adj1', 'Adj2', 'Incorrect']
ERROR_FRAMES = 5   # same as squat_camera_v3.py

df = pd.read_csv(DATA)
X, y = df[FEATURES], df[TARGETS]
print(f"Dataset: {len(df)} frames | {df.person_id.nunique()} people | "
      f"{df.groupby(['person_id','rep_id']).ngroups} reps\n")


def new_model():
    # same settings as train_squat3.py
    base = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    return MultiOutputClassifier(base)


def scores(y_true, y_pred):
    rows = []
    for i, t in enumerate(TARGETS):
        yt, yp = y_true.iloc[:, i], y_pred[:, i]
        rows.append({
            'joint': t.replace('_label', ''),
            'accuracy':     accuracy_score(yt, yp),
            'balanced_acc': balanced_accuracy_score(yt, yp),
            'macro_F1':     f1_score(yt, yp, average='macro', labels=LABELS, zero_division=0),
        })
    return pd.DataFrame(rows).set_index('joint')


def line(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


# ── 1. Random split (what train_squat3.py does) ───────────────────
line("1) RANDOM split (frames from the same person in train AND test)")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m = new_model().fit(Xtr, ytr)
random_scores = scores(yte, m.predict(Xte))
print((random_scores * 100).round(1).to_string())

# ── 2. Fair split by person ───────────────────────────────────────
line("2) PERSON split (test people never seen in training) <- the real score")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups=df.person_id))
Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]
print(f"Train people: {df.person_id.iloc[tr_idx].nunique()} | "
      f"Test people: {df.person_id.iloc[te_idx].nunique()}")
m = new_model().fit(Xtr, ytr)
y_pred = m.predict(Xte)
fair_scores = scores(yte, y_pred)
print((fair_scores * 100).round(1).to_string())

# ── 3. Per-class recall + confusion matrices ──────────────────────
line("3) Per-class recall (person split) - how many errors did we catch?")
for i, t in enumerate(TARGETS):
    yt, yp = yte.iloc[:, i], y_pred[:, i]
    rec = recall_score(yt, yp, labels=LABELS, average=None, zero_division=0)
    print(f"\n{t}: " + "  ".join(f"{n}={r*100:.0f}%" for n, r in zip(NAMES, rec)))
    cm = pd.DataFrame(confusion_matrix(yt, yp, labels=LABELS),
                      index=[f"true {n}" for n in NAMES],
                      columns=[f"pred {n}" for n in NAMES])
    print(cm.to_string())

# ── 4. Baseline ───────────────────────────────────────────────────
line("4) BASELINE - always predicts 'Correct'")
dummy = MultiOutputClassifier(DummyClassifier(strategy='most_frequent')).fit(Xtr, ytr)
base_scores = scores(yte, dummy.predict(Xte))
print((base_scores * 100).round(1).to_string())
print("\n-> If our accuracy is close to this, accuracy alone is misleading. "
      "Look at balanced_acc and macro_F1.")

# ── 5. Rep-level evaluation ───────────────────────────────────────
line(f"5) REP level - rep fails if an error lasts >= {ERROR_FRAMES} frames")
test = df.iloc[te_idx].copy()
test['pred_overall'] = y_pred.max(axis=1)

def rep_ok(errors):
    streak = 0
    for e in errors:
        streak = streak + 1 if e else 0
        if streak >= ERROR_FRAMES:
            return 0
    return 1

reps = (test.sort_values('frame')
            .groupby(['person_id', 'rep_id'])
            .agg(true_ok=('rep_correct', 'first'),
                 rep_type=('rep_type', 'first'),
                 pred_ok=('pred_overall', lambda s: rep_ok(s.values > 0))))
print(f"Reps tested: {len(reps)}")
print(f"Rep accuracy: {accuracy_score(reps.true_ok, reps.pred_ok)*100:.1f}%")
cm = pd.DataFrame(confusion_matrix(reps.true_ok, reps.pred_ok, labels=[1, 0]),
                  index=['true correct rep', 'true wrong rep'],
                  columns=['counted', 'rejected'])
print(cm.to_string())
print("\nAccuracy by rep type:")
print((reps.assign(hit=reps.true_ok == reps.pred_ok)
           .groupby('rep_type').hit.mean() * 100).round(1).to_string())

# ── 6. Speed of the saved model ───────────────────────────────────
line(f"6) SPEED of {MODEL} (one frame, like the API)")
with open(MODEL, 'rb') as f:
    saved = pickle.load(f)
one = X.iloc[[0]]
saved.predict(one)                                   # warm-up
times = []
for _ in range(100):
    t0 = time.perf_counter(); saved.predict(one); times.append((time.perf_counter() - t0) * 1000)
print(f"mean: {np.mean(times):.1f} ms | p95: {np.percentile(times, 95):.1f} ms "
      f"| camera timeout: 150 ms")

for est in saved.estimators_:                        # try single-thread
    est.set_params(n_jobs=1)
times1 = []
for _ in range(100):
    t0 = time.perf_counter(); saved.predict(one); times1.append((time.perf_counter() - t0) * 1000)
print(f"with n_jobs=1 -> mean: {np.mean(times1):.1f} ms | p95: {np.percentile(times1, 95):.1f} ms")

# ── Summary ───────────────────────────────────────────────────────
line("SUMMARY")
summary = pd.concat({'random split': random_scores['macro_F1'],
                     'person split': fair_scores['macro_F1'],
                     'baseline':     base_scores['macro_F1']}, axis=1)
print("macro F1 (%):")
print((summary * 100).round(1).to_string())
