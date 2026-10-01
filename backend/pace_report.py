"""
pace_report.py
==============
Closed-loop step: Measure → Remember → Identify weakness → Measurable goal → Re-measure → Verify → Adapt

Reads a session saved by squat_camera_v4.py (sessions/*_frames.csv + *_reps.csv) and:
  1. computes the rep-level and session metrics (same names as the app design)
  2. saves a session report (JSON) in profiles/<user>/sessions/
  3. checks last session's goal → achieved or not
  4. sets the next measurable goal (or moves the focus when the weakness is fixed)

Run:
    python pace_report.py                    # latest session, user "demo"
    python pace_report.py --user sarah
    python pace_report.py --frames sessions/session_..._frames.csv --user sarah
squat_camera_v4.py also calls it automatically when you press Q.

────────────────────────── METRIC DEFINITIONS ──────────────────────────
Per rep (one "attempt" = one IN_REP segment):
  Range of Motion   = (180 - min_knee) / (180 - 90)      → 90 deg knee = 100% (capped)
  Knee / Hip / Back = % of rep frames where the model said 0 (correct) for that joint
  Movement Control  = descent time / 1.0 s               → lowering in >= 1 s = 100% (capped)
  Warning (joint)   = that joint had an error lasting >= 0.15 s in the rep
  Rep Score         = average of the 5 scores above
Per session:
  each score  = average over reps
  Form Score  = average Rep Score
  Rep Completion = counted reps / attempts
  Warnings    = number of reps with a warning, per joint
Main focus = joint with the most warnings (tie → lowest score);
             if no warnings → the lowest score.
Goal       = focus warnings: N → floor(N * 0.6) or fewer   (5 → 3 → 1)
             achieved with <= 1 warning → focus resolved, move to the next weakness
"""

import os, sys, json, glob, math, argparse, time
import numpy as np
import pandas as pd

def read_model_version():
    """Which model is in use: current_model.txt → experiments/<exp_id>/info.json (train_experiment.py)."""
    try:
        with open("current_model.txt") as f:
            folder = f.read().strip()
        with open(os.path.join(folder, "info.json"), encoding="utf-8") as f:
            return json.load(f)["exp_id"]
    except Exception:
        return "unknown"

MODEL_VERSION   = read_model_version()
JOINTS          = ['knee', 'hip', 'back']
ERROR_FAIL_S    = 0.15
DEPTH_TARGET    = 90
CONTROL_DESCENT_S = 1.0
GOAL_FACTOR     = 0.6
RESOLVED_AT     = 1

NAMES_AR = {'knee': 'الركبة', 'hip': 'الورك', 'back': 'الظهر',
            'range_of_motion': 'مدى الحركة', 'movement_control': 'التحكم بالحركة'}


# ───────────────────────── 1. metrics ─────────────────────────
def _lasting(times, bad):
    """True if 'bad' stays True for >= ERROR_FAIL_S."""
    start = None
    for t, b in zip(times, bad):
        if b:
            start = t if start is None else start
            if t - start >= ERROR_FAIL_S:
                return True
        else:
            start = None
    return False


def rep_metrics(frames, reps):
    f = frames.copy()
    f['in_rep'] = f.state.eq('IN_REP')
    f['seg'] = (f.in_rep != f.in_rep.shift()).cumsum()
    segs = [g for _, g in f[f.in_rep].groupby('seg')]

    rows = []
    for i, g in enumerate(segs):
        info = reps.iloc[i] if i < len(reps) else None
        min_knee = g.knee.min()
        rom = min(max((180 - min_knee) / (180 - DEPTH_TARGET), 0), 1) * 100
        descent = g.time_s.loc[g.knee.idxmin()] - g.time_s.iloc[0]
        control = min(descent / CONTROL_DESCENT_S, 1) * 100
        row = dict(rep=i + 1, min_knee=round(min_knee, 1), range_of_motion=round(rom, 1),
                   movement_control=round(control, 1), descent_s=round(descent, 2))
        for j in JOINTS:
            lab = g[f"{j}_label"].fillna(0)
            row[j] = round((lab == 0).mean() * 100, 1)
            row[f"{j}_warning"] = _lasting(g.time_s.values, (lab > 0).values)
        row['rep_score'] = round(np.mean([row['range_of_motion'], row['movement_control'],
                                          *[row[j] for j in JOINTS]]), 1)
        row['counted'] = bool(info.counted) if info is not None else False
        row['reason'] = (info.reason if info is not None and isinstance(info.reason, str) else "")
        rows.append(row)
    return pd.DataFrame(rows)


def session_metrics(r):
    if r.empty:
        return None
    scores = {k: round(r[k].mean(), 1) for k in ['range_of_motion', 'movement_control', *JOINTS]}
    warnings = {j: int(r[f"{j}_warning"].sum()) for j in JOINTS}

    # main focus: most warnings, tie → lowest score
    if max(warnings.values()) > 0:
        focus = max(JOINTS, key=lambda j: (warnings[j], -scores[j]))
    else:
        focus = min(scores, key=scores.get)
    strongest = max(scores, key=scores.get)

    return dict(reps_attempted=len(r), reps_counted=int(r.counted.sum()),
                rep_completion=round(r.counted.mean() * 100, 1),
                form_score=round(r.rep_score.mean(), 1),
                scores=scores, warnings=warnings,
                main_focus=focus, strongest=strongest,
                worst_rep=int(r.loc[r.rep_score.idxmin(), 'rep']))


# ───────────────────────── 2-4. profile + goal loop ─────────────────────────
def load_profile(user):
    path = os.path.join("profiles", user, "profile.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f), path
    return {"user": user, "exercises": {}}, path


def make_goal(metric, current):
    if metric in JOINTS:
        return dict(metric=metric, kind="warnings", baseline=current,
                    target=int(math.floor(current * GOAL_FACTOR)))
    # score metric (no warnings): +10 points, max 100
    return dict(metric=metric, kind="score", baseline=current, target=min(100.0, round(current + 10, 1)))


def check_goal(goal, s):
    if goal['kind'] == "warnings":
        value = s['warnings'][goal['metric']]
        return value, value <= goal['target']
    value = s['scores'][goal['metric']]
    return value, value >= goal['target']


def coach_text(s, goal_result, new_goal):
    """Short Arabic summary — this is what Pace Coach (RAG) will build on."""
    lines = []
    f, n = s['main_focus'], s['reps_attempted']
    if goal_result:
        g, value, ok = goal_result
        name = NAMES_AR[g['metric']]
        if ok:
            lines.append(f"✓ حققتِ هدف الجلسة: {name} من {g['baseline']} إلى {value}.")
        else:
            lines.append(f"✗ ما تحقق الهدف بعد: {name} {value} والهدف {g['target']}. بنكمل على نفس النقطة.")
    if s['warnings'][f] if f in JOINTS else False:
        lines.append(f"أكثر نقطة احتاجت تعديل: {NAMES_AR[f]} — ظهرت في {s['warnings'][f]} من {n} تكرار.")
    else:
        lines.append(f"أضعف مؤشر: {NAMES_AR[f]} ({s['scores'][f]}%).")
    lines.append(f"أقوى نقطة: {NAMES_AR[s['strongest']]} ({s['scores'][s['strongest']]}%).")
    if new_goal:
        if new_goal['kind'] == "warnings":
            lines.append(f"هدف الجلسة القادمة: تنبيهات {NAMES_AR[new_goal['metric']]} "
                         f"{new_goal['baseline']} ← {new_goal['target']} أو أقل.")
        else:
            lines.append(f"هدف الجلسة القادمة: {NAMES_AR[new_goal['metric']]} "
                         f"{new_goal['baseline']}% ← {new_goal['target']}% أو أكثر.")
    return "\n".join(lines)


def build_and_save(frames_path, reps_path, user="demo", exercise="squat"):
    frames = pd.read_csv(frames_path)
    reps = pd.read_csv(reps_path) if os.path.exists(reps_path) else pd.DataFrame()
    r = rep_metrics(frames, reps)
    s = session_metrics(r)
    if s is None:
        print("No reps in this session - no report.")
        return None

    profile, profile_path = load_profile(user)
    ex = profile["exercises"].setdefault(exercise, {"history": [], "goal": None, "resolved": []})

    # 3. verify last goal (only if same model version → same measurement)
    goal_result = None
    if ex["goal"] and ex["goal"].get("model_version") == MODEL_VERSION:
        value, ok = check_goal(ex["goal"], s)
        goal_result = (ex["goal"], value, ok)

    # 4. next goal
    if goal_result and goal_result[2]:
        g, value = goal_result[0], goal_result[1]
        resolved = (g['kind'] == "warnings" and value <= RESOLVED_AT) or \
                   (g['kind'] == "score" and value >= 90)
        if resolved:                                   # weakness fixed → move focus
            ex["resolved"].append(g['metric'])
            focus = s['main_focus'] if s['main_focus'] not in ex["resolved"] else \
                min((m for m in s['scores'] if m not in ex["resolved"]), key=s['scores'].get, default=None)
            new_goal = make_goal(focus, s['warnings'][focus] if focus in JOINTS else s['scores'][focus]) if focus else None
        else:                                          # keep focus, push further
            new_goal = make_goal(g['metric'], value)
    elif goal_result:                                  # not achieved → same target again
        new_goal = dict(goal_result[0])
    else:                                              # first session
        f = s['main_focus']
        new_goal = make_goal(f, s['warnings'][f] if f in JOINTS else s['scores'][f])
    if new_goal:
        new_goal['model_version'] = MODEL_VERSION

    text = coach_text(s, goal_result, new_goal)
    if ex["goal"] and ex["goal"].get("model_version") != MODEL_VERSION:
        text = ("ملاحظة: المودل تغيّر منذ الجلسة السابقة، فهذه الجلسة بداية قياس جديدة "
                "(ما نقارنها بالقديمة).\n") + text

    # 2. save report
    report = dict(user=user, exercise=exercise, model_version=MODEL_VERSION,
                  date=time.strftime("%Y-%m-%d %H:%M"), source=os.path.basename(frames_path),
                  session=s,
                  goal_checked=(dict(goal_result[0], value=goal_result[1], achieved=goal_result[2])
                                if goal_result else None),
                  next_goal=new_goal, coach_summary=text,
                  reps=r.to_dict(orient="records"))
    folder = os.path.join("profiles", user, "sessions")
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, os.path.basename(frames_path).replace("_frames.csv", ".json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    ex["history"].append(dict(date=report["date"], report=out, form_score=s['form_score'],
                              scores=s['scores'], warnings=s['warnings']))
    ex["goal"] = new_goal
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print_report(s, text, out)
    return report


def print_report(s, text, out):
    print("\n" + "=" * 50 + "\nSESSION REPORT\n" + "=" * 50)
    print(f"Reps            : {s['reps_counted']} counted / {s['reps_attempted']} attempts "
          f"({s['rep_completion']}%)")
    print(f"Form Score      : {s['form_score']}%")
    for k, v in s['scores'].items():
        tag = "  <- main focus" if k == s['main_focus'] else ("  <- strongest" if k == s['strongest'] else "")
        print(f"  {k:17s}: {v}%{tag}")
    print("Warnings        : " + ", ".join(f"{j} {n}" for j, n in s['warnings'].items()))
    print("\n" + text)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="demo")
    ap.add_argument("--exercise", default="squat")
    ap.add_argument("--frames", default=None)
    a = ap.parse_args()
    frames_path = a.frames or max(glob.glob("sessions/*_frames.csv"), key=os.path.getmtime)
    build_and_save(frames_path, frames_path.replace("_frames.csv", "_reps.csv"), a.user, a.exercise)
