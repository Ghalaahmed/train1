"""
squat_camera_v4.py
==================
Squat analyzer — side view camera (uses the same API: main_v3.py)

What's new vs v3
----------------
1. NO FALSE REPS
   • Counting starts only after you stand straight, side-on, full body visible for 2 s ("Get ready")
   • If your body leaves the frame (e.g. fixing the camera) → counting pauses until you are ready again
   • A rep counts only if: knee reached depth AND hips really went down AND no lasting form error
   • When a rep is NOT counted, the screen tells you why (e.g. "not deep enough")

2. STABLE FEEDBACK (no flicker)
   • Angles are smoothed, labels use a majority vote over the last 0.5 s
   • An error is shown only if it lasts 0.4 s, then it STAYS on screen (min 3 s) until you fix it
   • When you fix it → "Correct position - continue" for 2 s
   • One message at a time (priority: back > knee > hip)

3. REAL-TIME ANGLES + SESSION LOG
   • Side panel: knee / hip / back angles colored by each joint's status, phase, last rep result
   • Every frame and every rep is saved in sessions/ (CSV) → analyse with plot_session.py
   • Summary printed in the terminal when you press Q

Run (2 terminals, inside backend/, squat_env active):
    uvicorn main_v3:app --port 8000
    python squat_camera_v4.py
"""

import os, csv, time
from collections import deque, Counter
import numpy as np

# ══════════════════════════ SETTINGS (edit here) ══════════════════════════
USER_ID          = "demo"  # whose profile the session report is saved to
API_URL          = 'http://127.0.0.1:8000/predict'
API_TIMEOUT      = 0.3     # seconds
POSE_MODEL       = "pose_landmarker_full.task"
SHOW_SKELETON    = True    # dev only — set False for a clean camera view

# Body detection
VIS_MIN          = 0.6     # landmark visibility needed (0..1)
READY_HOLD_S     = 2.0     # stand straight this long before counting starts
LOST_RESET_S     = 1.0     # body out of frame this long → back to "Get ready"
ANGLE_SMOOTH     = 0.4     # 0..1, lower = smoother angles (but slower)

# Rep counting
STAND_KNEE       = 160     # knee above this = standing
DESCENT_KNEE     = 140     # knee below this = rep started
VALID_DEPTH_KNEE = 115     # knee must reach this (or lower) to count
MIN_HIP_DROP     = 0.30    # hips must go down ≥ 30% of thigh length
MIN_REP_S        = 0.8     # faster than this = not a real rep
MAX_REP_S        = 15.0    # slower than this = not a real rep
ERROR_FAIL_S     = 0.15    # a form error lasting this long fails the rep (~5 frames)
PRE_REP_S        = 1.5     # an error while standing this close before the rep also fails it

# Feedback timing
LABEL_WINDOW_S    = 0.5    # majority vote window
ERROR_SHOW_S      = 0.4    # error must be stable this long before it is shown
ERROR_HOLD_S      = 3.0    # once shown, keep the message at least this long
CORRECT_CONFIRM_S = 0.8    # must be correct this long to clear the error
CONFIRM_SHOW_S    = 2.0    # show "Correct position - continue" this long
# ══════════════════════════════════════════════════════════════════════════

JOINTS   = ['knee', 'hip', 'back']
PRIORITY = ['back', 'knee', 'hip']          # safety first
PHASE_NAMES = {0: 'standing', 1: 'descent', 2: 'bottom', 3: 'ascent'}

# ASCII only — OpenCV cannot draw "°" or "—"
MESSAGES = {
    'knee': {1: "Squat deeper",
             2: "Too deep - stop at 90 deg",
             3: "Knee error - stop now!"},
    'hip':  {1: "Push your hips back",
             2: "Don't fold forward too much",
             3: "Hip error - stop now!"},
    'back': {1: "Raise your chest - back too low",
             2: "Lean forward slightly",
             3: "Back error - raise your chest now!"},
}

# BGR colors
COLORS = {-1: (80, 80, 80), 0: (0, 170, 0), 1: (0, 200, 220), 2: (0, 140, 255), 3: (0, 0, 220)}


# ─────────────────────────── helpers ───────────────────────────
def angle(a, b, c):
    """Angle at b (degrees) using the dot product."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class Smoother:
    """Exponential moving average for each angle."""
    def __init__(self, alpha=ANGLE_SMOOTH):
        self.alpha, self.v = alpha, {}

    def __call__(self, name, x):
        self.v[name] = x if name not in self.v else self.alpha * x + (1 - self.alpha) * self.v[name]
        return self.v[name]

    def reset(self):
        self.v = {}


class PhaseDetector:
    """0 standing | 1 descent | 2 bottom | 3 ascent — same rule as the dataset,
    but uses speed in deg/second (0.4 deg/frame at 30 fps = 12 deg/s) so it works at any FPS."""
    def __init__(self):
        self.hist, self.prev, self.prev_t = deque(maxlen=5), None, None

    def update(self, knee, t):
        self.hist.append(knee)
        smooth = sum(self.hist) / len(self.hist)
        if self.prev is None:
            self.prev, self.prev_t = smooth, t
        speed = (smooth - self.prev) / max(t - self.prev_t, 1e-3)
        self.prev, self.prev_t = smooth, t
        if smooth >= 160:     return 0
        if abs(speed) < 12:   return 2
        return 1 if speed < 0 else 3


class FeedbackStabilizer:
    """Turns noisy per-frame labels into calm, human-friendly feedback."""
    def __init__(self):
        self.hist = deque()                 # (t, {joint: label})
        self.mode = 'idle'                  # idle | error | confirm
        self.shown = None                   # (joint, label) currently on screen
        self.shown_at = self.cand_since = self.confirm_at = 0.0
        self.candidate = None
        self.correct_since = None
        self.stable = {j: 0 for j in JOINTS}
        self.started_at = None

    def reset(self):
        self.__init__()

    def _top_error(self, labels):
        errs = [(j, labels[j]) for j in PRIORITY if labels[j] > 0]
        if not errs:
            return None
        # severe (3) first, then priority order
        return max(errs, key=lambda e: (e[1] == 3, -PRIORITY.index(e[0])))

    def update(self, t, labels):
        self.hist.append((t, labels))
        self.started_at = self.started_at if self.started_at is not None else t
        while self.hist and t - self.hist[0][0] > LABEL_WINDOW_S:
            self.hist.popleft()
        self.stable = {j: Counter(l[j] for _, l in self.hist).most_common(1)[0][0] for j in JOINTS}

        err = self._top_error(self.stable)
        if err != self.candidate:
            self.candidate, self.cand_since = err, t
        lasted = t - self.cand_since

        if self.mode in ('idle', 'confirm'):
            warmed_up = t - self.started_at >= LABEL_WINDOW_S
            if err and lasted >= ERROR_SHOW_S and warmed_up:
                self.mode, self.shown, self.shown_at, self.correct_since = 'error', err, t, None
        elif self.mode == 'error':
            if err is None:
                self.correct_since = self.correct_since or t
                if t - self.shown_at >= ERROR_HOLD_S and t - self.correct_since >= CORRECT_CONFIRM_S:
                    self.mode, self.confirm_at, self.shown = 'confirm', t, None
            else:
                self.correct_since = None
                more_severe = err[1] == 3 and self.shown[1] != 3
                if err != self.shown and lasted >= ERROR_SHOW_S and \
                        (t - self.shown_at >= ERROR_HOLD_S or more_severe):
                    self.shown, self.shown_at = err, t

        if self.mode == 'confirm' and t - self.confirm_at >= CONFIRM_SHOW_S:
            self.mode = 'idle'
        return self.display()

    def display(self):
        """(text, label) for the top bar."""
        if self.mode == 'error':
            j, l = self.shown
            return MESSAGES[j][l], l
        if self.mode == 'confirm':
            return "Correct position - continue", 0
        return "Good form", 0


class RepCounter:
    """WAITING → STANDING → IN_REP → STANDING ..."""
    def __init__(self):
        self.state = 'WAITING'
        self.count = 0
        self.reps = []                       # one dict per attempt
        self.last = None                     # last rep result (for the screen)
        self.ready_since = self.lost_since = None
        self.stand_hip_y = self.thigh = None
        self._calib = []
        self.rep = None
        self.err_since = None             # raw error streak start
        self.stand_err = None             # (time, joints) of last lasting error while standing

    def _track_errors(self, t, raw):
        """Returns joints with an error that has lasted >= ERROR_FAIL_S (uses RAW model labels)."""
        bad = [j for j in JOINTS if raw and raw[j] > 0]
        if not bad:
            self.err_since = None
            return []
        self.err_since = self.err_since or t
        return bad if t - self.err_since >= ERROR_FAIL_S else []

    def update(self, t, visible, knee=None, hip_y=None, thigh=None, raw=None, back=None, hip=None):
        # ── body not visible ──
        if not visible:
            self.lost_since = self.lost_since or t
            if self.state != 'WAITING' and t - self.lost_since >= LOST_RESET_S:
                if self.rep:
                    self._finish(t, forced="left the frame")
                self.state, self.ready_since, self._calib = 'WAITING', None, []
            return
        self.lost_since = None

        # ── get ready: stand straight for READY_HOLD_S ──
        if self.state == 'WAITING':
            if knee >= STAND_KNEE:
                self.ready_since = self.ready_since or t
                self._calib.append((hip_y, thigh))
                if t - self.ready_since >= READY_HOLD_S:
                    self.stand_hip_y = float(np.median([c[0] for c in self._calib]))
                    self.thigh       = float(np.median([c[1] for c in self._calib]))
                    self.state = 'STANDING'
            else:
                self.ready_since, self._calib = None, []
            return

        lasting = self._track_errors(t, raw)

        if self.state == 'STANDING':
            if lasting:
                self.stand_err = (t, set(lasting))
            if knee >= STAND_KNEE:            # follow small position changes
                self.stand_hip_y = 0.9 * self.stand_hip_y + 0.1 * hip_y
            if knee < DESCENT_KNEE:
                self.state = 'IN_REP'
                self.rep = dict(start=t, min_knee=knee, min_hip=hip, max_back=back,
                                max_drop=0.0, err_joints=set(), failed=False)
                if self.stand_err and t - self.stand_err[0] <= PRE_REP_S:   # e.g. leaning before going down
                    self.rep['failed'] = True
                    self.rep['err_joints'].update(self.stand_err[1])
                self.stand_err = None
            return

        # ── IN_REP ──
        r = self.rep
        r['min_knee'] = min(r['min_knee'], knee)
        r['min_hip']  = min(r['min_hip'], hip)
        r['max_back'] = max(r['max_back'], back)
        r['max_drop'] = max(r['max_drop'], (hip_y - self.stand_hip_y) / max(self.thigh, 1e-6))

        if lasting:
            r['failed'] = True
            r['err_joints'].update(lasting)

        if knee >= STAND_KNEE:
            self._finish(t)
            self.state = 'STANDING'

    def _finish(self, t, forced=None):
        r, dur = self.rep, t - self.rep['start']
        if forced:                          reason = forced
        elif dur < MIN_REP_S:               reason = "too fast"
        elif dur > MAX_REP_S:               reason = "too slow"
        elif r['min_knee'] > VALID_DEPTH_KNEE: reason = "not deep enough"
        elif r['max_drop'] < MIN_HIP_DROP:  reason = "hips did not go down"
        elif r['failed']:                   reason = "form error: " + ", ".join(sorted(r['err_joints']))
        else:                               reason = None

        counted = reason is None
        if counted:
            self.count += 1
        result = dict(attempt=len(self.reps) + 1, start_s=round(r['start'], 2), duration_s=round(dur, 2),
                      min_knee=round(r['min_knee'], 1), min_hip=round(r['min_hip'], 1),
                      max_back=round(r['max_back'], 1), hip_drop=round(r['max_drop'], 2),
                      counted=counted, reason=reason or "")
        self.reps.append(result)
        self.last = dict(result, shown_at=t)
        self.rep = None


class SessionLogger:
    def __init__(self, folder="sessions"):
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.frames_path = os.path.join(folder, f"session_{stamp}_frames.csv")
        self.reps_path   = os.path.join(folder, f"session_{stamp}_reps.csv")
        self.f = open(self.frames_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["time_s", "state", "phase", "knee", "hip", "back",
                         "knee_label", "hip_label", "back_label", "message", "reps"])

    def frame(self, t, state, phase, angles, raw, message, reps):
        # raw = what the model said this frame (not smoothed) → best for analysis
        self.w.writerow([round(t, 3), state, "" if phase is None else PHASE_NAMES[phase],
                         *[round(angles[j], 1) if angles else "" for j in JOINTS],
                         *[raw[j] if raw else "" for j in JOINTS], message, reps])

    def close(self, reps):
        self.f.close()
        if reps:
            with open(self.reps_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(reps[0].keys()))
                w.writeheader(); w.writerows(reps)


def print_summary(counter, logger):
    reps = counter.reps
    print("\n" + "=" * 50 + "\nSESSION SUMMARY\n" + "=" * 50)
    print(f"Counted reps : {counter.count} / {len(reps)} attempts")
    if not reps:
        return
    reasons = Counter(r['reason'] for r in reps if not r['counted'])
    for reason, n in reasons.most_common():
        print(f"  not counted - {reason}: {n}")
    print(f"Avg depth (min knee) : {np.mean([r['min_knee'] for r in reps]):.0f} deg")
    print(f"Avg max back lean    : {np.mean([r['max_back'] for r in reps]):.0f} deg")
    print(f"\nSaved: {logger.frames_path}\n       {logger.reps_path}")
    print("Plot it:  python plot_session.py")


# ─────────────────────────── drawing ───────────────────────────
def put(frame, text, xy, scale=0.6, color=(255, 255, 255), thick=2):
    import cv2
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_panel(frame, angles, stable, phase, counter, api_ok):
    import cv2
    x, y = 10, 70
    cv2.rectangle(frame, (x, y), (x + 250, y + 200), (25, 25, 25), -1)
    put(frame, "ANGLES", (x + 10, y + 25), 0.55, (180, 180, 180), 1)
    for i, j in enumerate(JOINTS):
        yy = y + 55 + i * 30
        lab = stable[j] if stable else -1
        cv2.circle(frame, (x + 18, yy - 6), 7, COLORS[lab], -1)
        val = f"{angles[j]:.0f} deg" if angles else "--"
        put(frame, f"{j.capitalize():5s} {val}", (x + 35, yy), 0.6)
    put(frame, f"Phase: {PHASE_NAMES[phase] if phase is not None else '--'}", (x + 10, y + 155), 0.5, (200, 200, 200), 1)
    put(frame, f"State: {counter.state}", (x + 10, y + 180), 0.5, (200, 200, 200), 1)
    if not api_ok:
        put(frame, "API unavailable", (x + 10, y + 225), 0.55, (0, 0, 255), 2)


def draw_reps(frame, counter, t):
    import cv2
    h, w = frame.shape[:2]
    bx, by = w - 170, 70
    cv2.rectangle(frame, (bx, by), (w - 10, by + 90), (30, 30, 30), -1)
    cv2.rectangle(frame, (bx, by), (w - 10, by + 90), (0, 180, 255), 2)
    put(frame, "REPS", (bx + 50, by + 25), 0.6, (180, 180, 180), 1)
    put(frame, str(counter.count), (bx + 55, by + 75), 2.0, (255, 255, 255), 3)

    last = counter.last
    if last and t - last['shown_at'] < 2.5:          # flash the last rep result
        ok = last['counted']
        txt = f"Rep counted - depth {last['min_knee']:.0f} deg" if ok else f"Not counted: {last['reason']}"
        cv2.rectangle(frame, (0, h - 45), (w, h), (0, 140, 0) if ok else (0, 0, 170), -1)
        put(frame, txt, (12, h - 15), 0.65)


# ─────────────────────────── main loop ───────────────────────────
def main():
    import cv2, requests
    import mediapipe as mp

    BaseOptions  = mp.tasks.BaseOptions
    vision       = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL, delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,       # VIDEO = tracks between frames → less jitter
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
                   (23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]
    SIDES = {'right': (12, 24, 26, 28), 'left': (11, 23, 25, 27)}   # shoulder, hip, knee, ankle

    smoother, phases = Smoother(), PhaseDetector()
    feedback, counter = FeedbackStabilizer(), RepCounter()
    logger = SessionLogger()
    session = requests.Session()

    cap = cv2.VideoCapture(0)
    t0 = time.time()
    last_ts = -1

    with vision.PoseLandmarker.create_from_options(options) as lmker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time() - t0
            ts = max(int(t * 1000), last_ts + 1); last_ts = ts
            h, w = frame.shape[:2]

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = lmker.detect_for_video(mp_img, ts)

            angles = stable = raw = phase = None
            api_ok = True
            visible = False

            if result.pose_landmarks:
                L = result.pose_landmarks[0]
                # use the side facing the camera
                vis = lambda i: L[i].visibility if L[i].visibility is not None else 1.0
                side = max(SIDES, key=lambda s: sum(vis(i) for i in SIDES[s]))
                ids = SIDES[side]
                visible = all(vis(i) >= VIS_MIN and 0 <= L[i].x <= 1 and 0 <= L[i].y <= 1 for i in ids)

                if SHOW_SKELETON:
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in L]
                    for i, j in CONNECTIONS:
                        cv2.line(frame, pts[i], pts[j], (255, 80, 0), 2)
                    for i in ids:
                        cv2.circle(frame, pts[i], 7, (255, 255, 255), -1)

            if visible:
                sh, hp, kn, an = [(L[i].x * w, L[i].y * h) for i in ids]
                torso = np.array(sh) - np.array(hp)
                back_raw = float(np.degrees(np.arccos(np.clip(np.dot(torso, [0, -1]) / (np.linalg.norm(torso) + 1e-6), -1, 1))))
                angles = {'knee': smoother('knee', angle(hp, kn, an)),
                          'hip':  smoother('hip',  angle(sh, hp, kn)),
                          'back': smoother('back', back_raw)}
                phase = phases.update(angles['knee'], t)

                if counter.state != 'WAITING':
                    try:
                        r = session.post(API_URL, timeout=API_TIMEOUT, json={
                            'knee_angle': angles['knee'], 'hip_angle': angles['hip'],
                            'back_angle': angles['back'], 'phase_id': phase})
                        raw = r.json()['joint_labels']
                        feedback.update(t, raw)
                        stable = feedback.stable
                    except Exception:
                        api_ok = False

                thigh = float(np.hypot(hp[0] - kn[0], hp[1] - kn[1]))
                counter.update(t, True, knee=angles['knee'], hip_y=hp[1], thigh=thigh,
                               raw=raw, back=angles['back'], hip=angles['hip'])
            else:
                counter.update(t, False)
                if counter.state == 'WAITING':
                    smoother.reset(); feedback.reset()

            # ── top bar ──
            if counter.state == 'WAITING':
                if not visible:
                    msg, lab = "Step back - full body, side-on to the camera", -1
                else:
                    left = READY_HOLD_S - (t - counter.ready_since) if counter.ready_since else READY_HOLD_S
                    msg, lab = f"Stand straight... starting in {max(left, 0):.1f}s", -1
            else:
                msg, lab = feedback.display()
                if not visible:
                    msg, lab = "Body not fully visible", -1
            cv2.rectangle(frame, (0, 0), (w, 55), COLORS[lab], -1)
            put(frame, msg, (12, 38), 0.8)

            draw_panel(frame, angles, stable, phase, counter, api_ok)
            draw_reps(frame, counter, t)
            put(frame, "Q to quit", (w - 100, h - 55), 0.4, (150, 150, 150), 1)

            logger.frame(t, counter.state, phase, angles, raw, msg, counter.count)

            cv2.imshow('Squat Analyzer v4', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    logger.close(counter.reps)
    print_summary(counter, logger)

    # closed loop: metrics → report → goal check → next goal
    try:
        from pace_report import build_and_save
        build_and_save(logger.frames_path, logger.reps_path, user=USER_ID, exercise="squat")
    except Exception as e:
        print("Report not created:", e)


if __name__ == "__main__":
    main()
