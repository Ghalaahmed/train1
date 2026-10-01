"""
squat_camera_v3.py
==================
Squat analyzer — side view camera
State machine  : STANDING → DESCENDING → ASCENDING → STANDING
Feedback       : per-joint (knee / hip / back)
Rep counted only when: reached real depth AND no form error throughout entire rep

UPDATE (phases model):
  • get_phase() computes the movement phase (0 standing, 1 descent, 2 bottom, 3 ascent)
    with the same rule used to build squat_dataset_phases.csv
  • phase_id is sent to the API with the 3 angles (model now expects 4 inputs)
  • model is called in ALL phases → instant feedback anywhere (e.g. back lean mid-descent)
  • a rep is marked wrong only if an error lasts >= 5 consecutive frames (no flicker)
"""

import cv2
import mediapipe as mp
import numpy as np
import requests
from collections import deque

MODEL_PATH = "pose_landmarker_full.task"

BaseOptions         = mp.tasks.BaseOptions
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
RunningMode         = mp.tasks.vision.RunningMode

options = PoseLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=MODEL_PATH,delegate=BaseOptions.Delegate.CPU),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32),
]

def angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

# Feedback → color
def label_color(label):
    return {0: (0,200,0), 1: (0,200,200), 2: (0,140,255), 3: (0,0,220)}.get(label, (80,80,80))

# ── Phase detection (matches squat_dataset_phases.csv) ─────────────
# 0 = standing   : smoothed knee >= 160
# 2 = bottom     : knee almost not changing (< 0.4° per frame)
# 1 = descent    : knee decreasing
# 3 = ascent     : knee increasing
PHASE_NAMES  = {0: 'standing', 1: 'descent', 2: 'bottom', 3: 'ascent'}
knee_history = deque(maxlen=5)
prev_smooth  = None

def get_phase(knee):
    global prev_smooth
    knee_history.append(knee)
    smooth = sum(knee_history) / len(knee_history)
    if prev_smooth is None:
        prev_smooth = smooth
    change = smooth - prev_smooth
    prev_smooth = smooth
    if smooth >= 160:     return 0   # standing
    if abs(change) < 0.4: return 2   # bottom
    if change < 0:        return 1   # descent
    return 3                         # ascent

# ── State machine ──────────────────────────────────────────────────
# STANDING → DESCENDING → ASCENDING → STANDING
# Rep counted only when:
#   • reached real depth (min_knee_angle ≤ 115°)
#   • NO persistent form error (>= 5 consecutive frames) at any point during the rep

rep_count        = 0
state            = 'STANDING'
error_streak     = 0
min_knee_angle   = 180.0
had_error_in_rep = False   # True if an error lasted >= ERROR_FRAMES frames during the rep
prev_knee_angle  = 180.0

STAND_THRESHOLD   = 155   # above this → STANDING
DESCENT_TRIGGER   = 140   # below this → DESCENDING
VALID_DEPTH_MAX   = 115   # must reach this to count as real squat
ACTIVE_THRESHOLD  = 181   # call model in ALL phases (knee never reaches 181)
ERROR_FRAMES      = 5     # error must persist this many frames to fail a rep
# ──────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as lmker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = lmker.detect(mp_img)

        feedback = 'Stand in frame to begin'
        color    = (80, 80, 80)
        label    = -1
        phase    = None

        if result.pose_landmarks:
            L      = result.pose_landmarks[0]
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in L]

            for (i, j) in CONNECTIONS:
                if i < len(coords) and j < len(coords):
                    cv2.line(frame, coords[i], coords[j], (255, 80, 0), 3)
            for (cx, cy) in coords:
                cv2.circle(frame, (cx, cy), 10, (255, 80, 0), -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)

            def pt(idx):
                return [L[idx].x * w, L[idx].y * h, L[idx].z]

            shoulder = pt(12)
            hip      = pt(24)
            knee     = pt(26)
            ankle    = pt(28)

            knee_angle = angle(hip, knee, ankle)
            hip_angle  = angle(shoulder, hip, knee)
            torso      = np.array(shoulder[:2]) - np.array(hip[:2])
            cos_b      = np.dot(torso, [0, -1]) / (np.linalg.norm(torso) + 1e-6)
            back_angle = float(np.degrees(np.arccos(np.clip(cos_b, -1, 1))))

            # ── Movement phase (sent to the model as 4th input) ────
            phase = get_phase(knee_angle)

            # ── Call model (all phases) ────────────────────────────
            if knee_angle < ACTIVE_THRESHOLD:
                try:
                    r = requests.post(
                        'http://127.0.0.1:8000/predict',
                        json={
                            'knee_angle': knee_angle,
                            'hip_angle':  hip_angle,
                            'back_angle': back_angle,
                            'phase_id':   phase,
                        },
                        timeout=0.15,
                    )
                    d        = r.json()
                    label    = d['label']
                    feedback = d['feedback']
                    color    = label_color(label)

                except Exception:
                    feedback = 'API unavailable'
                    color    = (80, 80, 80)
            else:
                feedback = 'Ready — start your squat'
                color    = (60, 60, 60)
                label    = -1

            # ── State machine ──────────────────────────────────────
            if label in (1, 2, 3):
                error_streak += 1
            elif label == 0:
                error_streak = 0

            if state == 'STANDING':
                if knee_angle < DESCENT_TRIGGER:
                    state            = 'DESCENDING'
                    min_knee_angle   = knee_angle
                    had_error_in_rep = False        # reset for new rep

            elif state == 'DESCENDING':
                # Error only counts if it persists (not a single jittery frame)
                if error_streak >= ERROR_FRAMES:
                    had_error_in_rep = True
                if knee_angle <= min_knee_angle:
                    min_knee_angle = knee_angle
                else:
                    state = 'ASCENDING'

            elif state == 'ASCENDING':
                if error_streak >= ERROR_FRAMES:
                    had_error_in_rep = True
                if knee_angle > STAND_THRESHOLD:
                    state = 'STANDING'
                    # Count rep only if deep enough AND form was correct throughout
                    if min_knee_angle <= VALID_DEPTH_MAX and not had_error_in_rep:
                        rep_count += 1
                    min_knee_angle   = 180.0
                    had_error_in_rep = False
                elif knee_angle < prev_knee_angle - 5:
                    state = 'DESCENDING'

            prev_knee_angle = knee_angle
            # ──────────────────────────────────────────────────────

            # Angle labels on skeleton
            for name, val, idx in [('Knee', knee_angle, 26),
                                    ('Hip',  hip_angle,  24),
                                    ('Back', back_angle, 12)]:
                cx, cy = coords[idx]
                cv2.putText(frame, f'{name}: {val:.0f}°',
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)

        # ── Feedback bar ───────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 55), color, -1)
        cv2.putText(frame, feedback, (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # ── Rep counter ────────────────────────────────────────────
        box_x, box_y = w - 160, 65
        cv2.rectangle(frame, (box_x, box_y), (w - 10, box_y + 85), (30, 30, 30), -1)
        cv2.rectangle(frame, (box_x, box_y), (w - 10, box_y + 85), (0, 180, 255), 2)
        cv2.putText(frame, 'REPS', (box_x + 42, box_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(frame, str(rep_count), (box_x + 50, box_y + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

        # ── State + phase indicator ────────────────────────────────
        state_colors = {
            'STANDING':   (200, 200, 200),
            'DESCENDING': (0, 200, 255),
            'ASCENDING':  (0, 255, 100),
        }
        phase_txt = f'  | phase: {PHASE_NAMES[phase]}' if phase is not None else ''
        cv2.putText(frame, state + phase_txt, (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    state_colors.get(state, (200, 200, 200)), 2)

        # ── Error streak warning ───────────────────────────────────
        if error_streak >= 3:
            cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 180), -1)
            cv2.putText(frame, 'Consistent errors — check your form!',
                        (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        cv2.putText(frame, 'Q to quit', (w - 100, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow('Squat Analyzer v3', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()