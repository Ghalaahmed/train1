import cv2
import mediapipe as mp
import numpy as np
import requests
import os

MODEL_PATH = "pose_landmarker_full.task"

BaseOptions         = mp.tasks.BaseOptions
PoseLandmarker     = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOpts  = mp.tasks.vision.PoseLandmarkerOptions
RunningMode         = mp.tasks.vision.RunningMode

options = PoseLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
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
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def cross_z(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    return float(ba[0]*bc[1] - ba[1]*bc[0])

COLORS = {0:(0,200,0), 1:(0,200,200), 2:(0,140,255), 3:(0,0,220)}
LABELS = {
    0: 'Correct form ✓',
    1: 'Adjust: knee caving / too shallow',
    2: 'Adjust: knee bowing / too deep',
    3: 'INCORRECT — Check form!',
}

# ── Rep counter state ──────────────────────────────────────────────
rep_count       = 0
phase           = 'up'        # 'up' | 'down'
error_streak    = 0           # consecutive error frames
had_error_in_rep = False      # any error during current rep
SQUAT_THRESHOLD = 110         # knee angle below → considered DOWN
STAND_THRESHOLD = 155         # knee angle above → considered UP
KNEE_IDX        = 26
# ──────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as lmker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = lmker.detect(mp_img)

        feedback = 'Stand in front of camera'
        color    = (80, 80, 80)
        label    = -1

        if result.pose_landmarks:
            L      = result.pose_landmarks[0]
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in L]

            # Draw skeleton
            for (i, j) in CONNECTIONS:
                if i < len(coords) and j < len(coords):
                    cv2.line(frame, coords[i], coords[j], (255, 80, 0), 3)
            for (cx, cy) in coords:
                cv2.circle(frame, (cx, cy), 10, (255, 80, 0), -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)

            def pt(idx):
                return [L[idx].x*w, L[idx].y*h, L[idx].z]

            shoulder   = pt(12)
            hip        = pt(24)
            knee       = pt(26)
            ankle      = pt(28)

            knee_angle = angle(hip, knee, ankle)
            hip_angle  = angle(shoulder, hip, knee)
            torso      = np.array(shoulder[:2]) - np.array(hip[:2])
            cos_b      = np.dot(torso,[0,-1])/(np.linalg.norm(torso)+1e-6)
            back_angle = float(np.degrees(np.arccos(np.clip(cos_b,-1,1))))
            knee_cross = cross_z(hip, knee, ankle)
            hip_cross  = cross_z(shoulder, hip, knee)
            print(f"knee:{knee_angle:.1f} hip:{hip_angle:.1f} back:{back_angle:.1f} kc:{knee_cross:.1f} hc:{hip_cross:.1f}")

            try:
                r = requests.post(
                    'http://127.0.0.1:8000/predict',
                    json={'knee_angle':knee_angle,'knee_cross':knee_cross,
                          'hip_angle':hip_angle,'hip_cross':hip_cross,
                          'back_angle':back_angle},
                    timeout=0.15
                )
                d     = r.json()
                label = d['label']
                feedback = LABELS.get(label, d['feedback'])
                color    = COLORS.get(label, (80,80,80))
            except Exception:
                feedback = 'API unavailable'
                color    = (80,80,80)

            # ── Rep counting logic ────────────────────────────────
            if label == 0:
                error_streak = 0
            elif label in (1, 2, 3):
                error_streak += 1
                had_error_in_rep = True

            if label != -1:
                if phase == 'up' and knee_angle < SQUAT_THRESHOLD:
                    phase = 'down'

                elif phase == 'down' and knee_angle > STAND_THRESHOLD:
                    phase = 'up'
                    if not had_error_in_rep:
                        rep_count += 1
                    had_error_in_rep = False
            # ─────────────────────────────────────────────────────

            # Angle labels
            for name, val, idx in [('Knee', knee_angle, 26),
                                    ('Hip',  hip_angle,  24),
                                    ('Back', back_angle, 12)]:
                cx, cy = coords[idx]
                cv2.putText(frame, f'{name}: {val:.0f}°',
                            (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255,255,255), 2)

        # ── Feedback bar ──────────────────────────────────────────
        cv2.rectangle(frame, (0,0), (w,55), color, -1)
        cv2.putText(frame, feedback, (12,38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

        # ── Rep counter box (top right) ───────────────────────────
        box_x, box_y = w - 160, 65
        cv2.rectangle(frame, (box_x, box_y), (w-10, box_y+85), (30,30,30), -1)
        cv2.rectangle(frame, (box_x, box_y), (w-10, box_y+85), (0,180,255), 2)
        cv2.putText(frame, 'REPS', (box_x+42, box_y+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 1)
        cv2.putText(frame, str(rep_count), (box_x+50, box_y+72),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 3)

        # Phase indicator
        phase_color = (0,200,0) if phase == 'down' else (200,200,200)
        cv2.putText(frame, phase.upper(), (12, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_color, 2)

        # Error streak warning
        if error_streak >= 3:
            cv2.rectangle(frame, (0, h-45), (w, h), (0,0,180), -1)
            cv2.putText(frame, '⚠ Errors detected — watch your form!',
                        (12, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, 'Q to quit', (w-100,h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        cv2.imshow('Squat Analyzer', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
