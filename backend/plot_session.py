"""
plot_session.py
===============
Plots your last session (saved by squat_camera_v4.py in sessions/).

    python plot_session.py                 # latest session
    python plot_session.py sessions/session_20261002_181500_frames.csv

Shows knee / hip / back angles over time:
  • green band = counted rep | red band = not counted (reason written on it)
  • red dots   = frames where the model found an error for that joint
Saves a PNG next to the CSV.
"""

import sys, glob, os
import pandas as pd
import matplotlib.pyplot as plt

VALID_DEPTH_KNEE = 115

path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("sessions/*_frames.csv"), key=os.path.getmtime)
frames = pd.read_csv(path)
reps_path = path.replace("_frames.csv", "_reps.csv")
reps = pd.read_csv(reps_path) if os.path.exists(reps_path) else pd.DataFrame()

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
for ax, joint in zip(axes, ['knee', 'hip', 'back']):
    ax.plot(frames.time_s, frames[joint], color='#2a6fdb', lw=1.5)
    err = frames[frames[f"{joint}_label"].fillna(0) > 0]
    ax.scatter(err.time_s, err[joint], color='#d62728', s=10, zorder=3, label='error')
    ax.set_ylabel(f"{joint} (deg)")
    ax.grid(alpha=0.3)
    for _, r in reps.iterrows():
        ax.axvspan(r.start_s, r.start_s + r.duration_s,
                   color='#2ca02c' if r.counted else '#d62728', alpha=0.12)
axes[0].axhline(VALID_DEPTH_KNEE, ls='--', color='gray', lw=1)
axes[0].text(frames.time_s.min(), VALID_DEPTH_KNEE + 2, f"depth target {VALID_DEPTH_KNEE}", color='gray', fontsize=8)
for _, r in reps.iterrows():
    label = "OK" if r.counted else str(r.reason)
    axes[0].text(r.start_s, axes[0].get_ylim()[1], label, fontsize=7, va='top',
                 color='#2ca02c' if r.counted else '#d62728')
axes[-1].set_xlabel("time (s)")
n_ok = int(reps.counted.sum()) if len(reps) else 0
fig.suptitle(f"{os.path.basename(path)}  -  counted {n_ok} / {len(reps)} attempts")
fig.tight_layout()

png = path.replace("_frames.csv", ".png")
fig.savefig(png, dpi=120)
print("Saved:", png)

if len(reps):
    print("\nPer rep:")
    print(reps.to_string(index=False))
    print("\nErrors per joint (frames):")
    for j in ['knee', 'hip', 'back']:
        print(f"  {j}: {(frames[f'{j}_label'].fillna(0) > 0).sum()}")
plt.show()
