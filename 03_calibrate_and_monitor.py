"""
Step 3: calibration + live posture monitoring with gentle nudges.

Keys
  c : start calibration (sit in your best posture and hold still for 5 seconds)
  1 / 2 / 3 : sensitivity Low / Medium / High
  p : pause / resume analysis
  q : quit

Privacy: frames are analysed in memory on this device and are never saved.

Needs posture_metrics.py and posture_logic.py in the same folder.
"""
import time

import cv2
import mediapipe as mp

from posture_metrics import compute_indicators
from posture_logic import (SENSITIVITY_NAME, MESSAGES, Smoother,
                           deviations, make_baseline, problems)

# ---- Settings you can change ------------------------------------------------
CALIBRATION_SECONDS = 5
BAD_SECONDS = 10      # poor posture must last this long before a nudge (use 30-60 in real use)
COOLDOWN_SECONDS = 30 # minimum gap between nudges, so the tool is not annoying
NUDGE_SHOWN_SECONDS = 6
# -----------------------------------------------------------------------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
PL = mp_pose.PoseLandmark

GREEN, ORANGE, RED, WHITE = (0, 200, 0), (0, 165, 255), (0, 0, 255), (255, 255, 255)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam.")

state = "NEED_CALIBRATION"   # NEED_CALIBRATION -> CALIBRATING -> MONITORING
baseline = None
samples = []
calib_start = 0.0
message = ""
sensitivity = 2
paused = False
smoother = Smoother(10)
bad_since = None
last_nudge = 0.0
nudge_text, nudge_until = "", 0.0


def put(frame, text, y, color=WHITE, scale=0.6):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        h, w = frame.shape[:2]

        ind = None
        if not paused:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                ind = compute_indicators(results.pose_landmarks.landmark, PL)

        # ---------------- state handling ----------------
        if paused:
            put(frame, "PAUSED - analysis stopped (press p to resume)", 30, ORANGE)

        elif state == "NEED_CALIBRATION":
            put(frame, "Sit in your BEST posture, then press 'c' to calibrate", 30, ORANGE)
            if message:
                put(frame, message, 60, RED)

        elif state == "CALIBRATING":
            left = CALIBRATION_SECONDS - (now - calib_start)
            put(frame, f"Calibrating... hold still ({max(left, 0):.0f}s)", 30, ORANGE, 0.8)
            if ind:
                samples.append(ind)
            if left <= 0:
                try:
                    baseline = make_baseline(samples)
                    state, message = "MONITORING", ""
                    smoother.reset()
                    bad_since = None
                except ValueError as err:
                    state, message = "NEED_CALIBRATION", str(err)

        elif state == "MONITORING":
            status, color = "Good posture", GREEN
            if ind is None:
                status, color = "Cannot see you clearly", ORANGE
                bad_since = None
            else:
                smooth = smoother.add(ind)
                bad = problems(deviations(smooth, baseline), sensitivity)
                if bad:
                    status, color = "Drifting from your baseline", ORANGE
                    if bad_since is None:
                        bad_since = now
                    long_enough = now - bad_since >= BAD_SECONDS
                    cooled_down = now - last_nudge >= COOLDOWN_SECONDS
                    if long_enough and cooled_down:
                        nudge_text = MESSAGES[bad[0]]
                        nudge_until = now + NUDGE_SHOWN_SECONDS
                        last_nudge = now
                else:
                    bad_since = None

            put(frame, status, 30, color, 0.8)
            put(frame, f"Sensitivity: {SENSITIVITY_NAME[sensitivity]} (keys 1/2/3)", 60)

        # gentle nudge banner (no sound, no popup window)
        if now < nudge_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 120, 255), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            put(frame, nudge_text, h - 28, WHITE, 0.75)

        put(frame, "Processed on this device. No video is saved.", h - 80 if now >= nudge_until else h - 85,
            (200, 200, 200), 0.45)

        cv2.imshow("Posture monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and not paused:
            state, samples, calib_start, message = "CALIBRATING", [], now, ""
        elif key == ord("p"):
            paused = not paused
        elif key in (ord("1"), ord("2"), ord("3")):
            sensitivity = int(chr(key))

cap.release()
cv2.destroyAllWindows()
