"""
Step 3b: calibration + live monitoring, tuned for faster, clearer nudges.

What changed from 03_calibrate_and_monitor.py
  - New head-bend check (uses face landmarks, so it does not depend on the shoulder points)
  - Demo mode nudges after 3 seconds; real mode uses survey-based timing (press m to switch)
  - Less smoothing (5 frames instead of 10) so the status reacts sooner
  - Debug panel (press d) shows each indicator against its limit, so you can tune thresholds
  - Hint when your shoulders are at the bottom edge of the frame

Keys
  c : calibrate (sit in your best posture, hold still 5 seconds)
  1 / 2 / 3 : sensitivity Low / Medium / High
  m : switch DEMO / REAL timing
  d : show / hide the debug panel
  p : pause / resume analysis
  q : quit

Privacy: frames are analysed in memory on this device and never saved.
Needs posture_metrics.py and posture_logic.py in the same folder.
"""
import time

import cv2
import mediapipe as mp

from posture_metrics import compute_indicators
from posture_logic import (SENSITIVITY_NAME, MESSAGES, BASE_THRESHOLDS, Smoother,
                           deviations, limits, make_baseline, problems)

# ---- Settings you can change ------------------------------------------------
CALIBRATION_SECONDS = 5
SMOOTHING_FRAMES = 5
MODEL_COMPLEXITY = 1   # 0 fast, 1 default, 2 most accurate (try 2 if shoulder points sit too low)

# seconds poor posture must last before a nudge, and the minimum gap between nudges
TIMING = {
    "DEMO": {"bad": 3, "cooldown": 15},
    "REAL": {"bad": 30, "cooldown": 900},   # survey: most want 15 to 30 minutes between reminders
}
NUDGE_SHOWN_SECONDS = 6
# -----------------------------------------------------------------------------

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
PL = mp_pose.PoseLandmark

GREEN, ORANGE, RED, WHITE, GREY = (0, 200, 0), (0, 165, 255), (0, 0, 255), (255, 255, 255), (200, 200, 200)
LABELS = {"head_bend": "head bend", "head_drop": "head drop", "lean_in": "lean in",
          "shoulder_tilt": "shoulder tilt", "head_tilt": "head tilt"}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam.")

state = "NEED_CALIBRATION"   # NEED_CALIBRATION -> CALIBRATING -> MONITORING
baseline = None
samples = []
calib_start = 0.0
message = ""
sensitivity = 2
mode = "DEMO"
paused = False
debug = True
smoother = Smoother(SMOOTHING_FRAMES)
bad_since = None
last_nudge = -1e9
nudge_text, nudge_until = "", 0.0


def put(frame, text, y, color=WHITE, scale=0.6, x=10):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


with mp_pose.Pose(model_complexity=MODEL_COMPLEXITY, smooth_landmarks=True,
                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()
        h, w = frame.shape[:2]

        ind, shoulder_y = None, None
        if not paused:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                ind = compute_indicators(lm, PL)
                shoulder_y = (lm[PL.LEFT_SHOULDER].y + lm[PL.RIGHT_SHOULDER].y) / 2

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
            dev = None
            if ind is None:
                status, color = "Cannot see you clearly", ORANGE
                bad_since = None
            else:
                dev = deviations(smoother.add(ind), baseline)
                bad = problems(dev, sensitivity)
                if bad:
                    status, color = "Drifting from your baseline", ORANGE
                    if bad_since is None:
                        bad_since = now
                    t = TIMING[mode]
                    if now - bad_since >= t["bad"] and now - last_nudge >= t["cooldown"]:
                        nudge_text = MESSAGES[bad[0]]
                        nudge_until = now + NUDGE_SHOWN_SECONDS
                        last_nudge = now
                else:
                    bad_since = None

            put(frame, status, 30, color, 0.8)
            t = TIMING[mode]
            put(frame, f"Sensitivity: {SENSITIVITY_NAME[sensitivity]}   Mode: {mode} (nudge after {t['bad']}s)", 58, WHITE, 0.5)

            if debug and dev:
                lim = limits(sensitivity)
                y = 84
                for name in BASE_THRESHOLDS:
                    if name in dev:
                        over = dev[name] > lim[name]
                        put(frame, f"{LABELS[name]:13s} {dev[name]:+.2f} / limit {lim[name]:.2f}", y,
                            ORANGE if over else GREEN, 0.5)
                    else:
                        put(frame, f"{LABELS[name]:13s} not visible", y, GREY, 0.5)
                    y += 20

        # framing hint: shoulder points near the bottom edge are unreliable
        if shoulder_y is not None and shoulder_y > 0.88 and not paused:
            put(frame, "Shoulders near bottom edge: tilt screen back or sit farther away", h - 100, ORANGE, 0.5)

        # gentle nudge banner (no sound, no popup window)
        if now < nudge_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 120, 255), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            put(frame, nudge_text, h - 22, WHITE, 0.75)

        put(frame, "Processed on this device. No video is saved.", h - 80, GREY, 0.45)
        put(frame, "c calibrate | 1/2/3 sensitivity | m mode | d debug | p pause | q quit", h - 66, GREY, 0.4)

        cv2.imshow("Posture monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") and not paused:
            state, samples, calib_start, message = "CALIBRATING", [], now, ""
        elif key == ord("p"):
            paused = not paused
        elif key == ord("m"):
            mode = "REAL" if mode == "DEMO" else "DEMO"
        elif key == ord("d"):
            debug = not debug
        elif key in (ord("1"), ord("2"), ord("3")):
            sensitivity = int(chr(key))

cap.release()
cv2.destroyAllWindows()
