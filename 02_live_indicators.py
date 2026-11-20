"""
Step 2: live posture indicators.

Shows four raw indicator values on top of the webcam view. Nothing is saved.
Use it to see how the numbers change when you slouch, lean in, tilt your head
or raise one shoulder. Press 'q' to quit.

Needs posture_metrics.py in the same folder.
"""
import cv2
import mediapipe as mp

from posture_metrics import compute_indicators

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
PL = mp_pose.PoseLandmark

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        lines = ["No person detected"]
        color = (0, 0, 255)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            ind = compute_indicators(results.pose_landmarks.landmark, PL)
            if ind:
                color = (0, 255, 0)
                lines = [
                    f"Shoulder tilt   : {ind['shoulder_tilt_deg']:5.1f} deg",
                    f"Head tilt       : {ind['head_tilt_deg']:5.1f} deg",
                    f"Head height     : {ind['head_height_ratio']:5.2f}",
                    f"Distance proxy  : {ind['distance_proxy']:5.2f}",
                ]
            else:
                lines = ["Move back so head and both shoulders are clearly visible"]

        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2)

        cv2.imshow("Posture indicators (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
