"""
Step 1 test: open the webcam, detect body landmarks with MediaPipe Pose,
and draw them live. Frames are processed in memory only and never saved.

Press 'q' in the video window to quit.
"""
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps using it and try again.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame received from the webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror view, feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark
            ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
            shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            h, w = frame.shape[:2]
            cv2.putText(frame, f"ear x={ear.x:.2f} y={ear.y:.2f} | shoulder x={shoulder.x:.2f} y={shoulder.y:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No person detected - sit so your head and shoulders are visible",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Posture monitor - webcam test (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
