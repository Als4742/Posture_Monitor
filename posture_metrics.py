"""
Posture indicators computed from MediaPipe Pose landmarks.

Landmarks use normalised image coordinates: x and y are in 0..1, y grows DOWNWARD.
Because a laptop webcam looks at you from the front, we measure posture through
front-view proxies and compare them with each user's own calibrated baseline.

Indicators
- shoulder_tilt_deg : angle of the shoulder line vs horizontal (0 = level shoulders)
- head_tilt_deg     : angle of the ear line vs horizontal (0 = head level)
- head_height_ratio : (shoulder_mid_y - nose_y) / shoulder_width.
                      Drops when the head sinks toward the shoulders (slouching / head drop).
- distance_proxy    : shoulder width as a fraction of frame width.
                      Rises when you lean toward the screen (closer to the camera).
- head_pitch        : (nose_y - ear_mid_y) / shoulder_width.
                      Rises when you bend your head forward or down. It uses only the face
                      landmarks, so it does not depend on where the shoulder points land.
                      None when the ears are not clearly visible.
"""
import math

MIN_VISIBILITY = 0.5


def _angle_from_horizontal(p1, p2):
    """Absolute angle in degrees (0..90) of the line p1->p2 relative to horizontal."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = abs(math.degrees(math.atan2(dy, dx)))
    return min(angle, 180 - angle)


def compute_indicators(lm, PL):
    """
    lm : list of landmarks (results.pose_landmarks.landmark)
    PL : mediapipe PoseLandmark enum
    Returns a dict of indicators, or None if key landmarks are not clearly visible.
    """
    ls = lm[PL.LEFT_SHOULDER]
    rs = lm[PL.RIGHT_SHOULDER]
    le = lm[PL.LEFT_EAR]
    re = lm[PL.RIGHT_EAR]
    nose = lm[PL.NOSE]

    for p in (ls, rs, nose):
        if p.visibility < MIN_VISIBILITY:
            return None

    shoulder_width = math.hypot(rs.x - ls.x, rs.y - ls.y)
    if shoulder_width < 1e-6:
        return None

    shoulder_mid_y = (ls.y + rs.y) / 2

    # Ears can be hidden when the head turns; tilt falls back to 0 and pitch to None.
    if le.visibility >= MIN_VISIBILITY and re.visibility >= MIN_VISIBILITY:
        head_tilt = _angle_from_horizontal(le, re)
        head_pitch = (nose.y - (le.y + re.y) / 2) / shoulder_width
    else:
        head_tilt = 0.0
        head_pitch = None

    return {
        "shoulder_tilt_deg": _angle_from_horizontal(ls, rs),
        "head_tilt_deg": head_tilt,
        "head_height_ratio": (shoulder_mid_y - nose.y) / shoulder_width,
        "distance_proxy": shoulder_width,
        "head_pitch": head_pitch,
    }
