"""
Calibration and posture-judgement logic (no camera code here, so it is easy to test).

Idea: the user records a 'good posture' baseline once. After that we measure how far
the live indicators drift from THEIR baseline, instead of using one universal threshold.
"""
import statistics
from collections import deque

KEYS = ("shoulder_tilt_deg", "head_tilt_deg", "head_height_ratio", "distance_proxy", "head_pitch")

# Default drift allowed before a reading counts as 'poor posture' (at medium sensitivity).
# Order matters: when several problems occur at once, the first one is the one reported.
BASE_THRESHOLDS = {
    "head_bend": 0.06,      # head pitched forward/down: pitch value rose by this much
    "head_drop": 0.15,      # head height fell 15% below baseline
    "lean_in": 0.15,        # shoulders look 15% wider than baseline (closer to screen)
    "shoulder_tilt": 6.0,   # degrees more tilt than baseline
    "head_tilt": 10.0,      # degrees more head tilt than baseline
}

# Sensitivity 1 = low (tolerant), 2 = medium, 3 = high (strict). Multiplies the thresholds.
SENSITIVITY = {1: 1.5, 2: 1.0, 3: 0.7}
SENSITIVITY_NAME = {1: "Low", 2: "Medium", 3: "High"}

MESSAGES = {
    "head_bend": "Your head is bent forward - lift your chin",
    "head_drop": "Your head is sinking - sit tall",
    "lean_in": "You are leaning toward the screen - sit back a little",
    "shoulder_tilt": "Shoulders look uneven - relax and level them",
    "head_tilt": "Head is tilted - bring it back to centre",
}


def make_baseline(samples):
    """Median of each indicator over the calibration samples (robust to small jitters)."""
    if len(samples) < 20:
        raise ValueError("Not enough clear frames. Sit so head and shoulders are visible and retry.")
    base = {}
    for k in KEYS:
        values = [s[k] for s in samples if s.get(k) is not None]
        if len(values) >= 20:
            base[k] = statistics.median(values)
        elif k == "head_pitch":
            base[k] = None   # ears were rarely visible: head-bend check is skipped
        else:
            raise ValueError("Not enough clear frames. Sit so head and shoulders are visible and retry.")
    if base["head_height_ratio"] <= 0.05 or base["distance_proxy"] <= 0.01:
        raise ValueError("Baseline looks invalid. Check your position and retry.")
    return base


def default_baseline():
    """Used when the student skips calibration. Only the tilt checks can work without a personal
    baseline, so the other values are left empty and those checks are skipped."""
    return {"shoulder_tilt_deg": 0.0, "head_tilt_deg": 0.0, "head_height_ratio": None,
            "distance_proxy": None, "head_pitch": None}


def deviations(ind, base):
    """Positive numbers mean 'worse than baseline'. A check is left out if its data is missing."""
    dev = {
        "shoulder_tilt": ind["shoulder_tilt_deg"] - base["shoulder_tilt_deg"],
        "head_tilt": ind["head_tilt_deg"] - base["head_tilt_deg"],
    }
    if base.get("head_height_ratio") and ind.get("head_height_ratio") is not None:
        dev["head_drop"] = (base["head_height_ratio"] - ind["head_height_ratio"]) / base["head_height_ratio"]
    if base.get("distance_proxy") and ind.get("distance_proxy") is not None:
        dev["lean_in"] = (ind["distance_proxy"] - base["distance_proxy"]) / base["distance_proxy"]
    if ind.get("head_pitch") is not None and base.get("head_pitch") is not None:
        dev["head_bend"] = ind["head_pitch"] - base["head_pitch"]
    return dev


def limits(level=2):
    """Thresholds after applying the sensitivity level."""
    factor = SENSITIVITY[level]
    return {name: limit * factor for name, limit in BASE_THRESHOLDS.items()}


def problems(dev, level=2):
    """Names of the indicators that exceed their (sensitivity-scaled) threshold."""
    lim = limits(level)
    return [name for name in BASE_THRESHOLDS if name in dev and dev[name] > lim[name]]


class Smoother:
    """Moving average over the last N readings to reduce webcam jitter."""

    def __init__(self, size=5):
        self.buf = deque(maxlen=size)

    def add(self, ind):
        self.buf.append(ind)
        out = {}
        for k in KEYS:
            vals = [x[k] for x in self.buf if x.get(k) is not None]
            out[k] = sum(vals) / len(vals) if vals else None
        return out

    def reset(self):
        self.buf.clear()
