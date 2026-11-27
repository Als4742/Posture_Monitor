"""
Posture Monitor - Tkinter prototype.

Follows the design document: the eight screens, the state transition network
(Privacy note, No camera, Ready, Calibrating, Monitoring, Paused) and the style guide.

Privacy: video frames are processed in memory on this device. Nothing is saved or sent.
Pause releases the camera completely.

Keys: P pauses or resumes, Enter presses the focused button, Tab moves between controls.
Needs posture_metrics.py and posture_logic.py in the same folder.

Run:  python posture_app.py
"""
import time
import tkinter as tk

import cv2
import mediapipe as mp

from posture_metrics import compute_indicators
from posture_logic import (MESSAGES, SENSITIVITY_NAME, Smoother, default_baseline,
                           deviations, make_baseline, problems)

# ---- Settings ---------------------------------------------------------------
CALIBRATION_SECONDS = 5
TIMING = {
    "DEMO": {"bad": 3, "cooldown": 15},      # quick, for demonstrations
    "REAL": {"bad": 30, "cooldown": 900},    # survey: most want 15 to 30 minutes between reminders
}
NUDGE_SHOWN_SECONDS = 6
LOST_SECONDS = 1.0            # person must be missing this long before "Cannot see you"
SMOOTHING_FRAMES = 5
MODEL_COMPLEXITY = 1          # 0 fast, 1 default, 2 most accurate
PREVIEW_TITLE = "Camera preview (processed on this device)"
# -----------------------------------------------------------------------------

# Palette and type from the style guide
WHITE, SURFACE, DIVIDER = "#FFFFFF", "#F3F4F6", "#E5E7EB"
INK, QUIET, NEUTRAL = "#1F2937", "#4B5563", "#6B7280"
ACCENT, ACCENT_TINT, ACCENT_HOVER, ACCENT_LABEL = "#1D4ED8", "#DBEAFE", "#BFDBFE", "#1E3A8A"
GOOD, GOOD_TINT = "#15803D", "#DCFCE7"
WARN, WARN_TINT = "#B45309", "#FEF3C7"
FONT = "Segoe UI"
PRIVACY_LINE = "Processed on this device. No video is saved."

# App states
PRIVACY, NO_CAMERA, READY, CALIBRATING, MONITORING, PAUSED = (
    "PRIVACY", "NO_CAMERA", "READY", "CALIBRATING", "MONITORING", "PAUSED")
# Monitoring sub-states
GOOD_S, DRIFTING, NUDGE, CANNOT_SEE = "GOOD", "DRIFTING", "NUDGE", "CANNOT_SEE"


class CameraLost(Exception):
    pass


class PostureApp:
    def __init__(self, root, camera_index=0, clock=time.time):
        self.root = root
        self.clock = clock
        self.camera_index = camera_index

        self.cap = None
        self.pose = None
        self.preview_open = False

        self.state = PRIVACY
        self.sub = GOOD_S
        self.message = ""
        self.privacy_done = False
        self.using_defaults = False
        self.baseline = None
        self.samples = []
        self.calib_start = 0.0
        self.smoother = Smoother(SMOOTHING_FRAMES)
        self.bad_since = None
        self.lost_since = None
        self.last_nudge = -1e9
        self.nudge_text = ""
        self.nudge_until = 0.0
        self.rendered_key = None
        self.prog_canvas = self.prog_fill = self.prog_label = None
        self._render_pending = False

        root.title("Posture Monitor")
        root.configure(bg=WHITE)
        root.geometry("460x420")
        root.resizable(False, False)

        self.level = tk.IntVar(value=2)
        self.mode = tk.StringVar(value="DEMO")
        self.preview = tk.BooleanVar(value=False)
        for var in (self.level, self.mode):
            var.trace_add("write", lambda *a: self.request_render())

        # header: app name and camera status
        header = tk.Frame(root, bg=SURFACE, height=38)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="Posture Monitor", bg=SURFACE, fg=INK,
                 font=(FONT, 10, "bold")).pack(side="left", padx=16)
        self.chip_text = tk.Label(header, text="", bg=SURFACE, fg=QUIET, font=(FONT, 9))
        self.chip_text.pack(side="right", padx=(4, 16))
        self.chip_dot = tk.Canvas(header, width=12, height=12, bg=SURFACE, highlightthickness=0)
        self.chip_dot.pack(side="right")
        self.chip_oval = self.chip_dot.create_oval(2, 2, 10, 10, fill=NEUTRAL, outline=NEUTRAL)

        # footer: privacy line, always visible
        tk.Frame(root, bg=DIVIDER, height=1).pack(side="bottom", fill="x")
        footer = tk.Frame(root, bg=WHITE, height=34)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)
        tk.Label(footer, text="\u25A2  " + PRIVACY_LINE, bg=WHITE, fg=QUIET,
                 font=(FONT, 8)).pack(side="left", padx=16)

        self.content = tk.Frame(root, bg=WHITE)
        self.content.pack(fill="both", expand=True, padx=20, pady=14)

        root.bind("<Key-p>", self.on_key_p)
        root.bind("<Key-P>", self.on_key_p)
        root.bind("<Return>", self.on_return)

    # ------------------------------------------------------------------ camera
    def open_capture(self):
        return cv2.VideoCapture(self.camera_index)

    def camera_available(self):
        cap = self.open_capture()
        ok = cap.isOpened()
        cap.release()
        return ok

    def ensure_camera(self):
        if self.cap is None:
            cap = self.open_capture()
            if not cap.isOpened():
                cap.release()
                return False
            self.cap = cap
        return True

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.close_preview()

    def get_pose(self):
        if self.pose is None:
            self.pose = mp.solutions.pose.Pose(
                model_complexity=MODEL_COMPLEXITY, smooth_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return self.pose

    def grab(self):
        """Read one frame. Returns (indicators or None, frame for the optional preview)."""
        ok, frame = self.cap.read()
        if not ok:
            raise CameraLost()
        frame = cv2.flip(frame, 1)
        results = self.get_pose().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ind = None
        if results.pose_landmarks:
            ind = compute_indicators(results.pose_landmarks.landmark, mp.solutions.pose.PoseLandmark)
            if self.preview.get():
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return ind, frame

    def show_preview(self, frame):
        if self.preview.get() and frame is not None:
            cv2.imshow(PREVIEW_TITLE, frame)
            cv2.waitKey(1)
            self.preview_open = True
        elif self.preview_open:
            self.close_preview()

    def close_preview(self):
        if self.preview_open:
            try:
                cv2.destroyWindow(PREVIEW_TITLE)
            except cv2.error:
                pass
            self.preview_open = False

    # ----------------------------------------------------------------- lifecycle
    def start(self):
        self.state = PRIVACY if self.camera_available() else NO_CAMERA
        self.render()
        self.tick()

    def close(self):
        self.release_camera()
        if self.pose is not None:
            self.pose.close()
        self.root.destroy()

    def tick(self):
        self.step()
        self.root.after(15, self.tick)

    def step(self):
        """One pass of the main loop: read a frame if needed, update the state, redraw if changed."""
        now = self.clock()
        if self.state in (CALIBRATING, MONITORING) and self.cap is not None:
            try:
                ind, frame = self.grab()
            except CameraLost:
                self.camera_lost()
                return
            self.show_preview(frame)
            if self.state == CALIBRATING:
                self.step_calibration(ind, now)
            else:
                self.step_monitoring(ind, now)
        if self.state == MONITORING and self.sub == NUDGE and now >= self.nudge_until:
            self.sub = DRIFTING if self.bad_since is not None else GOOD_S
        if self.state == CALIBRATING:
            self.update_progress(now)
        key = (self.state, self.sub, self.message, self.nudge_text if self.sub == NUDGE else "")
        if key != self.rendered_key:
            self.render()

    def step_calibration(self, ind, now):
        if ind:
            self.samples.append(ind)
        if CALIBRATION_SECONDS - (now - self.calib_start) <= 0:
            try:
                self.baseline = make_baseline(self.samples)
            except ValueError as err:
                self.message = str(err)
                self.state = READY
                return
            self.using_defaults = False
            self.start_monitoring()

    def start_monitoring(self):
        self.state, self.sub, self.message = MONITORING, GOOD_S, ""
        self.smoother.reset()
        self.bad_since = self.lost_since = None

    def step_monitoring(self, ind, now):
        if ind is None:
            self.bad_since = None
            if self.lost_since is None:
                self.lost_since = now
            if now - self.lost_since >= LOST_SECONDS:
                self.sub = CANNOT_SEE
            return
        self.lost_since = None
        dev = deviations(self.smoother.add(ind), self.baseline)
        bad = problems(dev, self.level.get())
        if bad:
            if self.bad_since is None:
                self.bad_since = now
            if self.sub != NUDGE:
                self.sub = DRIFTING
            timing = TIMING[self.mode.get()]
            if now - self.bad_since >= timing["bad"] and now - self.last_nudge >= timing["cooldown"]:
                self.nudge_text = MESSAGES[bad[0]]
                self.nudge_until = now + NUDGE_SHOWN_SECONDS
                self.last_nudge = now
                self.sub = NUDGE
        else:
            self.bad_since = None
            self.sub = GOOD_S

    def camera_lost(self):
        self.release_camera()
        self.message = "The camera stopped sending images."
        self.state = NO_CAMERA
        self.render()

    # ------------------------------------------------------------------- actions
    def on_continue(self):
        self.privacy_done = True
        if self.ensure_camera():
            self.state, self.message = READY, ""
        else:
            self.message, self.state = "", NO_CAMERA
        self.render()

    def on_retry(self):
        if self.camera_available():
            self.message = ""
            self.state = READY if self.privacy_done else PRIVACY
            if self.state == READY:
                self.ensure_camera()
        self.render()

    def on_calibrate(self):
        if not self.ensure_camera():
            self.state, self.message = NO_CAMERA, ""
        else:
            self.samples, self.calib_start, self.message = [], self.clock(), ""
            self.state = CALIBRATING
        self.render()

    def on_skip(self):
        if not self.ensure_camera():
            self.state, self.message = NO_CAMERA, ""
        else:
            self.baseline = default_baseline()
            self.using_defaults = True
            self.start_monitoring()
        self.render()

    def on_cancel(self):
        self.state, self.message = READY, ""
        self.render()

    def on_pause(self):
        if self.state == MONITORING:
            self.release_camera()       # the camera really stops, as the screen says
            self.state = PAUSED
            self.render()

    def on_resume(self):
        if self.state == PAUSED:
            if self.ensure_camera():
                self.start_monitoring()
            else:
                self.state, self.message = NO_CAMERA, ""
            self.render()

    def on_key_p(self, event=None):
        if self.state == MONITORING:
            self.on_pause()
        elif self.state == PAUSED:
            self.on_resume()

    def on_return(self, event=None):
        widget = self.root.focus_get()
        if isinstance(widget, tk.Button):
            widget.invoke()

    def request_render(self):
        if not self._render_pending:
            self._render_pending = True
            self.root.after_idle(self._do_render)

    def _do_render(self):
        self._render_pending = False
        self.render()

    # ------------------------------------------------------------------ drawing
    def set_chip(self, text, colour):
        self.chip_text.config(text=text)
        self.chip_dot.itemconfig(self.chip_oval, fill=colour, outline=colour)

    def label(self, parent, text, size=10, bold=False, fg=INK, bg=WHITE, **kw):
        return tk.Label(parent, text=text, bg=bg, fg=fg, justify="left", anchor="w",
                        font=(FONT, size, "bold" if bold else "normal"), **kw)

    def dot(self, parent, colour, bg):
        c = tk.Canvas(parent, width=12, height=12, bg=bg, highlightthickness=0)
        c.create_oval(2, 2, 10, 10, fill=colour, outline=colour)
        return c

    def heading(self, text):
        self.label(self.content, text, 12, True).pack(anchor="w", pady=(0, 10))

    def status_card(self, text, colour, tint):
        card = tk.Frame(self.content, bg=tint, highlightbackground=colour, highlightthickness=1)
        card.pack(fill="x", pady=(0, 10))
        self.dot(card, colour, tint).pack(side="left", padx=(12, 6), pady=9)
        self.label(card, text, 11, True, bg=tint).pack(side="left", pady=8)

    def info_card(self, lines, dot_colour=None, fill=SURFACE, border=DIVIDER):
        card = tk.Frame(self.content, bg=fill, highlightbackground=border, highlightthickness=1)
        card.pack(fill="x", pady=(0, 10))
        for line in lines:
            row = tk.Frame(card, bg=fill)
            row.pack(fill="x", padx=12, pady=3)
            if dot_colour:
                self.dot(row, dot_colour, fill).pack(side="left", padx=(0, 8))
            self.label(row, line, 9, bg=fill).pack(side="left")

    def button(self, parent, text, command, main=False):
        border = ACCENT if main else NEUTRAL
        bg = ACCENT_TINT if main else WHITE
        hover = ACCENT_HOVER if main else SURFACE
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=ACCENT_LABEL if main else INK,
                      activebackground=hover, activeforeground=ACCENT_LABEL if main else INK,
                      relief="flat", bd=0, padx=14, pady=5, cursor="hand2", takefocus=True,
                      font=(FONT, 9, "bold" if main else "normal"),
                      highlightthickness=2 if main else 1, highlightbackground=border,
                      highlightcolor=ACCENT)
        b.bind("<Enter>", lambda e: b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        b.bind("<FocusIn>", lambda e: b.config(highlightthickness=3))
        b.bind("<FocusOut>", lambda e: b.config(highlightthickness=2 if main else 1))
        return b

    def button_row(self, buttons, right=False):
        row = tk.Frame(self.content, bg=WHITE)
        row.pack(side="bottom", fill="x", pady=(8, 0))
        for text, command, main in buttons:
            b = self.button(row, text, command, main)
            b.pack(side="right" if right else "left", padx=(0 if right else 0, 10))
            if main:
                b.focus_set()
        return row

    def segmented(self, caption, var, options):
        self.label(self.content, caption, 9, fg=QUIET).pack(anchor="w")
        row = tk.Frame(self.content, bg=WHITE, highlightbackground=NEUTRAL, highlightthickness=1)
        row.pack(anchor="w", pady=(2, 8))
        for text, value in options:
            tk.Radiobutton(row, text=text, variable=var, value=value, indicatoron=False,
                           bg=WHITE, fg=INK, selectcolor=ACCENT_TINT, activebackground=SURFACE,
                           relief="flat", bd=0, width=8, pady=3, font=(FONT, 9),
                           highlightthickness=0, takefocus=True).pack(side="left")

    def update_progress(self, now):
        if self.prog_canvas is None:
            return
        left = max(CALIBRATION_SECONDS - (now - self.calib_start), 0)
        frac = 1 - left / CALIBRATION_SECONDS
        self.prog_canvas.coords(self.prog_fill, 0, 0, 260 * frac, 12)
        self.prog_label.config(text=f"{left:.0f} s left")

    def render(self):
        for child in self.content.winfo_children():
            child.destroy()
        self.prog_canvas = self.prog_fill = self.prog_label = None
        self.rendered_key = (self.state, self.sub, self.message,
                             self.nudge_text if self.sub == NUDGE else "")

        if self.state == PRIVACY:
            self.set_chip("Camera: off", NEUTRAL)
            self.heading("Before you start")
            self.info_card(["The camera is used to watch your posture.",
                            "Video stays on this device and is never saved.",
                            "Only head and shoulder positions are analysed."], dot_colour=GOOD)
            self.button_row([("Continue", self.on_continue, True)], right=True)

        elif self.state == NO_CAMERA:
            self.set_chip("No camera", WARN)
            self.heading("No webcam found")
            text = self.message or "Check that a webcam is connected and not\nin use by another app, then try again."
            self.info_card(text.split("\n"), dot_colour=WARN, fill=WARN_TINT, border=WARN)
            self.button_row([("Retry", self.on_retry, True)], right=True)

        elif self.state == READY:
            self.set_chip("Camera: on", GOOD)
            self.heading("Ready to calibrate?")
            if self.message:
                self.info_card([self.message], dot_colour=WARN, fill=WARN_TINT, border=WARN)
            self.info_card(["Sit upright and look at the screen.",
                            "Calibration takes about 5 seconds."])
            self.button_row([("Calibrate", self.on_calibrate, True),
                             ("Skip, use defaults", self.on_skip, False)])

        elif self.state == CALIBRATING:
            self.set_chip("Camera: on", GOOD)
            self.heading("Calibrating")
            self.label(self.content, "Sit in your best posture and hold still.", 10, True).pack(
                anchor="w", pady=(4, 10))
            row = tk.Frame(self.content, bg=WHITE)
            row.pack(anchor="w")
            self.prog_canvas = tk.Canvas(row, width=260, height=12, bg=DIVIDER, highlightthickness=0)
            self.prog_canvas.pack(side="left")
            self.prog_fill = self.prog_canvas.create_rectangle(0, 0, 0, 12, fill=ACCENT, outline=ACCENT)
            self.prog_label = self.label(row, "5 s left", 9, fg=QUIET)
            self.prog_label.pack(side="left", padx=10)
            self.button_row([("Cancel", self.on_cancel, False)], right=True)

        elif self.state == PAUSED:
            self.set_chip("Camera: paused", NEUTRAL)
            self.status_card("Paused", NEUTRAL, SURFACE)
            self.info_card(["Analysis is stopped. Nothing is being", "recorded or analysed."])
            self.button_row([("Resume (P)", self.on_resume, True)], right=True)

        elif self.state == MONITORING:
            self.set_chip("Camera: on", GOOD)
            summary = (f"Sensitivity: {SENSITIVITY_NAME[self.level.get()]} \u00b7 "
                       f"Mode: {self.mode.get().title()}")
            if self.sub == CANNOT_SEE:
                self.status_card("Cannot see you clearly", WARN, WARN_TINT)
                self.label(self.content, "Sit so your head and both shoulders are\nvisible to the camera.",
                           10).pack(anchor="w", pady=(0, 10))
                self.info_card(["Tip: turn on a lamp or move back slightly."])
                self.button_row([("Pause (P)", self.on_pause, False)], right=True)
            elif self.sub == NUDGE:
                self.status_card("Drifting from your baseline", WARN, WARN_TINT)
                banner = tk.Frame(self.content, bg=ACCENT_TINT, highlightbackground=ACCENT,
                                  highlightthickness=2)
                banner.pack(fill="x", pady=(0, 10))
                self.label(banner, self.nudge_text, 10, True, bg=ACCENT_TINT).pack(
                    anchor="w", padx=12, pady=(8, 0))
                self.label(banner, "Shown in this window only: no pop-up, no sound.", 8,
                           fg=QUIET, bg=ACCENT_TINT).pack(anchor="w", padx=12, pady=(0, 8))
                self.label(self.content, summary, 9, fg=QUIET).pack(anchor="w")
                self.button_row([("Pause (P)", self.on_pause, False),
                                 ("Recalibrate", self.on_calibrate, False)])
            else:
                if self.sub == GOOD_S:
                    self.status_card("Good posture", GOOD, GOOD_TINT)
                else:
                    self.status_card("Drifting from your baseline", WARN, WARN_TINT)
                self.segmented("Sensitivity", self.level,
                               [(SENSITIVITY_NAME[i], i) for i in (1, 2, 3)])
                self.segmented("Mode", self.mode, [("Demo", "DEMO"), ("Real", "REAL")])
                tk.Checkbutton(self.content, text="Show camera preview", variable=self.preview,
                               bg=WHITE, fg=INK, activebackground=WHITE, selectcolor=ACCENT_TINT,
                               font=(FONT, 9), highlightthickness=0, takefocus=True).pack(anchor="w")
                if self.using_defaults:
                    self.label(self.content, "Default limits: only shoulder and head tilt are checked. "
                               "Recalibrate for the full checks.", 8, fg=QUIET, wraplength=400).pack(
                        anchor="w", pady=(4, 0))
                self.button_row([("Pause (P)", self.on_pause, False),
                                 ("Recalibrate", self.on_calibrate, False)])


def main():
    root = tk.Tk()
    app = PostureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    app.start()
    root.mainloop()


if __name__ == "__main__":
    main()
