"""Suivi en direct : regard -> application regardée, enregistrement, gestes et actions."""
from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from typing import Optional

import numpy as np

from .face import FaceFrame, FaceSource
from .gestures import ActionRunner, GestureDetector, load_config
from .model import GazeModel, OneEuroFilter
from .store import ABSENT, OFFSCREEN, Recorder
from .windows import WindowInfo, make_locator


class GazeTracker:
    """Appelé par FaceSource pour chaque image (dans le fil de la webcam)."""

    def __init__(self, model: GazeModel, screen, use_actions=False, record=True):
        self.model = model
        self.w, self.h = screen
        self.filter = OneEuroFilter()
        self.locator = None                     # créé dans le fil de la webcam (X11)
        self.recorder = Recorder(screen) if record else None
        gestures, actions, options = load_config()
        self.detector = GestureDetector(gestures)
        self.actions = ActionRunner(actions, options)
        self.actions.enabled = use_actions
        self.paused = False
        self.lock = threading.Lock()
        self.gaze: Optional[np.ndarray] = None
        self.window: Optional[WindowInfo] = None
        self.app = "—"
        self.face_ok = False
        self.last_t: Optional[float] = None
        self.last_events: list[tuple[float, str]] = []
        self._last_lookup = 0.0

    def on_frame(self, face: Optional[FaceFrame], t: float):
        if self.locator is None:
            self.locator = make_locator()
        dt = 0.0 if self.last_t is None else min(t - self.last_t, 0.25)
        self.last_t = t
        self.face_ok = face is not None

        # Gestes (fonctionnent même en pause, pour pouvoir réactiver)
        for g in self.detector.update(face.blend if face else None, t):
            result = self.actions.handle(g, tuple(self.gaze) if self.gaze is not None else None)
            label = g + (f" → {result}" if result else "")
            self.last_events = (self.last_events + [(time.time(), label)])[-5:]
            print(f"[geste] {label}")
            if self.recorder:
                self.recorder.event("gesture", label)

        if self.paused:
            return
        if face is None:
            with self.lock:
                self.app = ABSENT
            if self.recorder:
                self.recorder.add(dt, None, None, ABSENT)
            return

        # Pendant un clignement l'iris n'est pas visible : on garde la dernière position.
        if face.blink < 0.45 or self.gaze is None:
            raw = self.model.predict(face.features)
            gaze = self.filter(raw, t)
        else:
            gaze = self.gaze
        x, y = float(gaze[0]), float(gaze[1])
        margin = 0.08 * max(self.w, self.h)
        offscreen = x < -margin or y < -margin or x > self.w + margin or y > self.h + margin

        cx, cy = min(max(x, 0), self.w - 1), min(max(y, 0), self.h - 1)
        if offscreen:
            win, app = None, OFFSCREEN
        else:
            if t - self._last_lookup > 1 / 15 or self.window is None:
                self.window = self.locator.at(int(cx), int(cy))
                self._last_lookup = t
            win = self.window
            app = win.app if win else "Inconnu"

        with self.lock:
            self.gaze = np.array([cx, cy])
            self.app = app
        if not offscreen:
            self.actions.follow_gaze((cx, cy))
        if self.recorder:
            if win:
                self.recorder.add(dt, cx, cy, app, win.title, win.relative(cx, cy), win.rect[2:])
            else:
                self.recorder.add(dt, cx, cy, app)

    def close(self):
        if self.recorder:
            self.recorder.close()


class GazeDot:
    """Petit point rouge qui suit le regard, toujours au premier plan et transparent aux clics."""
    SIZE, KEY = 26, "#ff00fe"

    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        if sys.platform == "win32":
            bg = self.KEY
        elif sys.platform == "darwin":
            bg = "systemTransparent"
            self.top.attributes("-transparent", True)
            self.top.config(bg=bg)
        else:                                   # X11 : pas de couleur transparente
            bg = "#ff3b3b"
            self.SIZE = 14
        c = tk.Canvas(self.top, width=self.SIZE, height=self.SIZE, highlightthickness=0, bg=bg)
        c.pack()
        if self.SIZE > 14:
            c.create_oval(3, 3, self.SIZE - 3, self.SIZE - 3, outline="#ff3b3b", width=3)
            c.create_oval(10, 10, self.SIZE - 10, self.SIZE - 10, fill="#ff3b3b", outline="")
        if sys.platform == "win32":
            self.top.attributes("-transparentcolor", self.KEY)
            self.top.update_idletasks()
            self._click_through_win32()
        else:
            self.top.attributes("-alpha", 0.7)
        self.visible = True

    def _click_through_win32(self):
        import ctypes
        u = ctypes.windll.user32
        try:
            hwnd = int(self.top.wm_frame(), 16)
        except (ValueError, tk.TclError):
            hwnd = u.GetParent(self.top.winfo_id())
        style = u.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
        # WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        u.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20 | 0x80 | 0x08000000)

    def move(self, x, y):
        s = self.SIZE // 2
        self.top.geometry(f"+{int(x) - s}+{int(y) - s}")

    def show(self, on: bool):
        if on != self.visible:
            (self.top.deiconify if on else self.top.withdraw)()
            self.visible = on


class StatusWindow:
    """Petite fenêtre de contrôle : application regardée, état, options."""

    def __init__(self, source: FaceSource, tracker: GazeTracker, show_dot=True):
        self.source, self.tracker = source, tracker
        self.root = tk.Tk()
        self.root.title("Regard")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        pad = dict(padx=10, pady=3, sticky="w")
        self.app_var = tk.StringVar(value="—")
        self.title_var = tk.StringVar(value="")
        self.state_var = tk.StringVar(value="")
        self.events_var = tk.StringVar(value="")
        tk.Label(self.root, text="Vous regardez :", font=("Segoe UI", 9)).grid(row=0, column=0, **pad)
        tk.Label(self.root, textvariable=self.app_var, font=("Segoe UI", 14, "bold"),
                 width=28, anchor="w").grid(row=1, column=0, columnspan=3, **pad)
        tk.Label(self.root, textvariable=self.title_var, font=("Segoe UI", 9), fg="#666",
                 width=48, anchor="w").grid(row=2, column=0, columnspan=3, **pad)
        tk.Label(self.root, textvariable=self.state_var, font=("Segoe UI", 9),
                 anchor="w").grid(row=3, column=0, columnspan=3, **pad)
        self.dot_var = tk.BooleanVar(value=show_dot)
        self.act_var = tk.BooleanVar(value=tracker.actions.enabled)
        tk.Checkbutton(self.root, text="Afficher le point du regard", variable=self.dot_var
                       ).grid(row=4, column=0, columnspan=3, **pad)
        tk.Checkbutton(self.root, text="Actions par gestes du visage (expérimental)",
                       variable=self.act_var, command=self._toggle_actions
                       ).grid(row=5, column=0, columnspan=3, **pad)
        tk.Label(self.root, textvariable=self.events_var, font=("Segoe UI", 9), fg="#666",
                 justify="left", anchor="w").grid(row=6, column=0, columnspan=3, **pad)
        self.pause_btn = tk.Button(self.root, text="Pause", width=10, command=self._toggle_pause)
        self.pause_btn.grid(row=7, column=0, padx=10, pady=8, sticky="w")
        tk.Button(self.root, text="Quitter", width=10, command=self.quit
                  ).grid(row=7, column=2, padx=10, pady=8, sticky="e")
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"+{sw - self.root.winfo_width() - 30}+40")
        self.dot = GazeDot(self.root)
        tracker.actions.on_change = lambda: self.act_var.set(tracker.actions.enabled)
        self._tick()

    def _toggle_actions(self):
        self.tracker.actions.enabled = self.act_var.get()
        if not self.tracker.actions.enabled:
            self.tracker.actions.follow = False

    def _toggle_pause(self):
        self.tracker.paused = not self.tracker.paused
        self.pause_btn.config(text="Reprendre" if self.tracker.paused else "Pause")

    def _tick(self):
        tr = self.tracker
        with tr.lock:
            gaze, app = tr.gaze, tr.app
            win = tr.window
        self.app_var.set("(en pause)" if tr.paused else app)
        self.title_var.set((win.title[:70] if win and app not in ("—",) else "") if not tr.paused else "")
        face = "visage ✓" if tr.face_ok else "visage ✗"
        follow = " · curseur suit le regard" if tr.actions.follow else ""
        self.state_var.set(f"{self.source.fps:4.1f} img/s · {face}{follow}")
        recent = [lbl for ts, lbl in tr.last_events if time.time() - ts < 8]
        self.events_var.set("\n".join(recent[-3:]))
        show = self.dot_var.get() and gaze is not None and not tr.paused
        self.dot.show(show)
        if show:
            self.dot.move(*gaze)
        if self.source.error:
            self.state_var.set(f"Erreur webcam : {self.source.error}")
        self.root.after(33, self._tick)

    def quit(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()
