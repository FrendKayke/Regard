"""Calibration plein écran : on regarde des points, on apprend la correspondance regard -> écran."""
from __future__ import annotations

import time
import tkinter as tk
from typing import Optional

import numpy as np

from . import data_dir
from .face import FaceSource
from .model import GazeModel, clean_point_samples

BG, FG, ACCENT, GOOD, BAD = "#111418", "#e8eaed", "#ff4d4d", "#4caf50", "#ffb300"


def calibration_path():
    return data_dir() / "calibration.json"


def grid_points(rows: int, cols: int, margin: float = 0.06) -> list[tuple[float, float]]:
    xs = np.linspace(margin, 1 - margin, cols)
    ys = np.linspace(margin, 1 - margin, rows)
    pts = []
    for i, y in enumerate(ys):                       # parcours en serpentin
        row = xs if i % 2 == 0 else xs[::-1]
        pts += [(float(x), float(y)) for x in row]
    return pts


VALIDATION_POINTS = [(0.5, 0.5), (0.25, 0.3), (0.75, 0.3), (0.3, 0.75), (0.7, 0.72)]


class CalibrationUI:
    def __init__(self, source: FaceSource, rows=3, cols=5, settle=0.9, collect=1.0):
        self.source = source
        self.points = grid_points(rows, cols)
        self.settle, self.collect = settle, collect
        self.result: Optional[GazeModel] = None

        self.root = tk.Tk()
        self.root.title("Regard — calibration")
        self.root.configure(bg=BG, cursor="none")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.w, self.h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0, width=self.w, height=self.h)
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<Escape>", lambda e: self._quit())
        self.px_per_cm = float(self.root.winfo_fpixels("1c")) or 38.0
        self._phase = "intro"
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Return>", self._on_return)
        self.root.bind("r", lambda e: self._restart())
        self.root.bind("R", lambda e: self._restart())
        self._intro()

    # --- utilitaires d'affichage ---------------------------------------------------
    def _text(self, y, s, size=20, color=FG, tag="txt"):
        self.canvas.create_text(self.w / 2, y, text=s, fill=color, tag=tag,
                                font=("Segoe UI", size), justify="center")

    def _dot(self, x, y, r, color, tag="dot", fill=True):
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline=color, width=3,
                                fill=color if fill else "", tag=tag)

    # --- écran d'accueil ------------------------------------------------------------
    def _intro(self):
        self._phase = "intro"
        self.canvas.delete("all")
        self._text(self.h * 0.30, "Calibration du regard", 34)
        self._text(self.h * 0.42,
                   "Installez-vous comme d'habitude, la tête immobile.\n"
                   "Fixez chaque point rouge jusqu'à ce qu'il disparaisse.\n"
                   f"{len(self.points)} points, environ {int(len(self.points) * (self.settle + self.collect))} secondes.",
                   18)
        self._text(self.h * 0.62, "ESPACE pour commencer  ·  ÉCHAP pour annuler", 16, "#9aa0a6")
        self._update_face_status()

    def _update_face_status(self):
        if self._phase != "intro":
            return
        if self.source.error:
            self.canvas.delete("status")
            self._text(self.h * 0.53, f"Erreur webcam : {self.source.error}", 18, BAD, "status")
            return
        _, face, t = self.source.latest()
        ok = face is not None and time.monotonic() - t < 0.5
        self.canvas.delete("status")
        self._text(self.h * 0.53, "✓ visage détecté" if ok else "… recherche du visage",
                   18, GOOD if ok else BAD, "status")
        self.root.after(200, self._update_face_status)

    def _on_space(self, _):
        if self._phase == "intro" and not self.source.error:
            self._start_calibration()

    # --- collecte -------------------------------------------------------------------
    def _start_calibration(self):
        self._phase = "calib"
        self.samples_x, self.samples_y, self.groups = [], [], []
        self._run_points(self.points, self._on_calibration_done)

    def _run_points(self, points, done):
        self._queue = list(enumerate(points))
        self._collected: dict[int, list] = {}
        self._done_cb = done
        self._next_point()

    def _next_point(self):
        if not self._queue:
            self._done_cb()
            return
        self._idx, (fx, fy) = self._queue.pop(0)
        self._target = (fx * self.w, fy * self.h)
        self._t0 = time.monotonic()
        self._last_id = self.source.latest()[0]
        self._collected[self._idx] = []
        self._animate()

    def _animate(self):
        if self._phase not in ("calib", "valid"):
            return
        el = time.monotonic() - self._t0
        x, y = self._target
        self.canvas.delete("dot")
        if el < self.settle:                       # le cercle se resserre : on attend la saccade
            r = 8 + 28 * (1 - el / self.settle)
            self._dot(x, y, r, ACCENT, fill=False)
            self._dot(x, y, 5, ACCENT)
        elif el < self.settle + self.collect:      # collecte des images
            self._dot(x, y, 7, ACCENT)
            self._dot(x, y, 2, "#ffffff")
            fid, face, _ = self.source.latest()
            if fid != self._last_id:
                self._last_id = fid
                if face is not None and face.blink < 0.4:
                    self._collected[self._idx].append(face.features)
        else:
            self._next_point()
            return
        self.root.after(10, self._animate)

    # --- apprentissage ----------------------------------------------------------------
    def _on_calibration_done(self):
        xs, ys, gs = [], [], []
        for i, feats in self._collected.items():
            if len(feats) < 4:
                continue
            f = np.array(feats)
            f = f[clean_point_samples(f)]
            px, py = self.points[i]
            xs.append(f)
            ys.append(np.tile([px * self.w, py * self.h], (len(f), 1)))
            gs.append(np.full(len(f), i))
        self.canvas.delete("all")
        try:
            self.model = GazeModel.fit(np.vstack(xs), np.vstack(ys), np.concatenate(gs),
                                       (self.w, self.h))
        except (ValueError, np.linalg.LinAlgError) as e:
            self._phase = "error"
            self._text(self.h * 0.45, f"Calibration impossible : {e}", 20, BAD)
            self._text(self.h * 0.55, "R pour recommencer  ·  ÉCHAP pour quitter", 16, "#9aa0a6")
            return
        self._phase = "valid"
        self._text(self.h * 0.5, "Vérification : fixez encore 5 points", 22)
        self.root.after(1200, lambda: (self.canvas.delete("txt"),
                                       self._run_points(VALIDATION_POINTS, self._on_validation_done)))

    def _on_validation_done(self):
        self._phase = "result"
        self.canvas.delete("all")
        errors = []
        for i, (fx, fy) in enumerate(VALIDATION_POINTS):
            tx, ty = fx * self.w, fy * self.h
            self._dot(tx, ty, 8, ACCENT, tag="res")
            feats = self._collected.get(i, [])
            if len(feats) < 3:
                continue
            f = np.array(feats)
            pred = np.median(self.model.predict(f[clean_point_samples(f)]), axis=0)
            errors.append(float(np.hypot(pred[0] - tx, pred[1] - ty)))
            self.canvas.create_line(tx, ty, pred[0], pred[1], fill="#9aa0a6", dash=(4, 3))
            self._dot(pred[0], pred[1], 6, GOOD, tag="res")
        if errors:
            err = float(np.mean(errors))
            self.model.validation_error = err
            cm = err / self.px_per_cm
            quality = ("excellente" if cm < 2 else "bonne" if cm < 3.5
                       else "moyenne" if cm < 5.5 else "faible — recommencez")
            msg = f"Erreur moyenne : {err:.0f} px (≈ {cm:.1f} cm) — précision {quality}"
        else:
            msg = "Validation impossible (visage perdu)"
        self._text(self.h * 0.08, msg, 22)
        self._text(self.h * 0.93, "● rouge = cible   ● vert = regard estimé      "
                   "ENTRÉE pour enregistrer  ·  R pour recommencer  ·  ÉCHAP pour annuler",
                   15, "#9aa0a6")

    def _on_return(self, _):
        if self._phase == "result":
            self.model.save(calibration_path())
            self.result = self.model
            self._quit()

    def _restart(self):
        if self._phase in ("result", "error"):
            self._intro()

    def _quit(self):
        self._phase = "closed"
        self.root.destroy()

    def run(self) -> Optional[GazeModel]:
        self.root.focus_force()
        self.root.mainloop()
        return self.result


def calibrate(source: FaceSource, rows=3, cols=5) -> Optional[GazeModel]:
    return CalibrationUI(source, rows, cols).run()
