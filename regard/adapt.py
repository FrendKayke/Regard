"""Correction automatique par les clics.

Quand on clique, on regarde presque toujours le curseur. Chaque clic donne donc un point
de calibration gratuit : on connaît la vraie position regardée (le curseur) et ce que voyait
la webcam juste avant (~0,1–0,4 s).

Deux mécanismes complémentaires :
1. **Correction locale immédiate** : décalage moyen des clics récents et proches du point
   regardé (pondéré par la distance et l'ancienneté). Ça rattrape vite une dérive de posture.
2. **Ré-apprentissage** : toutes les quelques clics, le modèle est réentraîné en arrière-plan
   avec la calibration + les clics. La compensation des mouvements de tête s'améliore au fil
   de l'eau, puisque les clics sont faits dans toutes sortes de postures.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .model import GazeModel

CLICK_GROUP_OFFSET = 1000


@dataclass
class ClickSample:
    t: float
    features: np.ndarray        # caractéristiques moyennes juste avant le clic
    target: np.ndarray          # position du curseur (px)


class ClickCorrector:
    def __init__(self, model: GazeModel, screen, on_model: Callable[[GazeModel], None],
                 is_suppressed: Callable[[], bool] = lambda: False,
                 window=(0.40, 0.06), max_clicks=300, refit_every=6,
                 sigma_frac=0.3, tau=900.0, prior=1.0, click_weight=12.0):
        self.model = model
        self.w, self.h = screen
        self.on_model = on_model
        self.is_suppressed = is_suppressed
        self.window = window                  # fenêtre avant le clic : (début, fin) en s
        self.max_clicks, self.refit_every = max_clicks, refit_every
        self.sigma = sigma_frac * self.w
        self.tau, self.prior, self.click_weight = tau, prior, click_weight
        self.frames: deque = deque(maxlen=90)  # (t, features)
        self.clicks: list[ClickSample] = []
        self._residuals = np.zeros((0, 2))
        self._lock = threading.Lock()
        self._since_refit = 0
        self._refitting = False
        self.err_raw: deque = deque(maxlen=30)
        self.err_corr: deque = deque(maxlen=30)
        self.rejected = 0
        self.refits = 0
        self._listener = None

    # --- alimentation ---------------------------------------------------------------
    def add_frame(self, t: float, features: np.ndarray):
        self.frames.append((t, features))

    def start_listener(self):
        try:
            from pynput import mouse
        except Exception as e:  # noqa: BLE001
            print(f"[regard] correction par clics indisponible : {e!r}")
            return

        def on_click(x, y, button, pressed):
            if pressed:
                self.on_click(x, y, time.monotonic())

        self._listener = mouse.Listener(on_click=on_click)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()

    # --- clics ------------------------------------------------------------------------
    def on_click(self, x: float, y: float, t: float) -> Optional[str]:
        """Renvoie la raison d'un rejet, ou None si le clic a été utilisé."""
        if self.is_suppressed():
            return "clic généré par Regard / curseur piloté par le regard"
        if not (0 <= x < self.w and 0 <= y < self.h):
            return "hors de l'écran principal"
        t0, t1 = t - self.window[0], t - self.window[1]
        feats = np.array([f for ft, f in list(self.frames) if t0 <= ft <= t1])
        if len(feats) < 3:
            self.rejected += 1
            return "pas assez d'images avant le clic (visage perdu ou clignement)"
        # Le regard doit être stable (fixation), pas en plein mouvement.
        if np.max(feats[:, :2].std(axis=0)) > 0.012:
            self.rejected += 1
            return "regard en mouvement au moment du clic"
        f = np.median(feats, axis=0)
        target = np.array([x, y], dtype=float)
        raw = self.model.predict(f)
        corr = raw + self.offset(raw)
        # Si l'estimation est très loin du curseur, on regardait sans doute ailleurs.
        if np.linalg.norm(corr - target) > 0.3 * max(self.w, self.h):
            self.rejected += 1
            return "regard trop loin du curseur (vous regardiez ailleurs ?)"
        self.err_raw.append(float(np.linalg.norm(raw - target)))
        self.err_corr.append(float(np.linalg.norm(corr - target)))
        with self._lock:
            self.clicks.append(ClickSample(t, f, target))
            self.clicks = self.clicks[-self.max_clicks:]
            self._recompute_residuals()
            self._since_refit += 1
            need_refit = self._since_refit >= self.refit_every and not self._refitting
        if need_refit:
            self._start_refit()
        return None

    # --- correction locale --------------------------------------------------------------
    def _recompute_residuals(self):
        if not self.clicks:
            self._residuals = np.zeros((0, 2))
            return
        f = np.array([c.features for c in self.clicks])
        tg = np.array([c.target for c in self.clicks])
        self._residuals = tg - np.atleast_2d(self.model.predict(f))

    def offset(self, raw: np.ndarray) -> np.ndarray:
        with self._lock:
            if not self.clicks:
                return np.zeros(2)
            tg = np.array([c.target for c in self.clicks])
            ages = time.monotonic() - np.array([c.t for c in self.clicks])
            res = self._residuals
        d2 = np.sum((tg - raw) ** 2, axis=1)
        w = np.exp(-d2 / (2 * self.sigma ** 2)) * np.exp(-np.maximum(ages, 0) / self.tau)
        return (w[:, None] * res).sum(axis=0) / (w.sum() + self.prior)

    def correct(self, raw: np.ndarray) -> np.ndarray:
        return raw + self.offset(raw)

    # --- ré-apprentissage -------------------------------------------------------------------
    def _start_refit(self):
        self._refitting = True
        threading.Thread(target=self._refit, daemon=True).start()

    def _refit(self):
        try:
            with self._lock:
                clicks = list(self.clicks)
                self._since_refit = 0
            base = self.model.samples
            if base is None or len(clicks) < 3:
                return
            bx, by, bg = base
            cx = np.array([c.features for c in clicks])
            cy = np.array([c.target for c in clicks])
            cg = CLICK_GROUP_OFFSET + np.arange(len(clicks))
            # Les clics récents comptent plus ; chacun pèse comme une partie d'un point de grille.
            ages = time.monotonic() - np.array([c.t for c in clicks])
            cw = self.click_weight * np.exp(-ages / (4 * self.tau))
            x = np.vstack([bx, cx])
            y = np.vstack([by, cy])
            g = np.concatenate([bg, cg])
            w = np.concatenate([np.ones(len(bx)), cw])
            new = GazeModel.fit(x, y, g, (self.w, self.h), weights=w, keep_samples=False)
            new.samples = base                       # on garde la calibration d'origine
            new.validation_error = self.model.validation_error
            with self._lock:
                self.model = new
                self._recompute_residuals()
            self.refits += 1
            self.on_model(new)
        except Exception as e:  # noqa: BLE001
            print(f"[regard] ré-apprentissage impossible : {e!r}")
        finally:
            self._refitting = False

    # --- état ---------------------------------------------------------------------------------
    def summary(self) -> str:
        n = len(self.clicks)
        if not n:
            return "correction auto : en attente de clics"
        raw, corr = np.mean(self.err_raw), np.mean(self.err_corr)
        return f"correction auto : {n} clics · erreur {raw:.0f} → {corr:.0f} px"
