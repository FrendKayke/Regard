"""Modèle regard -> écran : régression ridge polynomiale, apprise pendant la calibration."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .face import FEATURE_NAMES

# Écart-type minimal par caractéristique (unités brutes). Pendant la calibration la tête
# bouge peu : sans ce plancher, un petit mouvement ultérieur ferait exploser la prédiction.
STD_FLOOR = np.array([0.004, 0.004, 0.01, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5])
# Pénalité relative par terme du modèle (voir expand). Si la tête a peu bougé pendant la
# calibration, on régularise beaucoup plus ses termes pour ne pas apprendre du bruit ;
# si elle a bougé (phase « tête mobile », clics), on les laisse apprendre normalement.
BASE_PENALTY = np.array([0.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
HEAD_TERMS = slice(4, 10)          # yaw, pitch, roll, tx, ty, tz dans expand()
HEAD_STILL_PENALTY = 20.0
HEAD_MOVED_STD = np.array([4.0, 4.0, 4.0, 2.0, 2.0, 2.0])  # au-delà : la tête a « bougé »
ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def expand(z: np.ndarray) -> np.ndarray:
    """Termes du modèle : constante, linéaires, et quadratiques sur la position de l'iris."""
    z = np.atleast_2d(z)
    ex, ey = z[:, 0], z[:, 1]
    return np.column_stack([np.ones(len(z)), z, ex * ex, ey * ey, ex * ey])


def penalties(x: np.ndarray) -> np.ndarray:
    pen = BASE_PENALTY.copy()
    head_std = np.asarray(x, float)[:, 3:9].std(axis=0)
    pen[HEAD_TERMS] = np.where(head_std >= HEAD_MOVED_STD, 1.0, HEAD_STILL_PENALTY)
    return pen


def _ridge(a, y, alpha, pen, w=None) -> np.ndarray:
    if w is not None:                      # moindres carrés pondérés
        sw = np.sqrt(w)[:, None]
        a, y = a * sw, y * sw
    reg = alpha * np.diag(pen[: a.shape[1]])  # la constante n'est pas pénalisée
    return np.linalg.solve(a.T @ a + reg, a.T @ y)


def clean_point_samples(x: np.ndarray) -> np.ndarray:
    """Retire les images aberrantes (saccades, clignements) d'un point de calibration."""
    if len(x) < 5:
        return np.ones(len(x), dtype=bool)
    med = np.median(x[:, :2], axis=0)
    mad = np.median(np.abs(x[:, :2] - med), axis=0) + 1e-4
    return np.all(np.abs(x[:, :2] - med) <= 3.0 * 1.4826 * mad, axis=1)


class GazeModel:
    def __init__(self, mean, std, weights, alpha, screen, cv_error=None, created=None):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.alpha = float(alpha)
        self.screen = tuple(int(v) for v in screen)
        self.cv_error = cv_error
        self.created = created or time.strftime("%Y-%m-%d %H:%M:%S")
        self.validation_error: Optional[float] = None
        self.head_free = False
        # Échantillons de calibration conservés pour les ré-apprentissages (clics).
        self.samples: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    # --- apprentissage -------------------------------------------------------------
    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, groups: np.ndarray, screen,
            weights: Optional[np.ndarray] = None, keep_samples=True) -> "GazeModel":
        """`groups` : un identifiant par cible (point de calibration ou clic) ;
        `weights` : poids de chaque échantillon (défaut 1)."""
        x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
        w_all = None if weights is None else np.asarray(weights, float)
        if len(np.unique(groups)) < 5:
            raise ValueError("Pas assez de points de calibration exploitables (visage perdu ?).")
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), STD_FLOOR)
        a = expand(np.clip((x - mean) / std, -6, 6))
        pen = penalties(x)

        # Choix de la régularisation par validation croisée « un point laissé de côté ».
        best_alpha, best_err = ALPHAS[0], math.inf
        for alpha in ALPHAS:
            errs = []
            for g in np.unique(groups):
                test = groups == g
                w = _ridge(a[~test], y[~test], alpha, pen,
                           None if w_all is None else w_all[~test])
                pred = a[test] @ w
                errs.append(float(np.median(np.linalg.norm(pred - y[test], axis=1))))
            err = float(np.mean(errs))
            if err < best_err:
                best_alpha, best_err = alpha, err
        coefs = _ridge(a, y, best_alpha, pen, w_all)
        model = cls(mean, std, coefs, best_alpha, screen, cv_error=best_err)
        model.head_free = bool(pen[4] == 1.0 or pen[5] == 1.0)   # lacet ou tangage appris
        if keep_samples:
            model.samples = (x, y, groups)
        return model

    # --- prédiction ---------------------------------------------------------------
    def predict(self, features: np.ndarray) -> np.ndarray:
        z = np.clip((np.asarray(features, float) - self.mean) / self.std, -6, 6)
        out = expand(z) @ self.weights
        return out[0] if np.ndim(features) == 1 else out

    # --- persistance ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return dict(version=1, features=FEATURE_NAMES, mean=self.mean.tolist(),
                    std=self.std.tolist(), weights=self.weights.tolist(), alpha=self.alpha,
                    screen=list(self.screen), cv_error=self.cv_error, created=self.created,
                    validation_error=self.validation_error, head_free=self.head_free,
                    samples=None if self.samples is None else dict(
                        x=np.round(self.samples[0], 5).tolist(),
                        y=np.round(self.samples[1], 1).tolist(),
                        groups=np.asarray(self.samples[2]).tolist()))

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GazeModel":
        d = json.loads(path.read_text(encoding="utf-8"))
        m = cls(d["mean"], d["std"], d["weights"], d["alpha"], d["screen"],
                d.get("cv_error"), d.get("created"))
        m.validation_error = d.get("validation_error")
        m.head_free = d.get("head_free", False)
        if d.get("samples"):
            s = d["samples"]
            m.samples = (np.array(s["x"], float), np.array(s["y"], float), np.array(s["groups"]))
        return m


class OneEuroFilter:
    """Filtre « 1€ » (Casiez et al.) : lisse fort à l'arrêt, réagit vite aux mouvements."""

    def __init__(self, min_cutoff: float = 0.4, beta: float = 0.003, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.t_prev is None or t - self.t_prev > 0.5:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        dx_hat = self.dx_prev + self._alpha(self.d_cutoff, dt) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        x_hat = self.x_prev + self._alpha(cutoff, dt) * (x - self.x_prev)
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat
