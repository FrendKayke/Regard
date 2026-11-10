"""Analyse du visage : webcam -> repères MediaPipe -> caractéristiques du regard.

Utilise MediaPipe Face Landmarker (478 points dont les iris, 52 « blendshapes »
d'expression faciale et la pose de la tête).
"""
from __future__ import annotations

import math
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import data_dir

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Repères MediaPipe (caméra non inversée : l'œil droit du sujet est à gauche de l'image).
# Pour chaque œil : coin gauche-image, coin droit-image, paupière haute, paupière basse, iris.
EYES = {
    "right": dict(a=33, b=133, up=159, low=145, iris=[468, 469, 470, 471, 472]),
    "left": dict(a=362, b=263, up=386, low=374, iris=[473, 474, 475, 476, 477]),
}

FEATURE_NAMES = ["eye_x", "eye_y", "opening", "yaw", "pitch", "roll", "tx", "ty", "tz"]


@dataclass
class FaceFrame:
    """Résultat de l'analyse d'une image où un visage a été trouvé."""
    t: float                      # horodatage (time.monotonic)
    features: np.ndarray          # vecteur de FEATURE_NAMES
    blend: dict                   # blendshapes : nom -> score 0..1
    landmarks: np.ndarray = field(repr=False)  # (478, 2) en pixels
    per_eye: dict = field(default_factory=dict)

    @property
    def blink(self) -> float:
        return max(self.blend.get("eyeBlinkLeft", 0.0), self.blend.get("eyeBlinkRight", 0.0))


def ensure_model() -> Path:
    path = data_dir() / "face_landmarker.task"
    if not path.exists() or path.stat().st_size < 1_000_000:
        print("Téléchargement du modèle de visage MediaPipe (~4 Mo)…")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


def _wrap90(deg: float) -> float:
    return ((deg + 90.0) % 180.0) - 90.0


def head_pose(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Matrice 4x4 MediaPipe -> (lacet, tangage, roulis en degrés, tx, ty, tz en cm)."""
    m = np.asarray(matrix, dtype=float).reshape(4, 4)
    r = m[:3, :3]
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    pitch = _wrap90(math.degrees(math.atan2(r[2, 1], r[2, 2])))
    roll = _wrap90(math.degrees(math.atan2(r[1, 0], r[0, 0])))
    return yaw, pitch, roll, float(m[0, 3]), float(m[1, 3]), float(m[2, 3])


def eye_features(pts: np.ndarray, eye: dict) -> tuple[float, float, float]:
    """Position de l'iris dans le repère de l'œil (normalisée par la largeur de l'œil)."""
    a, b = pts[eye["a"]], pts[eye["b"]]
    axis = b - a
    width = float(np.linalg.norm(axis)) or 1e-6
    ex = axis / width
    ey = np.array([-ex[1], ex[0]])            # perpendiculaire, vers le bas de l'image
    center = (a + b) / 2.0
    iris = pts[eye["iris"]].mean(axis=0)
    d = iris - center
    rel_x = float(d @ ex) / width
    rel_y = float(d @ ey) / width
    opening = float(np.linalg.norm(pts[eye["up"]] - pts[eye["low"]])) / width
    return rel_x, rel_y, opening


class FaceAnalyzer:
    """Encapsule MediaPipe Face Landmarker en mode vidéo."""

    def __init__(self, model_path: Optional[Path] = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path or ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, bgr: np.ndarray, t: Optional[float] = None) -> Optional[FaceFrame]:
        t = time.monotonic() if t is None else t
        ts = max(int(t * 1000), self._last_ts + 1)   # MediaPipe exige des horodatages croissants
        self._last_ts = ts
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        h, w = bgr.shape[:2]
        pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=float)
        if len(pts) < 478:
            return None
        blend = {c.category_name: float(c.score) for c in (result.face_blendshapes or [[]])[0]}
        if result.facial_transformation_matrixes:
            pose = head_pose(result.facial_transformation_matrixes[0])
        else:
            pose = (0.0,) * 6
        r = eye_features(pts, EYES["right"])
        l = eye_features(pts, EYES["left"])
        features = np.array([(r[0] + l[0]) / 2, (r[1] + l[1]) / 2, (r[2] + l[2]) / 2, *pose])
        return FaceFrame(t=t, features=features, blend=blend, landmarks=pts,
                         per_eye={"right": r, "left": l})

    def close(self):
        self._landmarker.close()


def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la webcam n°{index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


class FaceSource(threading.Thread):
    """Fil d'exécution qui lit la webcam en continu et analyse chaque image.

    - `latest()` renvoie la dernière analyse (pour la calibration) ;
    - `on_frame(frame_or_None, t)` est appelé pour chaque image (pour le suivi).
    """

    def __init__(self, camera: int = 0, width: int = 1280, height: int = 720):
        super().__init__(daemon=True)
        self.camera, self.width, self.height = camera, width, height
        self.on_frame: Optional[Callable[[Optional[FaceFrame], float], None]] = None
        self.error: Optional[BaseException] = None
        self.fps = 0.0
        self._lock = threading.Lock()
        self._latest: tuple[int, Optional[FaceFrame], float] = (0, None, 0.0)
        self._stop = threading.Event()
        self.ready = threading.Event()

    def latest(self) -> tuple[int, Optional[FaceFrame], float]:
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            analyzer = FaceAnalyzer()
            cap = open_camera(self.camera, self.width, self.height)
        except BaseException as e:  # noqa: BLE001 — remonté à l'interface
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        frame_id, last = 0, time.monotonic()
        try:
            while not self._stop.is_set():
                ok, bgr = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                t = time.monotonic()
                face = analyzer.process(bgr, t)
                frame_id += 1
                dt = t - last
                last = t
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                with self._lock:
                    self._latest = (frame_id, face, t)
                cb = self.on_frame
                if cb is not None:
                    try:
                        cb(face, t)
                    except Exception as e:  # noqa: BLE001
                        print(f"[regard] erreur dans le traitement d'image : {e!r}")
        finally:
            cap.release()
            analyzer.close()
