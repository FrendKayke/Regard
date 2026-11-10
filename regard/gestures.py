"""Gestes du visage (clin d'œil, bouche ouverte, sourcils levés…) et actions associées.

Les gestes sont détectés à partir des « blendshapes » MediaPipe (scores 0..1).
Utilisez `python regard.py preview` pour voir les scores en direct et ajuster les seuils
dans ~/.regard/gestures.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import data_dir

# Remarque : MediaPipe nomme « Left/Right » du point de vue de l'image ; vérifiez dans
# l'aperçu quel œil déclenche quel score et inversez si besoin.
DEFAULT_GESTURES = {
    # 3 clignements rapides (les deux yeux) en moins de 1,5 s
    "triple_clignement": {"type": "blinks", "count": 3, "window": 1.5,
                          "threshold": 0.5, "max_closed": 0.45},
    "clin_oeil_gauche": {"all_above": {"eyeBlinkLeft": 0.55}, "all_below": {"eyeBlinkRight": 0.3},
                         "hold": 0.25},
    "clin_oeil_droit": {"all_above": {"eyeBlinkRight": 0.55}, "all_below": {"eyeBlinkLeft": 0.3},
                        "hold": 0.25},
    "yeux_fermes_long": {"all_above": {"eyeBlinkLeft": 0.55, "eyeBlinkRight": 0.55}, "hold": 0.8},
    "bouche_ouverte": {"all_above": {"jawOpen": 0.45}, "hold": 0.3},
    "sourcils_leves": {"all_above": {"browInnerUp": 0.6}, "hold": 0.3},
    "sourire": {"all_above": {"mouthSmileLeft": 0.6, "mouthSmileRight": 0.6}, "hold": 0.4},
}

# Actions possibles : left_click, right_click, double_click, scroll_up, scroll_down,
# toggle_follow (le curseur suit le regard), toggle_actions, none.
# toggle_follow fonctionne toujours ; les autres actions seulement si les actions sont activées.
DEFAULT_ACTIONS = {
    "triple_clignement": "toggle_follow",
    "clin_oeil_gauche": "left_click",
    "clin_oeil_droit": "right_click",
    "yeux_fermes_long": "toggle_actions",
    "bouche_ouverte": "none",
    "sourcils_leves": "scroll_up",
    "sourire": "none",
}


def load_config() -> tuple[dict, dict, dict]:
    path = data_dir() / "gestures.json"
    if not path.exists():
        path.write_text(json.dumps({"gestures": DEFAULT_GESTURES, "actions": DEFAULT_ACTIONS,
                                    "options": {"click_at": "gaze"}},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    # Les gestes ajoutés dans une nouvelle version complètent la configuration existante.
    gestures = {**DEFAULT_GESTURES, **cfg.get("gestures", {})}
    actions = {**DEFAULT_ACTIONS, **cfg.get("actions", {})}
    return gestures, actions, cfg.get("options", {})


@dataclass
class _State:
    since: Optional[float] = None
    fired: bool = False
    last_fire: float = -1e9
    closed_since: Optional[float] = None
    blinks: list = field(default_factory=list)


@dataclass
class GestureDetector:
    gestures: dict
    cooldown: float = 0.6
    states: dict = field(default_factory=dict)

    def _active(self, spec: dict, blend: dict) -> bool:
        return (all(blend.get(k, 0.0) >= v for k, v in spec.get("all_above", {}).items())
                and all(blend.get(k, 0.0) <= v for k, v in spec.get("all_below", {}).items()))

    def _blinks(self, name, spec, st: _State, blend, t) -> bool:
        """Compte les clignements brefs des deux yeux ; vrai quand il y en a `count` d'affilée."""
        th = spec.get("threshold", 0.5)
        closed = blend is not None and min(blend.get("eyeBlinkLeft", 0.0),
                                           blend.get("eyeBlinkRight", 0.0)) >= th
        if closed:
            if st.closed_since is None:
                st.closed_since = t
            return False
        if st.closed_since is not None:
            dur = t - st.closed_since
            st.closed_since = None
            if 0.03 <= dur <= spec.get("max_closed", 0.45):
                st.blinks.append(t)
            else:
                st.blinks.clear()           # fermeture trop longue : ce n'est pas un clignement
        st.blinks = [b for b in st.blinks if t - b <= spec.get("window", 1.5)]
        if len(st.blinks) >= spec.get("count", 3) and t - st.last_fire >= self.cooldown:
            st.blinks.clear()
            st.last_fire = t
            return True
        return False

    def update(self, blend: Optional[dict], t: float) -> list[str]:
        """Renvoie les gestes déclenchés à cet instant (une fois par geste maintenu)."""
        fired = []
        for name, spec in self.gestures.items():
            st = self.states.setdefault(name, _State())
            if spec.get("type") == "blinks":
                if self._blinks(name, spec, st, blend, t):
                    fired.append(name)
                continue
            if blend is not None and self._active(spec, blend):
                if st.since is None:
                    st.since = t
                if (not st.fired and t - st.since >= spec.get("hold", 0.3)
                        and t - st.last_fire >= self.cooldown):
                    st.fired, st.last_fire = True, t
                    fired.append(name)
            else:
                st.since, st.fired = None, False
        # Un clin d'œil fait partie d'un « yeux fermés » : on ne garde que le plus spécifique.
        if "yeux_fermes_long" in fired:
            fired = [g for g in fired if not g.startswith("clin_oeil")]
        return fired


class ActionRunner:
    """Exécute les actions souris (via pynput). Désactivé tant que `enabled` est faux."""

    def __init__(self, actions: dict, options: dict):
        self.actions, self.options = actions, options
        self.enabled = False
        self.follow = False
        self._mouse = None
        self.last_synthetic = -1e9          # dernière action souris générée par nous
        self._cursor: Optional[tuple] = None
        # zone morte (px) : le curseur ne bouge pas tant que le regard reste dans ce rayon
        self.deadzone = float(options.get("follow_deadzone", 45))
        self.smoothing = float(options.get("follow_smoothing", 0.35))
        self.on_change: Optional[Callable[[], None]] = None

    @property
    def mouse(self):
        if self._mouse is None:
            from pynput.mouse import Controller
            self._mouse = Controller()
        return self._mouse

    def handle(self, gesture: str, gaze: Optional[tuple]) -> Optional[str]:
        action = self.actions.get(gesture, "none")
        if action == "toggle_actions":
            self.enabled = not self.enabled
            self.follow = self.follow and self.enabled
            if self.on_change:
                self.on_change()
            return f"actions {'activées' if self.enabled else 'désactivées'}"
        if action == "toggle_follow":
            self.follow = not self.follow
            self._cursor = None
            if self.on_change:
                self.on_change()
            return f"curseur piloté par le regard : {'OUI' if self.follow else 'non'}"
        if not self.enabled or action in (None, "none"):
            return None
        from pynput.mouse import Button
        self.last_synthetic = time.monotonic()
        if self.options.get("click_at", "gaze") == "gaze" and gaze is not None and not self.follow:
            self.mouse.position = (int(gaze[0]), int(gaze[1]))
        if action == "left_click":
            self.mouse.click(Button.left, 1)
        elif action == "right_click":
            self.mouse.click(Button.right, 1)
        elif action == "double_click":
            self.mouse.click(Button.left, 2)
        elif action == "scroll_up":
            self.mouse.scroll(0, 3)
        elif action == "scroll_down":
            self.mouse.scroll(0, -3)
        else:
            return None
        return action

    def follow_gaze(self, gaze: tuple):
        """Déplace le curseur vers le regard, avec zone morte et lissage contre les tremblements."""
        if not self.follow:
            return
        gx, gy = float(gaze[0]), float(gaze[1])
        if self._cursor is None:
            self._cursor = (gx, gy)
        else:
            cx, cy = self._cursor
            d = ((gx - cx) ** 2 + (gy - cy) ** 2) ** 0.5
            if d < self.deadzone:
                return
            k = self.smoothing if d < 4 * self.deadzone else 0.8   # grand saut : on suit vite
            self._cursor = (cx + (gx - cx) * k, cy + (gy - cy) * k)
        self.last_synthetic = time.monotonic()
        self.mouse.position = (int(self._cursor[0]), int(self._cursor[1]))
