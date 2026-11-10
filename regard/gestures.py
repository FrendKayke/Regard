"""Gestes du visage (clin d'œil, bouche ouverte, sourcils levés…) et actions associées.

Les gestes sont détectés à partir des « blendshapes » MediaPipe (scores 0..1).
Utilisez `python regard.py preview` pour voir les scores en direct et ajuster les seuils
dans ~/.regard/gestures.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import data_dir

# Remarque : MediaPipe nomme « Left/Right » du point de vue de l'image ; vérifiez dans
# l'aperçu quel œil déclenche quel score et inversez si besoin.
DEFAULT_GESTURES = {
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
DEFAULT_ACTIONS = {
    "clin_oeil_gauche": "left_click",
    "clin_oeil_droit": "right_click",
    "yeux_fermes_long": "toggle_actions",
    "bouche_ouverte": "toggle_follow",
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
    return (cfg.get("gestures", DEFAULT_GESTURES), cfg.get("actions", DEFAULT_ACTIONS),
            cfg.get("options", {}))


@dataclass
class _State:
    since: Optional[float] = None
    fired: bool = False
    last_fire: float = -1e9


@dataclass
class GestureDetector:
    gestures: dict
    cooldown: float = 0.6
    states: dict = field(default_factory=dict)

    def _active(self, spec: dict, blend: dict) -> bool:
        return (all(blend.get(k, 0.0) >= v for k, v in spec.get("all_above", {}).items())
                and all(blend.get(k, 0.0) <= v for k, v in spec.get("all_below", {}).items()))

    def update(self, blend: Optional[dict], t: float) -> list[str]:
        """Renvoie les gestes déclenchés à cet instant (une fois par geste maintenu)."""
        fired = []
        for name, spec in self.gestures.items():
            st = self.states.setdefault(name, _State())
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
        if not self.enabled or action in (None, "none"):
            return None
        from pynput.mouse import Button
        if action == "toggle_follow":
            self.follow = not self.follow
            if self.on_change:
                self.on_change()
            return f"suivi du curseur {'activé' if self.follow else 'désactivé'}"
        if self.options.get("click_at", "gaze") == "gaze" and gaze is not None:
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
        if self.enabled and self.follow:
            self.mouse.position = (int(gaze[0]), int(gaze[1]))
