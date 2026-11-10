"""Quelle application se trouve sous un point de l'écran ?

- Windows : API Win32 (WindowFromPoint) — support complet, y compris les applis UWP ;
- macOS   : Quartz (CGWindowList) — il faut l'autorisation « Enregistrement de l'écran »
            pour obtenir les titres de fenêtres ;
- Linux   : X11 via python-xlib (pas Wayland, qui interdit cette information).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import psutil

OWN_PID = os.getpid()


@dataclass
class WindowInfo:
    app: str            # nom de l'application (processus)
    title: str          # titre de la fenêtre
    rect: tuple         # (gauche, haut, largeur, hauteur) en pixels écran
    pid: int = 0

    def relative(self, x: float, y: float) -> tuple[float, float]:
        l, t, w, h = self.rect
        return ((x - l) / w if w else 0.0, (y - t) / h if h else 0.0)


_name_cache: dict[int, str] = {}


def process_name(pid: int) -> str:
    name = _name_cache.get(pid)
    if name is None:
        try:
            name = psutil.Process(pid).name()
        except (psutil.Error, ValueError):
            name = f"pid {pid}"
        if name.lower().endswith(".exe"):
            name = name[:-4]
        _name_cache[pid] = name
    return name


# --------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------
class _Win32Locator:
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self.ct, self.wt = ctypes, wintypes
        u = ctypes.windll.user32
        u.WindowFromPoint.argtypes = [wintypes.POINT]
        u.WindowFromPoint.restype = wintypes.HWND
        u.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
        u.GetAncestor.restype = wintypes.HWND
        u.GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
        u.GetWindow.restype = wintypes.HWND
        u.GetTopWindow.argtypes = [wintypes.HWND]
        u.GetTopWindow.restype = wintypes.HWND
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.IsWindowVisible.argtypes = [wintypes.HWND]
        self.u = u
        self.enum_proc_t = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _pid(self, hwnd) -> int:
        pid = self.wt.DWORD()
        self.u.GetWindowThreadProcessId(hwnd, self.ct.byref(pid))
        return pid.value

    def _title(self, hwnd) -> str:
        n = self.u.GetWindowTextLengthW(hwnd)
        buf = self.ct.create_unicode_buffer(n + 1)
        self.u.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def _rect(self, hwnd) -> tuple:
        r = self.wt.RECT()
        self.u.GetWindowRect(hwnd, self.ct.byref(r))
        return (r.left, r.top, r.right - r.left, r.bottom - r.top)

    def _uwp_child_pid(self, hwnd, host_pid) -> int:
        """Les applis UWP sont hébergées par ApplicationFrameHost : on cherche le vrai processus."""
        found = []

        def cb(child, _):
            p = self._pid(child)
            if p != host_pid:
                found.append(p)
                return False
            return True

        self.u.EnumChildWindows(hwnd, self.enum_proc_t(cb), 0)
        return found[0] if found else host_pid

    def at(self, x: int, y: int) -> Optional[WindowInfo]:
        hwnd = self.u.WindowFromPoint(self.wt.POINT(int(x), int(y)))
        if not hwnd:
            return None
        root = self.u.GetAncestor(hwnd, 2)  # GA_ROOT
        if root and self._pid(root) == OWN_PID:
            root = self._below_own(x, y)
            if not root:
                return None
        pid = self._pid(root)
        name = process_name(pid)
        if name == "ApplicationFrameHost":
            pid = self._uwp_child_pid(root, pid)
            name = process_name(pid)
        title = self._title(root)
        if name == "explorer" and not title:
            name = "Bureau / barre des tâches"
        return WindowInfo(name, title, self._rect(root), pid)

    def _below_own(self, x, y):
        """Parcourt les fenêtres de haut en bas en ignorant les nôtres (point du regard)."""
        h = self.u.GetTopWindow(None)
        while h:
            if self.u.IsWindowVisible(h) and self._pid(h) != OWN_PID:
                l, t, w, hh = self._rect(h)
                if l <= x < l + w and t <= y < t + hh:
                    return h
            h = self.u.GetWindow(h, 2)  # GW_HWNDNEXT
        return None


# --------------------------------------------------------------------------------------
# macOS
# --------------------------------------------------------------------------------------
class _QuartzLocator:
    def __init__(self):
        import Quartz  # pyobjc-framework-Quartz
        self.Q = Quartz
        self._cache, self._cache_t = [], 0.0

    def _windows(self):
        now = time.monotonic()
        if now - self._cache_t > 0.25:
            Q = self.Q
            opts = Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements
            self._cache = list(Q.CGWindowListCopyWindowInfo(opts, Q.kCGNullWindowID) or [])
            self._cache_t = now
        return self._cache

    def at(self, x, y) -> Optional[WindowInfo]:
        for w in self._windows():          # de l'avant vers l'arrière
            if w.get("kCGWindowLayer", 0) != 0 or w.get("kCGWindowOwnerPID") == OWN_PID:
                continue
            b = w.get("kCGWindowBounds", {})
            l, t, ww, hh = b.get("X", 0), b.get("Y", 0), b.get("Width", 0), b.get("Height", 0)
            if l <= x < l + ww and t <= y < t + hh:
                return WindowInfo(str(w.get("kCGWindowOwnerName", "?")),
                                  str(w.get("kCGWindowName", "") or ""),
                                  (l, t, ww, hh), int(w.get("kCGWindowOwnerPID", 0)))
        return WindowInfo("Bureau", "", (0, 0, 0, 0))


# --------------------------------------------------------------------------------------
# Linux (X11)
# --------------------------------------------------------------------------------------
class _X11Locator:
    def __init__(self):
        from Xlib import X, display
        self.X = X
        self.d = display.Display()
        self.root = self.d.screen().root
        a = self.d.intern_atom
        self.A = dict(stack=a("_NET_CLIENT_LIST_STACKING"), pid=a("_NET_WM_PID"),
                      name=a("_NET_WM_NAME"), utf8=a("UTF8_STRING"), state=a("_NET_WM_STATE"),
                      hidden=a("_NET_WM_STATE_HIDDEN"))
        self._cache, self._cache_t = [], 0.0

    def _windows(self):
        now = time.monotonic()
        if now - self._cache_t > 0.25:
            out = []
            prop = self.root.get_full_property(self.A["stack"], self.X.AnyPropertyType)
            if prop:
                wins = [self.d.create_resource_object("window", wid) for wid in prop.value]
            else:   # pas de gestionnaire de fenêtres compatible EWMH : arbre X brut
                wins = [w for w in self.root.query_tree().children
                        if w.get_attributes().map_state == self.X.IsViewable]
            for w in reversed(wins):             # du premier plan vers l'arrière-plan
                try:
                    st = w.get_full_property(self.A["state"], self.X.AnyPropertyType)
                    if st and self.A["hidden"] in st.value:
                        continue
                    g = w.get_geometry()
                    c = w.translate_coords(self.root, 0, 0)
                    p = w.get_full_property(self.A["pid"], self.X.AnyPropertyType)
                    n = w.get_full_property(self.A["name"], self.A["utf8"])
                    title = n.value.decode("utf-8", "replace") if n else (w.get_wm_name() or "")
                    pid = int(p.value[0]) if p else 0
                    cls = w.get_wm_class()
                    out.append((-c.x, -c.y, g.width, g.height, pid, title, cls))
                except Exception:  # noqa: BLE001 — fenêtre disparue entre-temps
                    continue
            self._cache, self._cache_t = out, now
        return self._cache

    def at(self, x, y) -> Optional[WindowInfo]:
        for l, t, w, h, pid, title, cls in self._windows():
            if pid == OWN_PID:
                continue
            if l <= x < l + w and t <= y < t + h:
                name = process_name(pid) if pid else (cls[0] if cls else "?")
                return WindowInfo(name, str(title), (l, t, w, h), pid)
        return WindowInfo("Bureau", "", (0, 0, 0, 0))


class _NullLocator:
    def at(self, x, y):
        return WindowInfo("Inconnu", "", (0, 0, 0, 0))


def make_locator():
    try:
        if sys.platform == "win32":
            return _Win32Locator()
        if sys.platform == "darwin":
            return _QuartzLocator()
        if os.environ.get("DISPLAY"):
            return _X11Locator()
    except Exception as e:  # noqa: BLE001
        print(f"[regard] détection des fenêtres indisponible : {e!r}")
    return _NullLocator()


def enable_dpi_awareness():
    """Windows : coordonnées en pixels physiques (sinon décalage avec la mise à l'échelle 125 %…)."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass
