"""Enregistrement des échantillons de regard dans une base SQLite locale (~/.regard/regard.db)."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from . import data_dir

ABSENT = "<visage non détecté>"
OFFSCREEN = "<hors écran>"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start REAL, end REAL, screen_w INTEGER, screen_h INTEGER
);
CREATE TABLE IF NOT EXISTS samples (
    session INTEGER, ts REAL, dt REAL,
    x REAL, y REAL,
    app TEXT, title TEXT,
    rel_x REAL, rel_y REAL, win_w INTEGER, win_h INTEGER
);
CREATE INDEX IF NOT EXISTS samples_ts ON samples(ts);
CREATE TABLE IF NOT EXISTS events (
    session INTEGER, ts REAL, kind TEXT, detail TEXT
);
"""


def db_path() -> Path:
    return data_dir() / "regard.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(str(path or db_path()), check_same_thread=False)
    con.executescript(SCHEMA)
    return con


class Recorder:
    """Accumule les échantillons et les écrit par lots (thread-safe)."""

    def __init__(self, screen, path: Path | None = None):
        self.con = connect(path)
        self._lock = threading.Lock()
        self._buf: list[tuple] = []
        self._events: list[tuple] = []
        self._last_flush = time.time()
        cur = self.con.execute("INSERT INTO sessions(start, screen_w, screen_h) VALUES (?,?,?)",
                               (time.time(), int(screen[0]), int(screen[1])))
        self.session = cur.lastrowid
        self.con.commit()

    def add(self, dt, x, y, app, title="", rel=(None, None), win=(None, None)):
        with self._lock:
            self._buf.append((self.session, time.time(), dt, x, y, app, title,
                              rel[0], rel[1], win[0], win[1]))
            if time.time() - self._last_flush > 2.0:
                self._flush_locked()

    def event(self, kind: str, detail: str = ""):
        with self._lock:
            self._events.append((self.session, time.time(), kind, detail))

    def _flush_locked(self):
        if self._buf:
            self.con.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)", self._buf)
            self._buf.clear()
        if self._events:
            self.con.executemany("INSERT INTO events VALUES (?,?,?,?)", self._events)
            self._events.clear()
        self.con.commit()
        self._last_flush = time.time()

    def close(self):
        with self._lock:
            self._flush_locked()
            self.con.execute("UPDATE sessions SET end=? WHERE id=?", (time.time(), self.session))
            self.con.commit()
            self.con.close()
