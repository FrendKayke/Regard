"""Regard — suivi du regard par webcam, par application."""
import os
from pathlib import Path

__version__ = "0.1.0"

DATA_DIR = Path(os.environ.get("REGARD_HOME", Path.home() / ".regard"))


def data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
