"""Knowledge-base version helpers for cache invalidation and metadata."""

from __future__ import annotations

import os
from pathlib import Path

from config import config_data as config


def _version_path() -> Path:
    path = Path(config.kb_version_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_kb_version() -> int:
    """Return the current knowledge-base version, creating it when missing."""
    path = _version_path()
    if not path.exists():
        path.write_text("1", encoding="utf-8")
        return 1
    raw = path.read_text(encoding="utf-8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        path.write_text("1", encoding="utf-8")
        return 1


def bump_kb_version() -> int:
    """Increment and persist the knowledge-base version."""
    version = get_kb_version() + 1
    path = _version_path()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(str(version), encoding="utf-8")
    os.replace(tmp_path, path)
    return version
