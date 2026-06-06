"""Append-only JSONL audit log of every retrieval and chat call."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from ..config import get_settings

_lock = Lock()


def log_event(event: str, **fields) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    path = Path(get_settings().audit_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
