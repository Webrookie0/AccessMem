"""Rotating JSONL audit events with optional question logging."""

import hashlib
import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ..config import get_settings

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


@lru_cache(maxsize=8)
def _logger(path_string: str, max_bytes: int, backups: int) -> logging.Logger:
    path = Path(path_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"accessmem.audit.{path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(event: str, **fields) -> None:
    settings = get_settings()
    question = fields.pop("question", None)
    if question is not None:
        fields["question_hash"] = hashlib.sha256(question.encode()).hexdigest()
        fields["question_length"] = len(question)
        if settings.audit_log_questions:
            fields["question"] = question
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        "request_id": request_id.get(),
        **fields,
    }
    _logger(
        settings.audit_log_path, settings.audit_log_max_bytes, settings.audit_log_backup_count
    ).info(json.dumps(record, ensure_ascii=False, default=str))
