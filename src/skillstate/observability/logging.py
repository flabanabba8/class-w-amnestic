"""Structured logging helpers. Works with the standard library alone; Logfire is optional."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict) and extra:
            base += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class FieldsAdapter(logging.LoggerAdapter):
    """``log.info("msg", step=3)`` → structured ``fields``."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        fields = {k: v for k, v in kwargs.items() if k not in {"exc_info", "stack_info", "stacklevel", "extra"}}
        for k in fields:
            kwargs.pop(k)
        extra = kwargs.get("extra") or {}
        extra = {**extra, "fields": {**(self.extra or {}), **fields}}
        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging(level: str = "INFO", *, json_output: bool = False, stream: Any = None) -> None:
    root = logging.getLogger("skillstate")
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else KeyValueFormatter())
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str, **bound: Any) -> FieldsAdapter:
    return FieldsAdapter(logging.getLogger(f"skillstate.{name}"), bound)


def maybe_configure_logfire(enabled: bool) -> bool:
    """Configure Pydantic Logfire if requested and installed. Never required."""
    if not enabled:
        return False
    try:
        import logfire  # type: ignore[import-not-found]
    except ImportError:
        logging.getLogger("skillstate").warning("SKILLSTATE_LOGFIRE set but logfire is not installed (uv add logfire)")
        return False
    logfire.configure()
    try:
        logfire.instrument_pydantic_ai()
    except Exception:  # pragma: no cover - optional
        pass
    return True
