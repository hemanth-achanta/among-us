"""
Structured logging utilities.

Supports two output formats:
  - "json"  → machine-parseable JSON lines (ideal for log aggregators)
  - "text"  → human-readable coloured output for local development

Usage
-----
    from utils.logger import get_logger

    log = get_logger(__name__)
    log.info("query_executed", sql="SELECT 1", duration_ms=12.4)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Lazy import to avoid hard dependency at module load time ──────────────────
try:
    import structlog  # type: ignore
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


# ── Constants ─────────────────────────────────────────────────────────────────

_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_INITIALIZED = False


# ── Internal helpers ──────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }

        # Attach any extra kwargs passed via log.info("...", extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class _BoundLogger:
    """
    Thin wrapper that adds structured key-value context to every log call
    without requiring structlog.
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        self._context: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> "_BoundLogger":
        """Return a new logger with additional context fields."""
        child = _BoundLogger(self._logger.name)
        child._context = {**self._context, **kwargs}
        return child

    def _log(self, level_no: int, event: str, **kwargs: Any) -> None:
        """
        Internal helper to send a log record.

        Note: the parameter is named ``level_no`` (not ``level``) so callers are
        free to include a ``level=...`` field in the structured context without
        colliding with the function signature.
        """
        extra = {**self._context, **kwargs}
        self._logger.log(level_no, event, extra=extra)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._logger.exception(event, extra={**self._context, **kwargs})


# ── Public API ────────────────────────────────────────────────────────────────

def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: str | None = None,
) -> None:
    """
    Initialise the root logging configuration.

    Must be called once at application startup before any loggers are used.

    Parameters
    ----------
    level:    One of DEBUG / INFO / WARNING / ERROR / CRITICAL.
    fmt:      "json" or "text".
    log_file: Optional path to a file; logs are written there in addition to
              stdout.
    """
    global _INITIALIZED

    numeric_level = _LOG_LEVEL_MAP.get(level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # ── stdout handler ──────────────────────────────────────────────────────
    stdout_handler = logging.StreamHandler(sys.stdout)
    if fmt.lower() == "json":
        stdout_handler.setFormatter(_JsonFormatter())
    else:
        stdout_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    handlers.append(stdout_handler)

    # ── file handler ────────────────────────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(_JsonFormatter())   # always JSON to file
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,  # override any previous basicConfig calls
    )

    # Silence noisy third-party libraries
    for noisy in ("sqlalchemy.engine", "urllib3", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str) -> _BoundLogger:
    """
    Return a bound logger for the given module name.

    Example
    -------
        log = get_logger(__name__)
        log.info("sql_generated", sql=query, model=model_id, duration_ms=42)
    """
    return _BoundLogger(name)


# ── Convenience: elapsed-time context manager ─────────────────────────────────

class Timer:
    """Context manager that measures elapsed wall-clock time in milliseconds."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1_000
