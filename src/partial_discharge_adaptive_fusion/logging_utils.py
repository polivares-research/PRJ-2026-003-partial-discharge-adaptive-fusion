"""Timestamped console/file progress logging for long V5 runs."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class _ElapsedFormatter(logging.Formatter):
    def __init__(self, started: float) -> None:
        super().__init__()
        self.started = started

    def format(self, record: logging.LogRecord) -> str:
        now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.perf_counter() - self.started
        return f"[{now}] [elapsed={elapsed:09.1f}s] {record.getMessage()}"


def configure_progress_logging(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Configure unbuffered console logging and optional mirrored file output."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    started = time.perf_counter()
    formatter = _ElapsedFormatter(started)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    if log_file is not None:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(destination, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    logger.info("logging initialized")
    return logger


def log_progress(
    logger: logging.Logger,
    stage: str,
    completed: int,
    total: int,
    started: float,
    *,
    every: int = 25,
    force: bool = False,
) -> None:
    """Emit progress, rate, and ETA at bounded intervals."""

    if not force and completed != total and completed % max(1, every) != 0:
        return
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0 else float("inf")
    percentage = 100.0 * completed / total if total else 100.0
    eta_text = f"{eta:.1f}s" if eta != float("inf") else "unknown"
    logger.info(
        f"{stage}: {completed}/{total} ({percentage:.1f}%), "
        f"rate={rate:.2f}/s, eta={eta_text}"
    )
