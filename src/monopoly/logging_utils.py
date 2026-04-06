from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path


DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_BYTES = 1_000_000
LOG_BACKUP_COUNT = 5


def default_log_directory() -> Path:
    configured_directory = os.environ.get("MONOPOLY_LOG_DIR")
    if configured_directory:
        return Path(configured_directory).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "logs"


def resolve_log_level(level: str | int | None = None) -> int:
    if isinstance(level, int):
        return level
    configured_level = DEFAULT_LOG_LEVEL if level is None else str(level)
    configured_level = os.environ.get("MONOPOLY_LOG_LEVEL", configured_level).upper()
    resolved_level = getattr(logging, configured_level, None)
    if not isinstance(resolved_level, int):
        raise ValueError(f"Unsupported log level: {configured_level}")
    return resolved_level


def configure_process_logging(component: str, *, log_directory: str | Path | None = None, level: str | int | None = None) -> Path:
    directory = default_log_directory() if log_directory is None else Path(log_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{component}.log"
    level_value = resolve_log_level(level)

    monopoly_logger = logging.getLogger("monopoly")
    configured_path = getattr(monopoly_logger, "_monopoly_log_path", None)
    configured_level = getattr(monopoly_logger, "_monopoly_log_level", None)
    if configured_path == str(log_path) and configured_level == level_value and monopoly_logger.handlers:
        return log_path

    for handler in list(monopoly_logger.handlers):
        monopoly_logger.removeHandler(handler)
        handler.close()

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=MAX_LOG_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(level_value)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    monopoly_logger.setLevel(level_value)
    monopoly_logger.propagate = False
    monopoly_logger.addHandler(file_handler)
    monopoly_logger._monopoly_log_path = str(log_path)
    monopoly_logger._monopoly_log_level = level_value
    return log_path