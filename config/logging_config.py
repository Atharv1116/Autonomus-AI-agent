"""
Logging configuration for the Autonomous Data Analyst Agent.

Provides structured logging with both console and file handlers.
Supports JSON-formatted log output for production environments.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color codes to console output."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color codes."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_file_logging: bool = True,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to './logs'.
        enable_file_logging: Whether to write logs to files.

    Returns:
        The root logger configured for the application.
    """
    logger = logging.getLogger("analyst_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers on re-initialization
    if logger.handlers:
        logger.handlers.clear()

    # --- Console Handler ---
    # Wrap stdout in a UTF-8 writer so Windows cp1252 consoles never crash
    # when log messages contain unicode characters (box-drawing, emoji, etc.)
    import io as _io
    utf8_stdout = _io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_format = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # --- File Handler ---
    if enable_file_logging:
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=os.path.join(log_dir, "analyst_agent.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ["httpx", "httpcore", "urllib3", "sqlalchemy.engine"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    logger.info("Logging initialized at %s level", level.upper())
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.

    Args:
        name: Module name (e.g., 'agents.planner').

    Returns:
        A logger instance scoped to the module.
    """
    return logging.getLogger(f"analyst_agent.{name}")
