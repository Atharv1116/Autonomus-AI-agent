"""
Execution timing utilities.

Provides a context manager and decorator for timing individual
agent steps and accumulating a full execution timeline.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from config.logging_config import get_logger

logger = get_logger("utils.timer")


class ExecutionTimer:
    """
    Tracks execution time across pipeline steps.

    Supports both context manager and decorator patterns.
    Accumulates per-step timing for the full pipeline.
    """

    def __init__(self) -> None:
        """Initialize the execution timer."""
        self._steps: list[dict[str, Any]] = []
        self._start_time: float = time.time()

    @contextmanager
    def track(self, step_name: str) -> Generator[None, None, None]:
        """
        Context manager to time a specific step.

        Args:
            step_name: Name of the step being timed.

        Yields:
            None. Duration is recorded on exit.

        Usage:
            with timer.track("planner"):
                result = planner.run(...)
        """
        start = time.perf_counter()
        logger.debug("Timer started: %s", step_name)

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._steps.append({
                "step": step_name,
                "duration_seconds": round(elapsed, 3),
                "duration_formatted": self._format_duration(elapsed),
            })
            logger.info("Timer [%s]: %.3fs", step_name, elapsed)

    @property
    def total_duration(self) -> float:
        """Total duration across all tracked steps in seconds."""
        return sum(s["duration_seconds"] for s in self._steps)

    @property
    def total_elapsed(self) -> float:
        """Wall-clock time since timer was created."""
        return time.time() - self._start_time

    @property
    def step_breakdown(self) -> list[dict[str, Any]]:
        """Return per-step timing breakdown."""
        return self._steps.copy()

    def get_summary(self) -> dict[str, Any]:
        """
        Return a complete timing summary.

        Returns:
            Dictionary with total and per-step timing info.
        """
        total = self.total_duration
        return {
            "total_seconds": round(total, 3),
            "total_formatted": self._format_duration(total),
            "num_steps": len(self._steps),
            "steps": self._steps.copy(),
            "slowest_step": max(self._steps, key=lambda s: s["duration_seconds"])["step"] if self._steps else None,
        }

    def reset(self) -> None:
        """Reset all tracked timings."""
        self._steps.clear()
        self._start_time = time.time()
        logger.debug("Timer reset")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """
        Format a duration in seconds to a human-readable string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted string (e.g., '1.23s', '45.6ms').
        """
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.0f}µs"
        elif seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
