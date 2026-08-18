"""
Export utilities for data, charts, and SQL.

Provides methods to export query results as CSV, Plotly charts
as PNG images, and generated SQL as .sql files.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.logging_config import get_logger

logger = get_logger("utils.export")


class ExportManager:
    """
    Handles exporting of query results, visualizations, and SQL.

    Supports in-memory exports (for Streamlit downloads) and
    file-based exports (for batch operations).
    """

    def __init__(self, export_dir: str = "exports") -> None:
        """
        Initialize the export manager.

        Args:
            export_dir: Default directory for file-based exports.
        """
        self._export_dir = export_dir

    def _ensure_dir(self) -> None:
        """Create export directory if it doesn't exist."""
        Path(self._export_dir).mkdir(parents=True, exist_ok=True)

    def export_csv(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None,
        to_bytes: bool = True,
    ) -> str | bytes:
        """
        Export a DataFrame as CSV.

        Args:
            df: Pandas DataFrame to export.
            filename: Output filename (without extension). If provided, saves to disk.
            to_bytes: If True, return CSV as bytes (for Streamlit download).

        Returns:
            CSV content as bytes (if to_bytes=True) or file path string.
        """
        if to_bytes:
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, encoding="utf-8")
            csv_bytes = buffer.getvalue().encode("utf-8")
            logger.info("CSV exported: %d rows, %d bytes", len(df), len(csv_bytes))
            return csv_bytes

        self._ensure_dir()
        filepath = os.path.join(self._export_dir, f"{filename or 'export'}.csv")
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.info("CSV saved: %s (%d rows)", filepath, len(df))
        return filepath

    def export_chart_png(
        self,
        fig: Any,
        filename: Optional[str] = None,
        to_bytes: bool = True,
        width: int = 1200,
        height: int = 700,
    ) -> str | bytes:
        """
        Export a Plotly figure as PNG.

        Args:
            fig: Plotly figure object.
            filename: Output filename (without extension).
            to_bytes: If True, return PNG as bytes.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            PNG content as bytes or file path string.
        """
        try:
            if to_bytes:
                png_bytes = fig.to_image(format="png", width=width, height=height)
                logger.info("Chart PNG exported: %d bytes", len(png_bytes))
                return png_bytes

            self._ensure_dir()
            filepath = os.path.join(self._export_dir, f"{filename or 'chart'}.png")
            fig.write_image(filepath, width=width, height=height)
            logger.info("Chart PNG saved: %s", filepath)
            return filepath

        except Exception as e:
            logger.warning(
                "PNG export failed (kaleido might not be installed): %s. "
                "Falling back to HTML export.",
                str(e),
            )
            # Fallback to HTML if kaleido is not available
            if to_bytes:
                html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                return html_bytes
            else:
                self._ensure_dir()
                filepath = os.path.join(self._export_dir, f"{filename or 'chart'}.html")
                fig.write_html(filepath, include_plotlyjs="cdn")
                return filepath

    def export_sql(
        self,
        sql: str,
        filename: Optional[str] = None,
        to_bytes: bool = True,
        metadata: Optional[dict[str, str]] = None,
    ) -> str | bytes:
        """
        Export generated SQL as a .sql file.

        Args:
            sql: The SQL query string.
            filename: Output filename (without extension).
            to_bytes: If True, return SQL as bytes.
            metadata: Optional metadata (question, timestamp, etc.) to include as comments.

        Returns:
            SQL content as bytes or file path string.
        """
        lines = ["-- Auto-generated SQL by Autonomous Data Analyst Agent"]

        if metadata:
            lines.append("-- " + "-" * 58)
            for key, value in metadata.items():
                lines.append(f"-- {key}: {value}")
            lines.append("-- " + "-" * 58)

        lines.append("")
        lines.append(sql.strip())
        lines.append("")

        content = "\n".join(lines)

        if to_bytes:
            sql_bytes = content.encode("utf-8")
            logger.info("SQL exported: %d bytes", len(sql_bytes))
            return sql_bytes

        self._ensure_dir()
        filepath = os.path.join(self._export_dir, f"{filename or 'query'}.sql")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("SQL saved: %s", filepath)
        return filepath
