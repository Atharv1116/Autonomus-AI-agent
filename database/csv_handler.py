"""
CSV upload handler for temporary analysis.

Allows users to upload CSV files that are loaded into temporary
SQLite tables for ad-hoc querying within the same session.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from config.logging_config import get_logger

logger = get_logger("database.csv_handler")


class CSVHandler:
    """
    Handles CSV file uploads for temporary SQL-based analysis.

    Creates an in-memory SQLite database to store uploaded CSV data,
    allowing users to run SQL queries against uploaded files.
    """

    def __init__(self) -> None:
        """Initialize with an in-memory SQLite engine."""
        self._engine: Optional[Engine] = None
        self._loaded_tables: dict[str, dict] = {}

    @property
    def engine(self) -> Engine:
        """Get or create the temporary SQLite engine."""
        if self._engine is None:
            self._engine = create_engine("sqlite:///:memory:")
            logger.info("Temporary SQLite engine created for CSV analysis")
        return self._engine

    def load_csv(
        self,
        file_path: str | None = None,
        file_content: bytes | None = None,
        file_name: str = "uploaded_data",
        table_name: str | None = None,
    ) -> dict:
        """
        Load a CSV file into a temporary SQLite table.

        Args:
            file_path: Path to the CSV file on disk.
            file_content: Raw bytes of the CSV file (for Streamlit uploads).
            file_name: Original file name (used for default table name).
            table_name: Override table name. Auto-generated from file_name if None.

        Returns:
            Dictionary with table info: {table_name, columns, row_count, dtypes}.
        """
        # Determine table name
        if table_name is None:
            # Sanitize file name for use as table name
            table_name = os.path.splitext(file_name)[0]
            table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
            table_name = table_name.lower().strip("_")

        # Read CSV with automatic encoding detection
        if file_content is not None:
            import io as _io
            df = self._read_csv_bytes(file_content)
        elif file_path is not None:
            with open(file_path, "rb") as fh:
                df = self._read_csv_bytes(fh.read())
        else:
            raise ValueError("Either file_path or file_content must be provided")

        # Clean column names (SQL-safe)
        df.columns = [
            "".join(c if c.isalnum() or c == "_" else "_" for c in col).lower().strip("_")
            for col in df.columns
        ]

        # Load into SQLite
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)

        table_info = {
            "table_name": table_name,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "row_count": len(df),
            "sample_data": df.head(5).to_dict(orient="records"),
        }

        self._loaded_tables[table_name] = table_info
        logger.info(
            "CSV loaded as table '%s': %d rows, %d columns",
            table_name, len(df), len(df.columns),
        )
        return table_info

    @staticmethod
    def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
        """
        Read CSV bytes into a DataFrame with automatic encoding detection.

        pandas raises ``ParserError`` (a ``ValueError``) — not a plain
        ``UnicodeDecodeError`` — when its C parser hits a bad byte, so we
        must catch ``Exception`` broadly in each attempt.

        Strategy:
        1. chardet auto-detection (best accuracy)
        2. Ranked fallback list of common encodings
        3. latin-1 as the guaranteed-safe last resort
           (ISO 8859-1 maps every possible byte 0x00-0xFF, so it
           never raises a decode error — though some chars may look wrong)

        Args:
            raw: Raw bytes of the CSV file.

        Returns:
            Parsed DataFrame.
        """
        import io

        # ── 1. chardet auto-detection ─────────────────────────────────────
        try:
            import chardet
            detected = chardet.detect(raw)
            enc = (detected.get("encoding") or "utf-8").strip()
            logger.debug("chardet: encoding=%s confidence=%.2f", enc, detected.get("confidence", 0))
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except ImportError:
            logger.debug("chardet not installed, falling back to manual list")
        except Exception as exc:
            logger.debug("chardet-detected encoding failed (%s), trying fallbacks", exc)

        # ── 2. Common encoding fallbacks ──────────────────────────────────
        fallback_encodings = [
            "utf-8",        # try pure UTF-8 first
            "utf-8-sig",    # UTF-8 with BOM (Excel CSV exports)
            "cp1252",       # Windows Western European (the most common culprit)
            "iso-8859-15",  # Latin-1 variant with € symbol
            "utf-16",       # Some Excel / Windows exports
        ]
        for enc in fallback_encodings:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                logger.info("CSV decoded with encoding=%s", enc)
                return df
            except Exception:
                continue

        # ── 3. latin-1 — guaranteed safe (maps all 256 byte values) ──────
        logger.warning(
            "All encodings failed; using latin-1 (characters may display oddly)"
        )
        return pd.read_csv(io.BytesIO(raw), encoding="latin-1")

    def get_loaded_tables(self) -> dict[str, dict]:
        """Return info about all loaded CSV tables."""
        return self._loaded_tables

    def get_schema_for_llm(self) -> str:
        """
        Generate LLM-readable schema for uploaded CSV tables.

        Returns:
            Schema description string, or empty string if no tables loaded.
        """
        if not self._loaded_tables:
            return ""

        lines = [
            "\nUPLOADED CSV TABLES (Temporary)",
            "=" * 40,
        ]

        for table_name, info in self._loaded_tables.items():
            lines.append(f"\nTABLE: {table_name} ({info['row_count']} rows)")
            lines.append("-" * 30)
            for col in info["columns"]:
                dtype = info["dtypes"].get(col, "unknown")
                lines.append(f"  • {col}: {dtype}")

        return "\n".join(lines)

    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query against the temporary CSV tables.

        Args:
            sql: SQL query string.

        Returns:
            Query results as a Pandas DataFrame.
        """
        return pd.read_sql(sql, self.engine)

    def remove_table(self, table_name: str) -> bool:
        """
        Remove a loaded CSV table.

        Args:
            table_name: Name of the table to remove.

        Returns:
            True if table was removed, False if not found.
        """
        if table_name in self._loaded_tables:
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()
            del self._loaded_tables[table_name]
            logger.info("Removed temporary table: %s", table_name)
            return True
        return False

    def cleanup(self) -> None:
        """Remove all temporary tables and dispose of the engine."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
        self._loaded_tables.clear()
        logger.info("CSV handler cleaned up")
