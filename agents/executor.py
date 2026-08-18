"""
SQL Executor Agent.

Executes validated SQL queries against the database using SQLAlchemy.
Handles timeouts, row limits, and error recovery with user-friendly messages.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agents.state import AgentState
from config.logging_config import get_logger

logger = get_logger("agents.executor")


class ExecutorAgent:
    """
    Executes validated SQL queries and returns results as DataFrames.

    Uses SQLAlchemy for dialect-agnostic execution with timeout
    support, row limits, and comprehensive error handling.
    """

    def __init__(
        self,
        engine: Engine,
        max_rows: int = 10000,
        timeout_seconds: int = 30,
    ) -> None:
        """
        Initialize the Executor Agent.

        Args:
            engine: SQLAlchemy engine for query execution.
            max_rows: Maximum number of rows to return.
            timeout_seconds: Query execution timeout.
        """
        self._engine = engine
        self._max_rows = max_rows
        self._timeout = timeout_seconds
        logger.info(
            "ExecutorAgent initialized (max_rows=%d, timeout=%ds)",
            max_rows, timeout_seconds,
        )

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the validated SQL query.

        Args:
            state: Current workflow state with 'generated_sql' and 'guardrail_result'.

        Returns:
            Updated state with 'query_results', 'result_columns', 'result_row_count'.
        """
        sql = state.get("generated_sql", "")
        guardrail = state.get("guardrail_result", {})

        # Safety check: only execute if guardrail passed
        if not guardrail.get("is_valid", False):
            state["error"] = "Cannot execute: SQL failed guardrail validation"
            state["current_step"] = "executor"
            logger.warning("Execution blocked: guardrail validation failed")
            return state

        if not sql.strip():
            state["error"] = "Cannot execute: empty SQL query"
            state["current_step"] = "executor"
            return state

        # Pre-execution sanity: reject if SQL still contains obvious non-SQL content
        non_sql_markers = [
            '```',      # leftover code fences
            '| ---',    # markdown table separator
            '\n|',      # pipe-table rows
            '\n#',      # markdown headings
            '\n*',      # markdown list items
        ]
        for marker in non_sql_markers:
            if marker in sql:
                state["error"] = (
                    f"SQL syntax error: generated query contains non-SQL content "
                    f"({repr(marker)}). The model included markdown — regenerating."
                )
                state["generated_sql"] = ""
                state["current_step"] = "executor"
                logger.warning("Execution blocked: non-SQL content in query: %s", sql[:200])
                return state

        logger.info("Executing SQL: %s", sql[:200])

        try:
            df = self._execute_query(sql)

            # Apply row limit
            if len(df) > self._max_rows:
                logger.warning(
                    "Result truncated: %d rows → %d rows",
                    len(df), self._max_rows,
                )
                df = df.head(self._max_rows)

            state["query_results"] = df.to_dict(orient="records")
            state["result_columns"] = list(df.columns)
            state["result_row_count"] = len(df)
            state["current_step"] = "executor"
            state["error"] = None

            logger.info(
                "Query executed: %d rows × %d columns",
                len(df), len(df.columns),
            )

        except Exception as e:
            error_msg = self._format_error(e)
            state["error"] = error_msg
            state["query_results"] = []
            state["result_columns"] = []
            state["result_row_count"] = 0
            state["current_step"] = "executor"
            logger.exception("Query execution failed")

        return state

    def _execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL and return results as a DataFrame.

        Args:
            sql: Validated SQL query string.

        Returns:
            Query results as a Pandas DataFrame.

        Raises:
            Exception: If query execution fails.
        """
        with self._engine.connect() as conn:
            # Set statement timeout for PostgreSQL
            dialect = self._engine.dialect.name
            if dialect == "postgresql" and self._timeout:
                conn.execute(
                    text(f"SET statement_timeout = {self._timeout * 1000}")
                )

            result = conn.execute(text(sql))

            if result.returns_rows:
                columns = list(result.keys())
                rows = result.fetchall()
                df = pd.DataFrame(rows, columns=columns)

                # Convert any non-serializable types
                for col in df.columns:
                    if df[col].dtype == "object":
                        df[col] = df[col].astype(str)

                return df
            else:
                # Query executed but returned no rows
                return pd.DataFrame()

    @staticmethod
    def _format_error(error: Exception) -> str:
        """
        Format a database error into a user-friendly message.

        Args:
            error: The caught exception.

        Returns:
            User-friendly error message with diagnostic info.
        """
        error_str = str(error).lower()

        # Common error patterns and friendly messages
        error_map = {
            "column": (
                "Column not found. The query references a column that "
                "doesn't exist in the database. Let me regenerate the SQL."
            ),
            "relation": (
                "Table not found. The query references a table that "
                "doesn't exist. Let me check the schema and try again."
            ),
            "syntax": (
                "SQL syntax error. The generated query has a syntax issue. "
                "Let me fix it and try again."
            ),
            "permission": (
                "Permission denied. The database user doesn't have access "
                "to the requested data."
            ),
            "timeout": (
                "Query timed out. The query took too long to execute. "
                "Try asking for a more specific subset of data."
            ),
            "connection": (
                "Database connection error. Please check that the database "
                "is running and accessible."
            ),
            "division by zero": (
                "Division by zero error in the query. Let me adjust "
                "the calculation to handle zero values."
            ),
        }

        for keyword, friendly_msg in error_map.items():
            if keyword in error_str:
                return f"{friendly_msg}\n\nTechnical detail: {str(error)[:500]}"

        return f"Query execution error: {str(error)[:500]}"

    def test_connection(self) -> bool:
        """
        Test the database connection.

        Returns:
            True if connection is successful.
        """
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
