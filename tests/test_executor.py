"""
Tests for the Executor Agent.

Validates query execution against SQLite in-memory database,
error handling, row limits, and result formatting.
"""

from __future__ import annotations

import pytest
from sqlalchemy.engine import Engine

from agents.executor import ExecutorAgent


class TestExecutorAgent:
    """Test the Executor Agent."""

    def test_execute_simple_select(self, test_engine: Engine) -> None:
        """Should execute a simple SELECT and return results."""
        executor = ExecutorAgent(test_engine, max_rows=100)
        state = {
            "generated_sql": "SELECT * FROM products",
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)

        assert result["error"] is None
        assert result["result_row_count"] == 5
        assert "product_name" in result["result_columns"]
        assert len(result["query_results"]) == 5

    def test_execute_with_where_clause(self, test_engine: Engine) -> None:
        """Should execute SELECT with WHERE clause."""
        executor = ExecutorAgent(test_engine)
        state = {
            "generated_sql": "SELECT product_name, unit_price FROM products WHERE unit_price > 50",
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)

        assert result["error"] is None
        assert result["result_row_count"] > 0
        # All prices should be > 50
        for row in result["query_results"]:
            assert float(row["unit_price"]) > 50

    def test_execute_with_aggregation(self, test_engine: Engine) -> None:
        """Should execute aggregation queries."""
        executor = ExecutorAgent(test_engine)
        state = {
            "generated_sql": "SELECT category, COUNT(*) as count, AVG(unit_price) as avg_price FROM products GROUP BY category",
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)

        assert result["error"] is None
        assert result["result_row_count"] > 0
        assert "count" in result["result_columns"]
        assert "avg_price" in result["result_columns"]

    def test_execute_with_join(self, test_engine: Engine) -> None:
        """Should execute JOIN queries."""
        executor = ExecutorAgent(test_engine)
        state = {
            "generated_sql": (
                "SELECT p.product_name, SUM(s.quantity) as total_sold "
                "FROM products p "
                "JOIN sales s ON p.product_id = s.product_id "
                "GROUP BY p.product_name "
                "ORDER BY total_sold DESC"
            ),
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)

        assert result["error"] is None
        assert result["result_row_count"] > 0

    def test_blocks_execution_when_guardrail_fails(self, test_engine: Engine) -> None:
        """Should NOT execute when guardrail has failed."""
        executor = ExecutorAgent(test_engine)
        state = {
            "generated_sql": "SELECT * FROM products",
            "guardrail_result": {"is_valid": False, "reason": "Blocked"},
        }

        result = executor.run(state)

        assert result["error"] is not None
        assert "guardrail" in result["error"].lower()

    def test_handles_invalid_sql(self, test_engine: Engine) -> None:
        """Should handle SQL syntax errors gracefully."""
        executor = ExecutorAgent(test_engine)
        state = {
            "generated_sql": "SELCT * FORM products",  # Intentional typo
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)

        assert result["error"] is not None
        assert result["result_row_count"] == 0

    def test_handles_nonexistent_table(self, test_engine: Engine) -> None:
        """Should handle references to non-existent tables."""
        executor = ExecutorAgent(test_engine)
        state = {
            "generated_sql": "SELECT * FROM nonexistent_table",
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)

        assert result["error"] is not None

    def test_handles_empty_sql(self, test_engine: Engine) -> None:
        """Should handle empty SQL gracefully."""
        executor = ExecutorAgent(test_engine)
        state = {
            "generated_sql": "",
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)

        assert result["error"] is not None

    def test_row_limit_applied(self, test_engine: Engine) -> None:
        """Should limit results to max_rows."""
        executor = ExecutorAgent(test_engine, max_rows=3)
        state = {
            "generated_sql": "SELECT * FROM products",
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)

        assert result["result_row_count"] <= 3

    def test_test_connection(self, test_engine: Engine) -> None:
        """Should successfully test database connection."""
        executor = ExecutorAgent(test_engine)
        assert executor.test_connection()

    def test_format_error_column_not_found(self) -> None:
        """Error formatter should recognize column errors."""
        msg = ExecutorAgent._format_error(Exception("column 'xyz' not found"))
        assert "column" in msg.lower()

    def test_format_error_timeout(self) -> None:
        """Error formatter should recognize timeout errors."""
        msg = ExecutorAgent._format_error(Exception("statement timeout"))
        assert "timeout" in msg.lower() or "timed out" in msg.lower()

    def test_sets_current_step(self, test_engine: Engine) -> None:
        """Should set current_step in state."""
        executor = ExecutorAgent(test_engine)
        state = {
            "generated_sql": "SELECT 1",
            "guardrail_result": {"is_valid": True},
        }

        result = executor.run(state)
        assert result["current_step"] == "executor"
