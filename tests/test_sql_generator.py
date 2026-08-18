"""
Tests for the SQL Generator Agent.

Validates SQL output format, cleaning, dialect handling,
and error recovery.
"""

from __future__ import annotations

import json

import pytest

from agents.sql_generator import SQLGeneratorAgent
from tests.conftest import MockLLM


class TestSQLGeneratorAgent:
    """Test the SQL Generator Agent."""

    def test_generates_select_query(self, mock_llm: MockLLM) -> None:
        """Should generate a SELECT query."""
        mock_llm.set_response("SELECT p.product_name, SUM(s.quantity) AS total_sold FROM products p JOIN sales s ON p.product_id = s.product_id GROUP BY p.product_name ORDER BY total_sold DESC LIMIT 10")

        generator = SQLGeneratorAgent(mock_llm)
        state = {
            "plan": {"understanding": "Top products", "tables_needed": ["products", "sales"]},
            "schema_info": "TABLE: products (product_id, product_name)",
            "database_dialect": "postgresql",
            "user_question": "Top 10 products",
            "error": "",
            "retry_count": 0,
        }

        result = generator.run(state)
        assert result["generated_sql"].upper().startswith("SELECT")
        assert "product_name" in result["generated_sql"]

    def test_cleans_markdown_code_blocks(self, mock_llm: MockLLM) -> None:
        """Should strip markdown code blocks from response."""
        mock_llm.set_response("```sql\nSELECT * FROM products\n```")

        generator = SQLGeneratorAgent(mock_llm)
        state = {
            "plan": {},
            "schema_info": "",
            "database_dialect": "postgresql",
            "user_question": "Show products",
            "error": "",
            "retry_count": 0,
        }

        result = generator.run(state)
        assert not result["generated_sql"].startswith("```")
        assert "SELECT" in result["generated_sql"].upper()

    def test_removes_trailing_semicolon(self, mock_llm: MockLLM) -> None:
        """Should remove trailing semicolons."""
        mock_llm.set_response("SELECT * FROM products;")

        generator = SQLGeneratorAgent(mock_llm)
        state = {
            "plan": {},
            "schema_info": "",
            "database_dialect": "postgresql",
            "user_question": "Show products",
            "error": "",
            "retry_count": 0,
        }

        result = generator.run(state)
        assert not result["generated_sql"].endswith(";")

    def test_sets_current_step(self, mock_llm: MockLLM) -> None:
        """Should set current_step in state."""
        mock_llm.set_response("SELECT 1")

        generator = SQLGeneratorAgent(mock_llm)
        state = {
            "plan": {},
            "schema_info": "",
            "database_dialect": "postgresql",
            "user_question": "Test",
            "error": "",
            "retry_count": 0,
        }

        result = generator.run(state)
        assert result["current_step"] == "sql_generator"

    def test_handles_llm_error(self, mock_llm: MockLLM) -> None:
        """Should handle LLM errors gracefully."""
        class FailingLLM:
            def invoke(self, messages, **kwargs):
                raise Exception("API timeout")

        generator = SQLGeneratorAgent(FailingLLM())
        state = {
            "plan": {},
            "schema_info": "",
            "database_dialect": "postgresql",
            "user_question": "Test",
            "error": "",
            "retry_count": 0,
        }

        result = generator.run(state)
        assert result["error"] is not None
        assert result["generated_sql"] == ""

    def test_extracts_select_from_verbose_response(self, mock_llm: MockLLM) -> None:
        """Should extract SELECT statement from verbose LLM response."""
        mock_llm.set_response(
            "Here is the SQL query you need:\n\n"
            "SELECT product_name, price FROM products WHERE price > 100"
        )

        generator = SQLGeneratorAgent(mock_llm)
        state = {
            "plan": {},
            "schema_info": "",
            "database_dialect": "postgresql",
            "user_question": "Expensive products",
            "error": "",
            "retry_count": 0,
        }

        result = generator.run(state)
        assert "SELECT" in result["generated_sql"].upper()

    def test_clean_sql_static_method(self) -> None:
        """Test the static _clean_sql method directly."""
        # Code block with sql language tag
        assert "SELECT 1" in SQLGeneratorAgent._clean_sql("```sql\nSELECT 1\n```")

        # Trailing semicolon
        assert not SQLGeneratorAgent._clean_sql("SELECT 1;").endswith(";")

        # Quoted SQL
        assert "SELECT 1" in SQLGeneratorAgent._clean_sql('"SELECT 1"')

        # Multiple spaces
        cleaned = SQLGeneratorAgent._clean_sql("SELECT   *   FROM   products")
        assert "  " not in cleaned
