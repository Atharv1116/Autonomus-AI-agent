"""
Tests for the Guardrail Agent.

Comprehensive test suite covering all blocked keywords, injection
patterns, edge cases, and valid SELECT queries.
"""

from __future__ import annotations

import pytest

from agents.guardrail import GuardrailAgent


class TestGuardrailBlockedKeywords:
    """Test that all blocked keywords are properly rejected."""

    @pytest.fixture(autouse=True)
    def setup(self, guardrail: GuardrailAgent) -> None:
        self.guardrail = guardrail

    @pytest.mark.parametrize("keyword", [
        "DROP TABLE users",
        "DELETE FROM orders WHERE id = 1",
        "UPDATE products SET price = 0",
        "INSERT INTO users VALUES (1, 'hacker')",
        "ALTER TABLE users ADD COLUMN admin BOOLEAN",
        "TRUNCATE TABLE sales",
        "CREATE TABLE evil (id INT)",
        "GRANT ALL ON users TO hacker",
        "REVOKE SELECT ON products FROM analyst",
        "COPY users TO '/tmp/data.csv'",
        "VACUUM FULL products",
    ])
    def test_blocked_keywords_rejected(self, keyword: str) -> None:
        """Each blocked keyword should be rejected."""
        result = self.guardrail.validate(keyword)
        assert not result["is_valid"], f"Should reject: {keyword}"
        assert len(result["checks_failed"]) > 0

    def test_drop_case_insensitive(self) -> None:
        """Blocked keywords should be case-insensitive."""
        for variant in ["DROP TABLE users", "drop table users", "Drop Table Users", "dRoP tAbLe users"]:
            result = self.guardrail.validate(variant)
            assert not result["is_valid"], f"Should reject: {variant}"

    def test_exec_blocked(self) -> None:
        """EXEC and EXECUTE should be blocked."""
        result = self.guardrail.validate("EXEC sp_executesql N'SELECT 1'")
        assert not result["is_valid"]

    def test_merge_blocked(self) -> None:
        """MERGE should be blocked."""
        result = self.guardrail.validate("MERGE INTO users USING temp ON ...")
        assert not result["is_valid"]


class TestGuardrailValidQueries:
    """Test that valid SELECT queries are accepted."""

    @pytest.fixture(autouse=True)
    def setup(self, guardrail: GuardrailAgent) -> None:
        self.guardrail = guardrail

    def test_simple_select(self) -> None:
        """Simple SELECT should pass."""
        result = self.guardrail.validate("SELECT * FROM products")
        assert result["is_valid"]

    def test_select_with_where(self) -> None:
        """SELECT with WHERE clause should pass."""
        result = self.guardrail.validate(
            "SELECT product_name, price FROM products WHERE price > 100"
        )
        assert result["is_valid"]

    def test_select_with_join(self) -> None:
        """SELECT with JOIN should pass."""
        result = self.guardrail.validate(
            "SELECT p.name, SUM(s.quantity) "
            "FROM products p JOIN sales s ON p.id = s.product_id "
            "GROUP BY p.name"
        )
        assert result["is_valid"]

    def test_select_with_aggregation(self) -> None:
        """SELECT with aggregation functions should pass."""
        result = self.guardrail.validate(
            "SELECT category, COUNT(*), AVG(price), SUM(quantity) "
            "FROM products GROUP BY category HAVING COUNT(*) > 5"
        )
        assert result["is_valid"]

    def test_select_with_subquery(self) -> None:
        """SELECT with a read-only subquery should pass."""
        result = self.guardrail.validate(
            "SELECT * FROM products WHERE category IN "
            "(SELECT category FROM categories WHERE active = 1)"
        )
        assert result["is_valid"]

    def test_select_with_cte(self) -> None:
        """WITH/CTE queries should pass."""
        result = self.guardrail.validate(
            "WITH top_products AS (SELECT product_id, SUM(quantity) AS total "
            "FROM sales GROUP BY product_id) "
            "SELECT p.name, t.total FROM products p JOIN top_products t "
            "ON p.product_id = t.product_id ORDER BY t.total DESC LIMIT 10"
        )
        assert result["is_valid"]

    def test_select_with_window_function(self) -> None:
        """SELECT with window functions should pass."""
        result = self.guardrail.validate(
            "SELECT product_name, price, "
            "RANK() OVER (ORDER BY price DESC) AS price_rank "
            "FROM products"
        )
        assert result["is_valid"]

    def test_select_with_case_when(self) -> None:
        """SELECT with CASE WHEN should pass."""
        result = self.guardrail.validate(
            "SELECT product_name, "
            "CASE WHEN price > 100 THEN 'Expensive' ELSE 'Affordable' END AS tier "
            "FROM products"
        )
        assert result["is_valid"]


class TestGuardrailInjectionPatterns:
    """Test SQL injection detection."""

    @pytest.fixture(autouse=True)
    def setup(self, guardrail: GuardrailAgent) -> None:
        self.guardrail = guardrail

    def test_multiple_statements_semicolon(self) -> None:
        """Multiple statements separated by semicolons should be rejected."""
        result = self.guardrail.validate("SELECT 1; DROP TABLE users")
        assert not result["is_valid"]
        assert "single_statement" in result["checks_failed"] or "no_blocked_keywords" in result["checks_failed"]

    def test_comment_injection(self) -> None:
        """SQL comment injection should be detected."""
        result = self.guardrail.validate("SELECT * FROM users -- AND admin = false")
        assert not result["is_valid"]

    def test_union_injection_system_tables(self) -> None:
        """UNION with system table access should be rejected."""
        result = self.guardrail.validate(
            "SELECT id FROM users UNION SELECT column_name FROM information_schema.columns"
        )
        assert not result["is_valid"]

    def test_sleep_injection(self) -> None:
        """SLEEP function should be blocked."""
        result = self.guardrail.validate("SELECT SLEEP(10)")
        assert not result["is_valid"]

    def test_benchmark_injection(self) -> None:
        """BENCHMARK function should be blocked."""
        result = self.guardrail.validate("SELECT BENCHMARK(1000000, SHA1('test'))")
        assert not result["is_valid"]

    def test_into_outfile(self) -> None:
        """INTO OUTFILE should be blocked."""
        result = self.guardrail.validate("SELECT * FROM users INTO OUTFILE '/tmp/dump.csv'")
        assert not result["is_valid"]

    def test_modifying_subquery(self) -> None:
        """Subqueries with modification statements should be rejected."""
        result = self.guardrail.validate(
            "SELECT * FROM (DELETE FROM users RETURNING *) AS deleted"
        )
        assert not result["is_valid"]


class TestGuardrailEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def setup(self, guardrail: GuardrailAgent) -> None:
        self.guardrail = guardrail

    def test_empty_query(self) -> None:
        """Empty query should be rejected."""
        result = self.guardrail.validate("")
        assert not result["is_valid"]

    def test_whitespace_only(self) -> None:
        """Whitespace-only query should be rejected."""
        result = self.guardrail.validate("   \n\t  ")
        assert not result["is_valid"]

    def test_very_long_query(self) -> None:
        """Extremely long queries should be rejected."""
        long_query = "SELECT " + ", ".join(f"col_{i}" for i in range(2000)) + " FROM big_table"
        result = self.guardrail.validate(long_query)
        assert not result["is_valid"]
        assert "reasonable_length" in result["checks_failed"]

    def test_blocked_word_in_string_literal(self) -> None:
        """Blocked words inside string literals should NOT trigger rejection."""
        result = self.guardrail.validate(
            "SELECT * FROM products WHERE name = 'DROP it like its hot'"
        )
        assert result["is_valid"]

    def test_column_name_containing_blocked_word(self) -> None:
        """Column names containing blocked words should pass if part of larger name."""
        result = self.guardrail.validate(
            "SELECT updated_at, created_at FROM products"
        )
        assert result["is_valid"]

    def test_run_updates_state(self) -> None:
        """The run method should properly update AgentState."""
        state = {
            "generated_sql": "SELECT * FROM products",
            "guardrail_result": {},
        }
        updated = self.guardrail.run(state)
        assert updated["guardrail_result"]["is_valid"]
        assert updated["current_step"] == "guardrail"

    def test_run_empty_sql(self) -> None:
        """Run with empty SQL should fail gracefully."""
        state = {"generated_sql": "", "guardrail_result": {}}
        updated = self.guardrail.run(state)
        assert not updated["guardrail_result"]["is_valid"]

    def test_error_suggestion(self) -> None:
        """Error suggestions should provide helpful messages."""
        msg = GuardrailAgent.get_error_suggestion("Query must start with SELECT")
        assert "SELECT" in msg
        assert len(msg) > 20
