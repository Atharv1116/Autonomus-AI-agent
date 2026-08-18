"""
SQL Guardrail Agent.

Validates every generated SQL query to ensure it is read-only (SELECT).
Blocks all destructive operations, injection attempts, and non-standard
SQL patterns. Pure Python implementation — no LLM needed.
"""

from __future__ import annotations

import re
from typing import Any

from agents.state import AgentState
from config.logging_config import get_logger

logger = get_logger("agents.guardrail")


class GuardrailAgent:
    """
    Validates SQL queries for safety before execution.

    Uses regex and pattern matching to detect and block:
    - DDL operations (CREATE, DROP, ALTER, TRUNCATE)
    - DML operations (INSERT, UPDATE, DELETE)
    - Permission operations (GRANT, REVOKE)
    - System operations (COPY, VACUUM)
    - SQL injection patterns (UNION injection, multiple statements)
    - Non-SELECT queries
    """

    # Blocked SQL keywords (case-insensitive)
    BLOCKED_KEYWORDS: list[str] = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
        "COPY",
        "VACUUM",
        "EXEC",
        "EXECUTE",
        "CALL",
        "MERGE",
        "REPLACE",
        "UPSERT",
        "LOAD",
        "IMPORT",
        "EXPORT",
    ]

    # Patterns that indicate injection or dangerous constructs
    DANGEROUS_PATTERNS: list[tuple[str, str]] = [
        (r";\s*\w", "Multiple SQL statements detected (semicolon followed by keyword)"),
        (r"--\s*\w", "SQL comment injection detected"),
        (r"/\*.*?\*/", "Block comment detected"),
        (r"xp_\w+", "Extended stored procedure call detected"),
        (r"sp_\w+", "Stored procedure call detected"),
        (r"INTO\s+OUTFILE", "INTO OUTFILE clause detected"),
        (r"INTO\s+DUMPFILE", "INTO DUMPFILE clause detected"),
        (r"LOAD_FILE\s*\(", "LOAD_FILE function detected"),
        (r"BENCHMARK\s*\(", "BENCHMARK function detected"),
        (r"SLEEP\s*\(", "SLEEP function detected"),
        (r"WAITFOR\s+DELAY", "WAITFOR DELAY detected"),
        (r"INFORMATION_SCHEMA\.", "Direct INFORMATION_SCHEMA access detected"),
        (r"pg_\w+\s*\(", "PostgreSQL system function detected"),
        (r"sys\.\w+", "System table access detected"),
    ]

    def __init__(self) -> None:
        """Initialize the Guardrail Agent."""
        # Pre-compile regex patterns for performance
        self._blocked_patterns: list[re.Pattern] = [
            re.compile(rf"\b{kw}\b", re.IGNORECASE)
            for kw in self.BLOCKED_KEYWORDS
        ]
        self._dangerous_patterns: list[tuple[re.Pattern, str]] = [
            (re.compile(pattern, re.IGNORECASE | re.DOTALL), message)
            for pattern, message in self.DANGEROUS_PATTERNS
        ]
        logger.info("GuardrailAgent initialized with %d blocked keywords", len(self.BLOCKED_KEYWORDS))

    def run(self, state: AgentState) -> AgentState:
        """
        Validate the generated SQL query.

        Args:
            state: Current workflow state with 'generated_sql'.

        Returns:
            Updated state with 'guardrail_result'.
        """
        sql = state.get("generated_sql", "")

        if not sql or not sql.strip():
            state["guardrail_result"] = {
                "is_valid": False,
                "reason": "Empty SQL query received",
                "checks_passed": [],
                "checks_failed": ["not_empty"],
            }
            state["current_step"] = "guardrail"
            return state

        result = self.validate(sql)
        state["guardrail_result"] = result
        state["current_step"] = "guardrail"

        if result["is_valid"]:
            logger.info("SQL validation PASSED: %d checks passed", len(result["checks_passed"]))
        else:
            logger.warning("SQL validation FAILED: %s", result["reason"])
            state["error"] = f"SQL Guardrail: {result['reason']}"

        return state

    def validate(self, sql: str) -> dict[str, Any]:
        """
        Run all validation checks on a SQL query.

        Args:
            sql: The SQL query to validate.

        Returns:
            Validation result dict with is_valid, reason, and check details.
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        reasons: list[str] = []

        sql_normalized = sql.strip()

        # ========================================
        # Check 1: Must start with SELECT
        # ========================================
        if not self._is_select_query(sql_normalized):
            checks_failed.append("is_select")
            reasons.append("Query must start with SELECT. Only read-only queries are allowed.")
        else:
            checks_passed.append("is_select")

        # ========================================
        # Check 2: No blocked keywords
        # ========================================
        blocked = self._check_blocked_keywords(sql_normalized)
        if blocked:
            checks_failed.append("no_blocked_keywords")
            reasons.append(f"Blocked keyword(s) detected: {', '.join(blocked)}. Only SELECT queries are allowed.")
        else:
            checks_passed.append("no_blocked_keywords")

        # ========================================
        # Check 3: No multiple statements
        # ========================================
        if self._has_multiple_statements(sql_normalized):
            checks_failed.append("single_statement")
            reasons.append("Multiple SQL statements detected. Only a single SELECT query is allowed.")
        else:
            checks_passed.append("single_statement")

        # ========================================
        # Check 4: No dangerous patterns
        # ========================================
        danger = self._check_dangerous_patterns(sql_normalized)
        if danger:
            checks_failed.append("no_dangerous_patterns")
            reasons.append(f"Dangerous pattern detected: {danger}")
        else:
            checks_passed.append("no_dangerous_patterns")

        # ========================================
        # Check 5: No UNION injection attempts
        # ========================================
        if self._has_union_injection(sql_normalized):
            checks_failed.append("no_union_injection")
            reasons.append(
                "Suspicious UNION usage detected. UNION with different column "
                "patterns may indicate injection."
            )
        else:
            checks_passed.append("no_union_injection")

        # ========================================
        # Check 6: No modifying subqueries
        # ========================================
        if self._has_modifying_subquery(sql_normalized):
            checks_failed.append("no_modifying_subqueries")
            reasons.append("Subquery contains data modification statements.")
        else:
            checks_passed.append("no_modifying_subqueries")

        # ========================================
        # Check 7: Reasonable query length
        # ========================================
        if len(sql_normalized) > 10000:
            checks_failed.append("reasonable_length")
            reasons.append("Query exceeds maximum allowed length (10,000 characters).")
        else:
            checks_passed.append("reasonable_length")

        is_valid = len(checks_failed) == 0
        reason = "; ".join(reasons) if reasons else "All checks passed"

        return {
            "is_valid": is_valid,
            "reason": reason,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "query_preview": sql_normalized[:200],
        }

    @staticmethod
    def _is_select_query(sql: str) -> bool:
        """Check if the query starts with SELECT (allowing WITH/CTE)."""
        upper = sql.upper().lstrip()
        return upper.startswith("SELECT") or upper.startswith("WITH")

    def _check_blocked_keywords(self, sql: str) -> list[str]:
        """
        Check for blocked SQL keywords.

        Returns list of blocked keywords found. Handles the case where
        blocked words appear inside string literals or column names.
        """
        found: list[str] = []

        # Remove string literals to avoid false positives
        cleaned = re.sub(r"'[^']*'", "''", sql)
        cleaned = re.sub(r'"[^"]*"', '""', cleaned)

        for keyword, pattern in zip(self.BLOCKED_KEYWORDS, self._blocked_patterns):
            if pattern.search(cleaned):
                # Double-check: ensure it's not part of a column/table name
                # by checking for standalone keyword usage
                standalone = re.compile(
                    rf"(?<![.\w]){keyword}(?![.\w])",
                    re.IGNORECASE,
                )
                if standalone.search(cleaned):
                    found.append(keyword)

        return found

    @staticmethod
    def _has_multiple_statements(sql: str) -> bool:
        """Check for multiple SQL statements separated by semicolons."""
        # Remove string literals
        cleaned = re.sub(r"'[^']*'", "''", sql)
        cleaned = re.sub(r'"[^"]*"', '""', cleaned)

        # Remove trailing semicolons/whitespace
        cleaned = cleaned.rstrip().rstrip(";").rstrip()

        # Check for remaining semicolons
        return ";" in cleaned

    def _check_dangerous_patterns(self, sql: str) -> str | None:
        """
        Check for dangerous SQL patterns.

        Returns the warning message if found, None otherwise.
        """
        for pattern, message in self._dangerous_patterns:
            if pattern.search(sql):
                return message
        return None

    @staticmethod
    def _has_union_injection(sql: str) -> bool:
        """
        Detect suspicious UNION-based injection attempts.

        Legitimate UNION queries are rare in analyst queries.
        Flag UNION combined with suspicious patterns.
        """
        if not re.search(r"\bUNION\b", sql, re.IGNORECASE):
            return False

        # Check for UNION + SELECT with system table access
        suspicious_patterns = [
            r"UNION\s+(ALL\s+)?SELECT\s+.*\bFROM\s+information_schema",
            r"UNION\s+(ALL\s+)?SELECT\s+.*\bFROM\s+pg_",
            r"UNION\s+(ALL\s+)?SELECT\s+.*\bFROM\s+sys\.",
            r"UNION\s+(ALL\s+)?SELECT\s+(NULL|1|2|3|')",
        ]

        for pat in suspicious_patterns:
            if re.search(pat, sql, re.IGNORECASE | re.DOTALL):
                return True

        return False

    @staticmethod
    def _has_modifying_subquery(sql: str) -> bool:
        """Check if any subquery contains modification statements."""
        # Extract subqueries (content within parentheses)
        subqueries = re.findall(r'\(([^()]+)\)', sql)
        modifying_keywords = re.compile(
            r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE)\b',
            re.IGNORECASE,
        )
        for subquery in subqueries:
            if modifying_keywords.search(subquery):
                return True
        return False

    @staticmethod
    def get_error_suggestion(reason: str) -> str:
        """
        Generate a helpful error message for the user.

        Args:
            reason: The validation failure reason.

        Returns:
            User-friendly error suggestion.
        """
        suggestions = {
            "SELECT": (
                "Your query must be a SELECT statement. I can only run "
                "read-only queries to protect your data."
            ),
            "Blocked keyword": (
                "Your query contains a restricted operation. I can only "
                "run SELECT queries for data analysis — no modifications."
            ),
            "Multiple": (
                "Please provide a single query. Running multiple SQL "
                "statements at once is not allowed for safety."
            ),
            "injection": (
                "The query contains patterns that look like an injection "
                "attempt. Please rephrase your question."
            ),
        }

        for key, suggestion in suggestions.items():
            if key.lower() in reason.lower():
                return suggestion

        return (
            "The query didn't pass safety validation. Please try "
            "rephrasing your question in simpler terms."
        )
