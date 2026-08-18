"""
SQL Generator Agent.

Converts structured analysis plans into optimized, dialect-specific
SQL queries. Uses few-shot examples and schema context for accuracy.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from config.logging_config import get_logger
from utils.prompt_budget import (
    compress_plan,
    filter_schema_for_tables,
    fits_in_budget,
    MAX_PROMPT_CHARS,
)

logger = get_logger("agents.sql_generator")

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "sql.txt")
_FEW_SHOT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "few_shot_examples.json")


class SQLGeneratorAgent:
    """
    Generates SQL queries from structured analysis plans.

    Uses few-shot examples, schema context, and dialect-specific
    rules to produce valid, optimized SELECT queries.
    """

    def __init__(self, llm: BaseChatModel, max_rows: int = 10000) -> None:
        """
        Initialize the SQL Generator Agent.

        Args:
            llm: LangChain chat model instance.
            max_rows: Maximum rows to return (added as LIMIT).
        """
        self._llm = llm
        self._max_rows = max_rows
        self._prompt_template = self._load_prompt()
        self._few_shot_examples = self._load_few_shot_examples()
        logger.info("SQLGeneratorAgent initialized (max_rows=%d)", max_rows)

    def _load_prompt(self) -> str:
        """Load the SQL prompt template."""
        try:
            with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("SQL prompt not found, using default")
            return (
                "Generate a SQL SELECT query for the following plan.\n"
                "Schema:\n{schema}\nDialect: {dialect}\nPlan:\n{plan}\n"
                "Question:\n{question}\nOutput only the SQL query."
            )

    def _load_few_shot_examples(self) -> str:
        """Load few-shot SQL examples."""
        try:
            with open(_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                examples = data.get("examples", [])
                lines = []
                for ex in examples[:2]:  # Use top 2 examples
                    lines.append(f"Question: {ex['question']}")
                    lines.append(f"SQL: {ex['sql']}")
                    lines.append("")
                return "\n".join(lines)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Few-shot examples not found, proceeding without them")
            return "No examples available."

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the SQL generator agent.

        Args:
            state: Current workflow state with plan and schema_info.

        Returns:
            Updated state with 'generated_sql' populated.
        """
        plan = state.get("plan", {})
        full_schema = state.get("schema_info", "")
        dialect = state.get("database_dialect", "postgresql")
        question = state.get("user_question", "")
        error_context = state.get("error", "")
        retry_count = state.get("retry_count", 0)

        logger.info(
            "Generating SQL (dialect=%s, retry=%d): '%s'",
            dialect, retry_count, question[:80],
        )

        # ── Budget-aware prompt construction ─────────────────────────────
        # 1. Filter schema to only the tables the planner identified
        tables_needed = plan.get("tables_needed", [])
        schema = filter_schema_for_tables(full_schema, tables_needed)
        logger.debug(
            "Schema filtered: %d -> %d chars (tables=%s)",
            len(full_schema), len(schema), tables_needed,
        )

        # 2. Compress plan JSON (drop verbose fields, no indentation)
        plan_str = compress_plan(plan)

        # 3. Build prompt with examples first; drop them if over budget
        error_ctx = (error_context or "")[:300] if retry_count > 0 else "No previous errors."

        def _build_prompt(include_examples: bool) -> str:
            return self._prompt_template.format(
                schema=schema,
                dialect=dialect,
                plan=plan_str,
                few_shot_examples=self._few_shot_examples if include_examples else "",
                question=question,
                max_rows=self._max_rows,
                error_context=error_ctx,
            )

        system_msg = (
            "You are an expert SQL engineer. Your ONLY output must be a raw SQL SELECT query.\n"
            "STRICT RULES:\n"
            "- Output ONLY the SQL query. Nothing else.\n"
            "- NO markdown (no ```, no **bold**, no #heading).\n"
            "- NO explanations, comments, or result tables.\n"
            "- NO leading/trailing text of any kind.\n"
            "- NO semicolon at the end.\n"
            "- Start your response directly with SELECT (or WITH for CTEs)."
        )

        prompt = _build_prompt(include_examples=True)
        if not fits_in_budget(prompt, system_msg):
            logger.warning(
                "Prompt too large (%d chars) — dropping few-shot examples", len(prompt)
            )
            prompt = _build_prompt(include_examples=False)

        logger.debug("Final prompt size: %d chars", len(prompt))

        # Call LLM with a very strict system prompt
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            raw = response.content
            sql = self._extract_sql(raw)

            if not sql:
                logger.warning("Could not extract valid SQL from response: %s", raw[:300])
                raise ValueError(
                    f"Model returned non-SQL output. Response preview: {raw[:200]}"
                )

            logger.info("SQL generated (%d chars): %s", len(sql), sql[:200])
            state["generated_sql"] = sql
            state["current_step"] = "sql_generator"
            state["error"] = None

        except Exception as e:
            error_msg = str(e)
            logger.exception("SQL generation failed")
            state["error"] = f"SQL generation failed: {error_msg}"
            state["generated_sql"] = ""

        return state

    @classmethod
    def _extract_sql(cls, raw: str) -> str:
        """
        Robustly extract a pure SQL SELECT query from any LLM response.

        Handles all common LLM output patterns including:
        - Clean SQL responses
        - Markdown code blocks (```sql ... ``` or ``` ... ```)
        - SQL followed by markdown explanations / result tables
        - SQL preceded by introductory text
        - Mixed content with backticks, headings, pipes (| table rows)

        Args:
            raw: Raw LLM response string.

        Returns:
            Cleaned SQL string, or empty string if none found.
        """
        text = raw.strip()

        # ── Strategy 1: extract content from first ```...``` block ────────────
        code_block = re.search(
            r'```(?:sql|SQL)?\s*\n?([\s\S]*?)```',
            text,
        )
        if code_block:
            candidate = code_block.group(1).strip()
            sql = cls._isolate_select(candidate)
            if sql:
                return cls._normalise(sql)

        # ── Strategy 2: find the first SELECT/WITH keyword in the whole text ──
        sql = cls._isolate_select(text)
        if sql:
            return cls._normalise(sql)

        return ""

    @staticmethod
    def _isolate_select(text: str) -> str:
        """
        Given a block of text, extract the SQL SELECT statement and discard
        everything that follows the query (markdown tables, explanations, etc.).

        Returns cleaned SQL or empty string.
        """
        # Find the start of SELECT or WITH (CTE)
        match = re.search(r'(?i)\b(SELECT|WITH)\b', text)
        if not match:
            return ""

        sql = text[match.start():]

        # Truncate at the first line that looks like non-SQL content:
        # markdown headings (#), pipe-table rows (|), triple-backtick fences,
        # or lines starting with natural language words after a blank line.
        cutoff_pattern = re.compile(
            r'^(?:'           # start of line
            r'\s*```'         # closing code fence
            r'|\s*#'          # markdown heading
            r'|\s*\|'         # pipe-table row
            r'|\s*[-*]{3,}'   # horizontal rule
            r'|\s*Note[:\s]'  # note/explanation prefix
            r'|\s*This query' # explanation prefix
            r'|\s*The (?:query|above|result|SQL)'  # explanation
            r')',
            re.IGNORECASE | re.MULTILINE,
        )

        cutoff_match = cutoff_pattern.search(sql)
        if cutoff_match:
            sql = sql[:cutoff_match.start()]

        # Strip trailing semicolons and whitespace
        sql = sql.rstrip().rstrip(';').rstrip()

        # Remove any surrounding single/double quotes
        if len(sql) >= 2 and sql[0] in ('"', "'") and sql[-1] == sql[0]:
            sql = sql[1:-1].strip()

        return sql.strip()

    @staticmethod
    def _normalise(sql: str) -> str:
        """Collapse internal whitespace without destroying the query structure."""
        # Collapse runs of spaces/tabs to a single space,
        # but preserve newlines so multi-line queries stay readable.
        sql = re.sub(r'[^\S\n]+', ' ', sql)   # collapse horizontal whitespace
        sql = re.sub(r'\n{2,}', '\n', sql)    # collapse blank lines
        return sql.strip()
