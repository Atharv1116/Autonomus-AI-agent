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
        schema = state.get("schema_info", "")
        dialect = state.get("database_dialect", "postgresql")
        question = state.get("user_question", "")
        error_context = state.get("error", "")
        retry_count = state.get("retry_count", 0)

        logger.info(
            "Generating SQL (dialect=%s, retry=%d): '%s'",
            dialect, retry_count, question[:80],
        )

        # Build prompt
        prompt = self._prompt_template.format(
            schema=schema,
            dialect=dialect,
            plan=json.dumps(plan, indent=2),
            few_shot_examples=self._few_shot_examples,
            question=question,
            max_rows=self._max_rows,
            error_context=error_context if retry_count > 0 else "No previous errors.",
        )

        # Call LLM
        messages = [
            SystemMessage(
                content="You are an expert SQL engineer. Generate ONLY the SQL query. "
                "No markdown, no code blocks, no explanations. Just pure SQL."
            ),
            HumanMessage(content=prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            sql = self._clean_sql(response.content)

            # Validate basic structure
            if not sql.strip().upper().startswith("SELECT"):
                logger.warning("Generated SQL doesn't start with SELECT: %s", sql[:100])
                # Try to extract SELECT statement
                select_match = re.search(r'(SELECT\s+[\s\S]+)', sql, re.IGNORECASE)
                if select_match:
                    sql = select_match.group(1)
                else:
                    raise ValueError("Generated output is not a valid SELECT query")

            logger.info("SQL generated: %s", sql[:200])
            state["generated_sql"] = sql
            state["current_step"] = "sql_generator"
            state["error"] = None

        except Exception as e:
            logger.exception("SQL generation failed")
            state["error"] = f"SQL generation failed: {str(e)}"
            state["generated_sql"] = ""

        return state

    @staticmethod
    def _clean_sql(raw: str) -> str:
        """
        Clean the LLM output to extract pure SQL.

        Removes markdown code blocks, trailing semicolons, and
        extra whitespace.

        Args:
            raw: Raw LLM response.

        Returns:
            Cleaned SQL string.
        """
        sql = raw.strip()

        # Remove markdown code blocks
        if sql.startswith("```"):
            lines = sql.split("\n")
            # Remove first line (```sql or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            sql = "\n".join(lines).strip()

        # Remove trailing semicolons
        sql = sql.rstrip(";").strip()

        # Remove any leading/trailing quotes
        if (sql.startswith('"') and sql.endswith('"')) or \
           (sql.startswith("'") and sql.endswith("'")):
            sql = sql[1:-1].strip()

        # Normalize whitespace (collapse multiple spaces/newlines)
        sql = re.sub(r'\s+', ' ', sql).strip()

        return sql
