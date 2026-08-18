"""
Planner Agent.

Understands the user's natural language question, analyzes the database
schema, and produces a structured analysis plan that guides SQL generation.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from config.logging_config import get_logger

logger = get_logger("agents.planner")

# Load prompt template
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "planner.txt")


class PlannerAgent:
    """
    Understands user intent and creates a structured analysis plan.

    The plan includes required tables, columns, joins, filters,
    aggregations, and sorting — everything the SQL Generator needs.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """
        Initialize the Planner Agent.

        Args:
            llm: LangChain chat model instance.
        """
        self._llm = llm
        self._prompt_template = self._load_prompt()
        logger.info("PlannerAgent initialized")

    def _load_prompt(self) -> str:
        """Load the planner prompt template from file."""
        try:
            with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Planner prompt not found at %s, using default", _PROMPT_PATH)
            return self._default_prompt()

    @staticmethod
    def _default_prompt() -> str:
        """Fallback prompt if file is missing."""
        return (
            "You are an expert Data Analysis Planner. Given a database schema "
            "and a user question, create a structured JSON analysis plan with: "
            "understanding, analysis_type, tables_needed, columns_needed, "
            "joins, filters, aggregations, group_by, order_by, limit, notes.\n\n"
            "Schema:\n{schema}\n\nQuestion:\n{question}"
        )

    def run(self, state: AgentState) -> AgentState:
        """
        Execute the planner agent.

        Args:
            state: Current workflow state with user_question and schema_info.

        Returns:
            Updated state with 'plan' populated.
        """
        question = state.get("user_question", "")
        schema = state.get("schema_info", "")
        history = state.get("conversation_history", [])

        logger.info("Planning analysis for: '%s'", question[:100])

        # Format conversation history
        history_str = ""
        if history:
            history_str = "\n".join(
                f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
                for msg in history[-5:]  # Last 5 messages for context
            )

        # Build prompt
        prompt = self._prompt_template.format(
            schema=schema,
            question=question,
            conversation_history=history_str or "No previous conversation.",
        )

        # Call LLM
        messages = [
            SystemMessage(content="You are a data analysis planning expert. Always respond with valid JSON."),
            HumanMessage(content=prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            plan = self._parse_plan(response.content)

            logger.info(
                "Plan created: type=%s, tables=%s",
                plan.get("analysis_type", "unknown"),
                plan.get("tables_needed", []),
            )

            state["plan"] = plan
            state["current_step"] = "planner"
            state["error"] = None

        except Exception as e:
            logger.exception("Planner failed")
            state["error"] = f"Planning failed: {str(e)}"
            state["plan"] = self._fallback_plan(question, schema)

        return state

    def _parse_plan(self, content: str) -> dict[str, Any]:
        """
        Parse the LLM response into a structured plan.

        Handles various response formats (raw JSON, markdown code blocks).

        Args:
            content: Raw LLM response string.

        Returns:
            Parsed plan dictionary.
        """
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (code block markers)
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            content = content.strip()

        try:
            plan = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed content
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    plan = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse plan JSON, creating minimal plan")
                    plan = {"understanding": content, "analysis_type": "detail"}
            else:
                plan = {"understanding": content, "analysis_type": "detail"}

        # Ensure required fields have defaults
        plan.setdefault("understanding", "")
        plan.setdefault("analysis_type", "detail")
        plan.setdefault("tables_needed", [])
        plan.setdefault("columns_needed", {})
        plan.setdefault("joins", [])
        plan.setdefault("filters", [])
        plan.setdefault("aggregations", [])
        plan.setdefault("group_by", [])
        plan.setdefault("order_by", [])
        plan.setdefault("limit", None)
        plan.setdefault("notes", "")

        return plan

    @staticmethod
    def _fallback_plan(question: str, schema: str) -> dict[str, Any]:
        """
        Create a minimal fallback plan when LLM parsing fails.

        Args:
            question: Original user question.
            schema: Database schema string.

        Returns:
            Basic plan dictionary.
        """
        return {
            "understanding": question,
            "analysis_type": "detail",
            "tables_needed": [],
            "columns_needed": {},
            "joins": [],
            "filters": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": 100,
            "notes": "Fallback plan — LLM response could not be parsed.",
        }
