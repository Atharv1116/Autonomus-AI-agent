"""
Shared agent state definition.

Defines the TypedDict that flows through the LangGraph workflow,
carrying data between all six agent nodes.
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed through the LangGraph workflow.

    Each agent reads from and writes to this state dictionary.
    Fields are added progressively as data flows through the pipeline.
    """

    # --- Input ---
    user_question: str
    """The user's original natural language question."""

    conversation_history: list[dict[str, str]]
    """Previous conversation messages for context."""

    schema_info: str
    """LLM-formatted database schema string."""

    database_dialect: str
    """Database dialect name (postgresql, mysql, sqlite)."""

    # --- Planner Output ---
    plan: dict[str, Any]
    """Structured analysis plan from the Planner Agent."""

    # --- SQL Generator Output ---
    generated_sql: str
    """The generated SQL query string."""

    # --- Guardrail Output ---
    guardrail_result: dict[str, Any]
    """Validation result: {is_valid: bool, reason: str, checks_passed: list}."""

    # --- Executor Output ---
    query_results: Any
    """Query results as a Pandas DataFrame (serialized)."""

    result_columns: list[str]
    """Column names from the query results."""

    result_row_count: int
    """Number of rows returned."""

    # --- Visualization Output ---
    visualization: Optional[dict[str, Any]]
    """Plotly figure as JSON-serializable dict."""

    chart_config: Optional[dict[str, Any]]
    """Chart configuration metadata (type, title, axes)."""

    # --- Insight Output ---
    insights: str
    """Markdown-formatted executive insights."""

    # --- Control Flow ---
    error: Optional[str]
    """Error message if any step fails."""

    retry_count: int
    """Number of SQL generation retries attempted."""

    max_retries: int
    """Maximum allowed retries."""

    current_step: str
    """Name of the currently executing step."""

    # --- Metrics ---
    execution_times: dict[str, float]
    """Per-step execution times in seconds."""

    token_usage: dict[str, Any]
    """Per-step token usage breakdown."""

    # --- Metadata ---
    metadata: dict[str, Any]
    """Additional metadata (timestamps, versions, etc.)."""
