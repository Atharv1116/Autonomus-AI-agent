"""
Insight Agent.

Generates executive-level analysis summaries from query results.
Covers key findings, business implications, recommendations,
anomalies, patterns, and future actions.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from config.logging_config import get_logger

logger = get_logger("agents.insights")

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "insight.txt")


class InsightAgent:
    """
    Generates executive-level data insights from query results.

    Produces markdown-formatted analysis covering key findings,
    business implications, trends, anomalies, and recommendations.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """
        Initialize the Insight Agent.

        Args:
            llm: LangChain chat model instance.
        """
        self._llm = llm
        self._prompt_template = self._load_prompt()
        logger.info("InsightAgent initialized")

    def _load_prompt(self) -> str:
        """Load the insight prompt template."""
        try:
            with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("Insight prompt not found, using default")
            return (
                "Analyze the following query results and provide executive insights.\n\n"
                "Question: {question}\nSQL: {sql}\nResults:\n{results}\n"
                "Rows: {row_count}, Columns: {columns}\n\n"
                "Cover: Key Findings, Business Implications, Trends, Anomalies, Recommendations."
            )

    def run(self, state: AgentState) -> AgentState:
        """
        Generate executive insights from query results.

        Args:
            state: Current workflow state with query_results.

        Returns:
            Updated state with 'insights' populated.
        """
        results = state.get("query_results", [])
        columns = state.get("result_columns", [])
        row_count = state.get("result_row_count", 0)
        question = state.get("user_question", "")
        sql = state.get("generated_sql", "")

        if not results or row_count == 0:
            state["insights"] = self._empty_results_insight(question)
            state["current_step"] = "insights"
            return state

        logger.info("Generating insights for %d rows", row_count)

        # Prepare data summary for the LLM
        df = pd.DataFrame(results)
        results_summary = self._prepare_results_summary(df)

        # Build prompt
        prompt = self._prompt_template.format(
            question=question,
            sql=sql,
            results=results_summary,
            row_count=row_count,
            columns=", ".join(columns),
        )

        # Call LLM
        messages = [
            SystemMessage(
                content=(
                    "You are a Senior Data Analyst providing executive insights. "
                    "Be specific with numbers. Be concise and actionable. "
                    "Use professional business language with markdown formatting."
                )
            ),
            HumanMessage(content=prompt),
        ]

        try:
            response = self._llm.invoke(messages)
            insights = response.content.strip()

            # Ensure insights are not empty
            if not insights:
                insights = self._fallback_insights(df, question)

            state["insights"] = insights
            state["current_step"] = "insights"
            state["error"] = None

            logger.info("Insights generated: %d characters", len(insights))

        except Exception as e:
            logger.exception("Insight generation failed")
            state["insights"] = self._fallback_insights(df, question)
            state["current_step"] = "insights"

        return state

    def _prepare_results_summary(self, df: pd.DataFrame, max_rows: int = 50) -> str:
        """
        Prepare a concise summary of the results for the LLM.

        For large datasets, includes statistical summary + sample rows.

        Args:
            df: Query results DataFrame.
            max_rows: Maximum rows to include in the summary.

        Returns:
            Formatted results string.
        """
        lines: list[str] = []

        if len(df) <= max_rows:
            lines.append(df.to_string(index=False))
        else:
            # For large datasets, provide stats + sample
            lines.append(f"Dataset: {len(df)} rows × {len(df.columns)} columns")
            lines.append("\n--- Statistical Summary ---")
            lines.append(df.describe().to_string())
            lines.append(f"\n--- First {max_rows} rows ---")
            lines.append(df.head(max_rows).to_string(index=False))

        # Add column type info
        lines.append("\n--- Column Types ---")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            lines.append(f"  {col}: {dtype} ({unique_count} unique, {null_count} nulls)")

        return "\n".join(lines)

    @staticmethod
    def _empty_results_insight(question: str) -> str:
        """Generate insight message when no results are returned."""
        return (
            "### 📊 Key Findings\n\n"
            f"The query for *\"{question}\"* returned no results.\n\n"
            "### 💼 Business Implications\n\n"
            "- No data matches the specified criteria\n"
            "- This could indicate a gap in data collection or filtering\n\n"
            "### ✅ Recommendations\n\n"
            "- Try broadening the search criteria\n"
            "- Check if the relevant data exists in the database\n"
            "- Verify date ranges or filter conditions\n"
        )

    @staticmethod
    def _fallback_insights(df: pd.DataFrame, question: str) -> str:
        """
        Generate basic statistical insights when LLM fails.

        Args:
            df: Query results DataFrame.
            question: Original user question.

        Returns:
            Markdown-formatted basic insights.
        """
        lines = [
            "### 📊 Key Findings\n",
            f"Query results for *\"{question}\"*:\n",
            f"- **Total Records**: {len(df):,}",
            f"- **Columns**: {len(df.columns)}",
        ]

        # Add numeric column stats
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            lines.append("\n**Numeric Summary:**\n")
            for col in numeric_cols[:5]:  # Limit to first 5
                lines.append(
                    f"- **{col}**: min={df[col].min():,.2f}, "
                    f"max={df[col].max():,.2f}, "
                    f"avg={df[col].mean():,.2f}"
                )

        # Add top categorical values
        cat_cols = df.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            lines.append("\n**Top Values:**\n")
            for col in cat_cols[:3]:
                top = df[col].value_counts().head(3)
                top_str = ", ".join(f"{k} ({v})" for k, v in top.items())
                lines.append(f"- **{col}**: {top_str}")

        lines.append("\n### ✅ Recommendations\n")
        lines.append("- Review the detailed data table for specific insights")
        lines.append("- Consider filtering by specific dimensions for deeper analysis")

        return "\n".join(lines)
