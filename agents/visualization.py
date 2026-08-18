"""
Visualization Agent.

Automatically selects the best chart type and generates Plotly
visualizations based on query result data shape and column types.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import AgentState
from config.logging_config import get_logger

logger = get_logger("agents.visualization")

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "visualization.txt")


class VisualizationAgent:
    """
    Generates Plotly visualizations from query results.

    Uses a combination of LLM-based chart recommendation and
    rule-based fallback to select the optimal chart type.
    Supports bar, line, pie, scatter, histogram, heatmap, and table.
    """

    # Dark theme template for consistent styling
    DARK_THEME = {
        "template": "plotly_dark",
        "paper_bgcolor": "#0E1117",
        "plot_bgcolor": "#0E1117",
        "font": {"color": "#FAFAFA", "family": "Inter, sans-serif"},
        "colorway": [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        ],
    }

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        """
        Initialize the Visualization Agent.

        Args:
            llm: Optional LLM for chart type recommendation.
                 Falls back to rule-based selection if None.
        """
        self._llm = llm
        self._prompt_template = self._load_prompt()
        logger.info("VisualizationAgent initialized (llm=%s)", "yes" if llm else "rule-based")

    def _load_prompt(self) -> str:
        """Load the visualization prompt template."""
        try:
            with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def run(self, state: AgentState) -> AgentState:
        """
        Generate a visualization from query results.

        Args:
            state: Current workflow state with query_results.

        Returns:
            Updated state with 'visualization' and 'chart_config'.
        """
        results = state.get("query_results", [])
        columns = state.get("result_columns", [])
        question = state.get("user_question", "")
        row_count = state.get("result_row_count", 0)

        if not results or row_count == 0:
            state["visualization"] = None
            state["chart_config"] = None
            state["current_step"] = "visualization"
            logger.info("No data to visualize")
            return state

        df = pd.DataFrame(results)

        try:
            # Get chart recommendation
            config = self._get_chart_config(df, question)

            # Generate the chart
            fig = self._create_chart(df, config)

            # Apply dark theme
            fig.update_layout(**self.DARK_THEME)
            fig.update_layout(
                title={
                    "text": config.get("title", "Query Results"),
                    "font": {"size": 18},
                    "x": 0.5,
                    "xanchor": "center",
                },
                margin=dict(l=60, r=40, t=80, b=60),
                height=500,
            )

            state["visualization"] = fig.to_json()
            state["chart_config"] = config
            state["current_step"] = "visualization"
            state["error"] = None

            logger.info("Chart created: type=%s, title='%s'", config.get("chart_type"), config.get("title"))

        except Exception as e:
            logger.exception("Visualization failed")
            # Fallback: create a simple table
            state["visualization"] = None
            state["chart_config"] = {"chart_type": "table", "title": "Query Results"}
            state["current_step"] = "visualization"

        return state

    def _get_chart_config(self, df: pd.DataFrame, question: str) -> dict[str, Any]:
        """
        Determine the best chart configuration.

        Uses LLM if available, otherwise falls back to rule-based logic.
        """
        if self._llm and self._prompt_template:
            try:
                return self._llm_recommend(df, question)
            except Exception as e:
                logger.warning("LLM chart recommendation failed: %s. Using rules.", str(e))

        return self._rule_based_recommend(df, question)

    def _llm_recommend(self, df: pd.DataFrame, question: str) -> dict[str, Any]:
        """Use LLM to recommend chart type."""
        column_types = {col: str(df[col].dtype) for col in df.columns}
        sample = df.head(5).to_string(index=False)

        prompt = self._prompt_template.format(
            question=question,
            columns=list(df.columns),
            column_types=json.dumps(column_types),
            row_count=len(df),
            sample_data=sample,
        )

        messages = [
            SystemMessage(content="You are a data visualization expert. Respond only with valid JSON."),
            HumanMessage(content=prompt),
        ]

        response = self._llm.invoke(messages)
        content = response.content.strip()

        # Parse JSON from response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            config = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                config = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse chart config JSON")

        return config

    def _rule_based_recommend(self, df: pd.DataFrame, question: str) -> dict[str, Any]:
        """
        Rule-based chart type recommendation.

        Analyzes column types, data shape, and question keywords
        to select the most appropriate chart type.
        """
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = []

        # Detect date columns (may be stored as strings)
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    pd.to_datetime(df[col].head(10), errors="raise")
                    datetime_cols.append(col)
                    if col in categorical_cols:
                        categorical_cols.remove(col)
                except (ValueError, TypeError):
                    pass
            elif "datetime" in str(df[col].dtype).lower():
                datetime_cols.append(col)

        question_lower = question.lower()
        n_rows = len(df)
        n_cols = len(df.columns)

        # --- Decision tree ---
        chart_type = "bar"
        x_col = None
        y_col = None
        color_col = None
        orientation = "v"

        # Very few rows or many columns → table
        if n_rows <= 3 or n_cols > 8:
            chart_type = "table"
        # Time series detection
        elif datetime_cols and numeric_cols:
            chart_type = "line"
            x_col = datetime_cols[0]
            y_col = numeric_cols[0]
        # "trend" or "over time" keywords → line
        elif any(kw in question_lower for kw in ["trend", "over time", "monthly", "weekly", "daily", "yearly"]):
            chart_type = "line"
            x_col = categorical_cols[0] if categorical_cols else (df.columns[0] if len(df.columns) > 0 else None)
            y_col = numeric_cols[0] if numeric_cols else (df.columns[1] if len(df.columns) > 1 else None)
        # Distribution keywords → histogram
        elif any(kw in question_lower for kw in ["distribution", "spread", "histogram"]):
            chart_type = "histogram"
            x_col = numeric_cols[0] if numeric_cols else df.columns[0]
        # Proportion/percentage → pie (only with few categories)
        elif any(kw in question_lower for kw in ["proportion", "percentage", "share", "breakdown"]) and n_rows <= 8:
            chart_type = "pie"
            x_col = categorical_cols[0] if categorical_cols else df.columns[0]
            y_col = numeric_cols[0] if numeric_cols else df.columns[1] if len(df.columns) > 1 else None
        # Correlation → scatter
        elif len(numeric_cols) >= 2 and any(kw in question_lower for kw in ["correlation", "scatter", "vs", "versus"]):
            chart_type = "scatter"
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        # Ranking/top → horizontal bar
        elif any(kw in question_lower for kw in ["top", "ranking", "best", "worst", "highest", "lowest"]):
            chart_type = "bar"
            orientation = "h"
            x_col = numeric_cols[0] if numeric_cols else df.columns[-1]
            y_col = categorical_cols[0] if categorical_cols else df.columns[0]
        # Default: categorical + numeric → bar
        elif categorical_cols and numeric_cols:
            chart_type = "bar"
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
        # Two numeric → scatter
        elif len(numeric_cols) >= 2:
            chart_type = "scatter"
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        # Single column → histogram
        elif len(df.columns) == 1:
            chart_type = "histogram"
            x_col = df.columns[0]

        # Auto-select columns if not set
        if x_col is None and len(df.columns) > 0:
            x_col = df.columns[0]
        if y_col is None and len(df.columns) > 1:
            y_col = df.columns[1]

        # Color column for multi-dimensional data
        if len(categorical_cols) >= 2 and chart_type in ("bar", "line"):
            color_col = categorical_cols[1]

        title = self._generate_title(question, chart_type)

        return {
            "chart_type": chart_type,
            "title": title,
            "x_column": x_col,
            "y_column": y_col,
            "color_column": color_col,
            "orientation": orientation,
            "sort_values": chart_type == "bar",
            "reasoning": f"Rule-based: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, {len(datetime_cols)} datetime cols",
        }

    def _create_chart(self, df: pd.DataFrame, config: dict[str, Any]) -> go.Figure:
        """
        Create a Plotly figure based on the chart configuration.

        Args:
            df: Query results DataFrame.
            config: Chart configuration from recommendation.

        Returns:
            Plotly Figure object.
        """
        chart_type = config.get("chart_type", "bar")
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        color_col = config.get("color_column")
        orientation = config.get("orientation", "v")
        title = config.get("title", "Query Results")

        # Ensure columns exist in DataFrame
        x_col = x_col if x_col in df.columns else df.columns[0] if len(df.columns) > 0 else None
        y_col = y_col if y_col and y_col in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        color_col = color_col if color_col and color_col in df.columns else None

        # Sort for bar charts
        if config.get("sort_values") and y_col and y_col in df.columns:
            df = df.sort_values(by=y_col, ascending=(orientation == "v"))

        chart_creators = {
            "bar": self._create_bar,
            "line": self._create_line,
            "pie": self._create_pie,
            "scatter": self._create_scatter,
            "histogram": self._create_histogram,
            "heatmap": self._create_heatmap,
            "table": self._create_table,
        }

        creator = chart_creators.get(chart_type, self._create_bar)
        return creator(df, x_col, y_col, color_col, orientation, title)

    @staticmethod
    def _create_bar(df, x, y, color, orientation, title) -> go.Figure:
        if orientation == "h":
            return px.bar(df, x=y, y=x, color=color, orientation="h", title=title)
        return px.bar(df, x=x, y=y, color=color, title=title)

    @staticmethod
    def _create_line(df, x, y, color, orientation, title) -> go.Figure:
        return px.line(df, x=x, y=y, color=color, title=title, markers=True)

    @staticmethod
    def _create_pie(df, x, y, color, orientation, title) -> go.Figure:
        return px.pie(df, names=x, values=y, title=title, hole=0.4)

    @staticmethod
    def _create_scatter(df, x, y, color, orientation, title) -> go.Figure:
        return px.scatter(df, x=x, y=y, color=color, title=title)

    @staticmethod
    def _create_histogram(df, x, y, color, orientation, title) -> go.Figure:
        return px.histogram(df, x=x, color=color, title=title, nbins=30)

    @staticmethod
    def _create_heatmap(df, x, y, color, orientation, title) -> go.Figure:
        numeric_df = df.select_dtypes(include=["number"])
        if len(numeric_df.columns) >= 2:
            return px.imshow(
                numeric_df.corr(),
                title=title,
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
        return px.bar(df, x=x, y=y, title=title)

    @staticmethod
    def _create_table(df, x, y, color, orientation, title) -> go.Figure:
        fig = go.Figure(
            data=[go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color="#262730",
                    font=dict(color="white", size=13),
                    align="left",
                    height=35,
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="#0E1117",
                    font=dict(color="#FAFAFA", size=12),
                    align="left",
                    height=30,
                ),
            )]
        )
        fig.update_layout(title=title)
        return fig

    @staticmethod
    def _generate_title(question: str, chart_type: str) -> str:
        """Generate a descriptive chart title from the question."""
        # Clean and capitalize the question for use as title
        title = question.strip().rstrip("?").strip()
        if len(title) > 60:
            title = title[:57] + "..."
        return title.title()
