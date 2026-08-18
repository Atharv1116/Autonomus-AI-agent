"""
Tests for the full LangGraph workflow.

Integration tests that run the complete pipeline with mocked LLM
and SQLite in-memory database.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy.engine import Engine

from agents.executor import ExecutorAgent
from agents.guardrail import GuardrailAgent
from agents.insights import InsightAgent
from agents.planner import PlannerAgent
from agents.sql_generator import SQLGeneratorAgent
from agents.visualization import VisualizationAgent
from database.reflection import SchemaReflector
from graphs.workflow import AnalystWorkflow
from tests.conftest import MockLLM


class TestAnalystWorkflow:
    """Test the full analyst workflow end-to-end."""

    @pytest.fixture
    def workflow(self, test_engine: Engine) -> tuple[AnalystWorkflow, MockLLM]:
        """Create a workflow with mock LLM and test database."""
        # Configure mock LLM to return appropriate responses for each step
        llm = MockLLM()

        # The mock will be called multiple times; we need it to return
        # appropriate responses for planner, sql_generator, visualization, and insight.
        # Since MockLLM returns the same response each time, we use a SequentialMockLLM.
        sequential_llm = SequentialMockLLM([
            # Planner response
            json.dumps({
                "understanding": "Show all products",
                "analysis_type": "detail",
                "tables_needed": ["products"],
                "columns_needed": {"products": ["product_name", "unit_price"]},
                "joins": [],
                "filters": [],
                "aggregations": [],
                "group_by": [],
                "order_by": [],
                "limit": 100,
                "notes": "",
            }),
            # SQL Generator response
            "SELECT product_name, unit_price FROM products ORDER BY unit_price DESC",
            # Visualization response
            json.dumps({
                "chart_type": "bar",
                "title": "Products by Price",
                "x_column": "product_name",
                "y_column": "unit_price",
                "color_column": None,
                "orientation": "v",
                "sort_values": True,
                "reasoning": "Bar chart for categorical vs numeric data",
            }),
            # Insight response
            "### 📊 Key Findings\n- Found 5 products\n- Price range: $29.99 - $1,299.99",
        ])

        workflow = AnalystWorkflow(
            planner=PlannerAgent(sequential_llm),
            sql_generator=SQLGeneratorAgent(sequential_llm),
            guardrail=GuardrailAgent(),
            executor=ExecutorAgent(test_engine),
            visualization=VisualizationAgent(sequential_llm),
            insight=InsightAgent(sequential_llm),
            max_retries=2,
        )

        return workflow, sequential_llm

    def test_full_pipeline_success(self, workflow: tuple, test_engine: Engine) -> None:
        """Full pipeline should execute successfully with valid query."""
        wf, llm = workflow

        reflector = SchemaReflector(test_engine)
        schema = reflector.get_schema_for_llm()

        result = wf.run(
            question="Show all products with prices",
            schema_info=schema,
            database_dialect="sqlite",
        )

        # Should have results
        assert result.get("result_row_count", 0) > 0
        assert result.get("query_results") is not None
        assert len(result.get("result_columns", [])) > 0

        # Should have execution times
        assert "total" in result.get("execution_times", {})

        # Should have insights
        assert result.get("insights") is not None

    def test_pipeline_has_execution_times(self, workflow: tuple, test_engine: Engine) -> None:
        """Pipeline should track execution times for each step."""
        wf, _ = workflow

        reflector = SchemaReflector(test_engine)
        result = wf.run(
            question="Show products",
            schema_info=reflector.get_schema_for_llm(),
            database_dialect="sqlite",
        )

        times = result.get("execution_times", {})
        assert "planner" in times
        assert "sql_generator" in times
        assert "guardrail" in times
        assert "executor" in times
        assert "total" in times

    def test_graph_visualization(self, workflow: tuple) -> None:
        """Should generate a Mermaid diagram."""
        wf, _ = workflow
        mermaid = wf.get_graph_visualization()
        assert "Planner" in mermaid
        assert "Guardrail" in mermaid
        assert "Executor" in mermaid


class SequentialMockLLM:
    """Mock LLM that returns different responses for sequential calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_index = 0

    def invoke(self, messages, **kwargs):
        from unittest.mock import MagicMock

        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1

        mock = MagicMock()
        mock.content = response
        mock.response_metadata = {}
        mock.usage_metadata = None
        return mock


class TestSchemaReflector:
    """Test the SchemaReflector utility."""

    def test_reflect_detects_tables(self, test_engine: Engine) -> None:
        """Should detect all tables in the test database."""
        reflector = SchemaReflector(test_engine)
        schema = reflector.reflect()

        assert schema["total_tables"] == 4
        table_names = schema["table_names"]
        assert "products" in table_names
        assert "customers" in table_names
        assert "orders" in table_names
        assert "sales" in table_names

    def test_reflect_detects_columns(self, test_engine: Engine) -> None:
        """Should detect columns for each table."""
        reflector = SchemaReflector(test_engine)
        columns = reflector.get_column_names("products")

        assert "product_id" in columns
        assert "product_name" in columns
        assert "unit_price" in columns

    def test_llm_schema_format(self, test_engine: Engine) -> None:
        """LLM schema format should include all tables and columns."""
        reflector = SchemaReflector(test_engine)
        schema_str = reflector.get_schema_for_llm()

        assert "products" in schema_str
        assert "customers" in schema_str
        assert "product_name" in schema_str

    def test_schema_caching(self, test_engine: Engine) -> None:
        """Schema should be cached after first reflection."""
        reflector = SchemaReflector(test_engine, cache_ttl=300)

        schema1 = reflector.reflect()
        schema2 = reflector.reflect()

        # Should be the same cached instance
        assert schema1 is schema2

    def test_force_refresh(self, test_engine: Engine) -> None:
        """Force refresh should bypass cache."""
        reflector = SchemaReflector(test_engine)

        schema1 = reflector.reflect()
        schema2 = reflector.reflect(force=True)

        # Both should have same content
        assert schema1["total_tables"] == schema2["total_tables"]

    def test_column_search(self, test_engine: Engine) -> None:
        """Semantic search should find relevant columns."""
        reflector = SchemaReflector(test_engine)
        reflector.reflect()

        matches = reflector.search_columns("price")
        assert len(matches) > 0
        # Should find unit_price in products and/or sales
        column_names = [m["column"] for m in matches]
        assert any("price" in c for c in column_names)
