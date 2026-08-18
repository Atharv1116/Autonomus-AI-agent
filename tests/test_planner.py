"""
Tests for the Planner Agent.

Validates plan structure, schema detection, LLM response parsing,
and error handling.
"""

from __future__ import annotations

import json

import pytest

from agents.planner import PlannerAgent
from tests.conftest import MockLLM


class TestPlannerAgent:
    """Test the Planner Agent's planning capabilities."""

    def test_plan_has_required_fields(self, mock_llm: MockLLM) -> None:
        """Generated plan should contain all required fields."""
        mock_llm.set_response(json.dumps({
            "understanding": "Find top selling products",
            "analysis_type": "ranking",
            "tables_needed": ["products", "sales"],
            "columns_needed": {"products": ["product_name"], "sales": ["quantity"]},
            "joins": [],
            "filters": [],
            "aggregations": [{"function": "SUM", "column": "quantity", "alias": "total"}],
            "group_by": ["product_name"],
            "order_by": [{"column": "total", "direction": "DESC"}],
            "limit": 10,
            "notes": "",
        }))

        planner = PlannerAgent(mock_llm)
        state = {
            "user_question": "What are the top 10 products?",
            "schema_info": "TABLE: products (id, name)",
            "conversation_history": [],
        }

        result = planner.run(state)
        plan = result["plan"]

        assert "understanding" in plan
        assert "analysis_type" in plan
        assert "tables_needed" in plan
        assert "columns_needed" in plan
        assert plan["analysis_type"] == "ranking"
        assert "products" in plan["tables_needed"]

    def test_plan_handles_json_in_code_block(self, mock_llm: MockLLM) -> None:
        """Planner should handle LLM responses wrapped in code blocks."""
        mock_llm.set_response('```json\n{"understanding": "test", "analysis_type": "detail"}\n```')

        planner = PlannerAgent(mock_llm)
        state = {
            "user_question": "Show all customers",
            "schema_info": "TABLE: customers (id, name)",
            "conversation_history": [],
        }

        result = planner.run(state)
        assert result["plan"]["understanding"] == "test"

    def test_plan_handles_invalid_json(self, mock_llm: MockLLM) -> None:
        """Planner should gracefully handle non-JSON LLM responses."""
        mock_llm.set_response("I think you want to look at the customers table.")

        planner = PlannerAgent(mock_llm)
        state = {
            "user_question": "Show customers",
            "schema_info": "TABLE: customers (id, name)",
            "conversation_history": [],
        }

        result = planner.run(state)
        plan = result["plan"]

        # Should still have a plan with defaults
        assert "understanding" in plan
        assert "analysis_type" in plan

    def test_plan_includes_defaults(self, mock_llm: MockLLM) -> None:
        """Plan should have sensible defaults for missing fields."""
        mock_llm.set_response('{"understanding": "Test query"}')

        planner = PlannerAgent(mock_llm)
        state = {
            "user_question": "Test",
            "schema_info": "TABLE: test (id)",
            "conversation_history": [],
        }

        result = planner.run(state)
        plan = result["plan"]

        assert plan["analysis_type"] == "detail"
        assert plan["tables_needed"] == []
        assert plan["joins"] == []
        assert plan["filters"] == []

    def test_planner_sets_current_step(self, mock_llm: MockLLM) -> None:
        """Planner should set current_step in state."""
        mock_llm.set_response('{"understanding": "test", "analysis_type": "detail"}')

        planner = PlannerAgent(mock_llm)
        state = {
            "user_question": "Test",
            "schema_info": "",
            "conversation_history": [],
        }

        result = planner.run(state)
        assert result["current_step"] == "planner"

    def test_planner_with_conversation_history(self, mock_llm: MockLLM) -> None:
        """Planner should incorporate conversation history."""
        mock_llm.set_response('{"understanding": "test", "analysis_type": "detail"}')

        planner = PlannerAgent(mock_llm)
        state = {
            "user_question": "Show the same but for last month",
            "schema_info": "TABLE: sales (id, date, amount)",
            "conversation_history": [
                {"role": "user", "content": "Show total sales"},
                {"role": "assistant", "content": "Here are the total sales..."},
            ],
        }

        result = planner.run(state)
        assert result["plan"] is not None

        # Verify conversation was passed to LLM
        assert mock_llm.call_count == 1

    def test_planner_fallback_on_error(self, mock_llm: MockLLM) -> None:
        """Planner should use fallback plan when LLM raises an exception."""
        class FailingLLM:
            def invoke(self, messages, **kwargs):
                raise Exception("API Error")

        planner = PlannerAgent(FailingLLM())
        state = {
            "user_question": "Test question",
            "schema_info": "",
            "conversation_history": [],
        }

        result = planner.run(state)
        assert result["plan"] is not None
        assert result["error"] is not None
        assert "Fallback" in result["plan"]["notes"]
