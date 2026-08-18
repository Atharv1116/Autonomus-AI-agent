"""
LangGraph workflow orchestration.

Defines the multi-agent StateGraph that chains all six agents:
Planner → SQL Generator → Guardrail → Executor → Visualization → Insight

Includes conditional routing for guardrail retries and error handling.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from agents.executor import ExecutorAgent
from agents.guardrail import GuardrailAgent
from agents.insights import InsightAgent
from agents.planner import PlannerAgent
from agents.sql_generator import SQLGeneratorAgent
from agents.state import AgentState
from agents.visualization import VisualizationAgent
from config.logging_config import get_logger
from utils.timer import ExecutionTimer
from utils.token_tracker import TokenTracker

logger = get_logger("graphs.workflow")


class AnalystWorkflow:
    """
    Orchestrates the multi-agent analysis pipeline using LangGraph.

    Pipeline: Planner → SQL Generator → Guardrail → Executor → Visualization → Insight

    Guardrail failures trigger retries back to the SQL Generator (max 3).
    """

    def __init__(
        self,
        planner: PlannerAgent,
        sql_generator: SQLGeneratorAgent,
        guardrail: GuardrailAgent,
        executor: ExecutorAgent,
        visualization: VisualizationAgent,
        insight: InsightAgent,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the workflow with all agent instances.

        Args:
            planner: Planner agent instance.
            sql_generator: SQL generator agent instance.
            guardrail: Guardrail agent instance.
            executor: Executor agent instance.
            visualization: Visualization agent instance.
            insight: Insight agent instance.
            max_retries: Maximum SQL generation retries on guardrail failure.
        """
        self._planner = planner
        self._sql_generator = sql_generator
        self._guardrail = guardrail
        self._executor = executor
        self._visualization = visualization
        self._insight = insight
        self._max_retries = max_retries

        self._graph = self._build_graph()
        self._compiled = self._graph.compile()

        logger.info("AnalystWorkflow initialized (max_retries=%d)", max_retries)

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph with all nodes and edges.

        Returns:
            Configured StateGraph (not yet compiled).
        """
        graph = StateGraph(AgentState)

        # --- Add nodes ---
        graph.add_node("planner", self._run_planner)
        graph.add_node("sql_generator", self._run_sql_generator)
        graph.add_node("guardrail", self._run_guardrail)
        graph.add_node("executor", self._run_executor)
        graph.add_node("visualization", self._run_visualization)
        graph.add_node("insight", self._run_insight)

        # --- Define edges ---
        graph.set_entry_point("planner")

        # Planner → SQL Generator
        graph.add_edge("planner", "sql_generator")

        # SQL Generator → Guardrail
        graph.add_edge("sql_generator", "guardrail")

        # Guardrail → (conditional) → Executor OR SQL Generator (retry)
        graph.add_conditional_edges(
            "guardrail",
            self._route_after_guardrail,
            {
                "executor": "executor",
                "sql_generator": "sql_generator",
                "end": END,
            },
        )

        # Executor → Visualization
        graph.add_edge("executor", "visualization")

        # Visualization → Insight
        graph.add_edge("visualization", "insight")

        # Insight → END
        graph.add_edge("insight", END)

        logger.info("Workflow graph built: 6 nodes, conditional guardrail routing")
        return graph

    def _route_after_guardrail(self, state: AgentState) -> str:
        """
        Route after guardrail: to executor if valid, retry if invalid.

        Args:
            state: Current workflow state.

        Returns:
            Next node name: "executor", "sql_generator", or "end".
        """
        guardrail_result = state.get("guardrail_result", {})
        retry_count = state.get("retry_count", 0)

        if guardrail_result.get("is_valid", False):
            logger.info("Guardrail PASSED → routing to executor")
            return "executor"

        if retry_count < self._max_retries:
            logger.warning(
                "Guardrail FAILED (retry %d/%d) → routing back to sql_generator",
                retry_count + 1, self._max_retries,
            )
            return "sql_generator"

        logger.error("Guardrail FAILED: max retries (%d) exceeded", self._max_retries)
        return "end"

    # ========================================
    # Node runner methods (with timing)
    # ========================================

    def _run_planner(self, state: AgentState) -> AgentState:
        """Run the planner agent with execution timing."""
        start = time.perf_counter()
        state = self._planner.run(state)
        elapsed = time.perf_counter() - start

        times = state.get("execution_times", {})
        times["planner"] = round(elapsed, 3)
        state["execution_times"] = times

        return state

    def _run_sql_generator(self, state: AgentState) -> AgentState:
        """Run the SQL generator agent with retry tracking."""
        start = time.perf_counter()

        # Increment retry count if this is a retry
        guardrail_result = state.get("guardrail_result", {})
        if guardrail_result and not guardrail_result.get("is_valid", True):
            state["retry_count"] = state.get("retry_count", 0) + 1
            state["error"] = (
                f"Previous SQL failed validation: {guardrail_result.get('reason', 'unknown')}. "
                f"Retry {state['retry_count']}/{self._max_retries}. Please fix the issues."
            )

        state = self._sql_generator.run(state)
        elapsed = time.perf_counter() - start

        times = state.get("execution_times", {})
        times["sql_generator"] = round(elapsed, 3)
        state["execution_times"] = times

        return state

    def _run_guardrail(self, state: AgentState) -> AgentState:
        """Run the guardrail agent."""
        start = time.perf_counter()
        state = self._guardrail.run(state)
        elapsed = time.perf_counter() - start

        times = state.get("execution_times", {})
        times["guardrail"] = round(elapsed, 3)
        state["execution_times"] = times

        return state

    def _run_executor(self, state: AgentState) -> AgentState:
        """Run the executor agent."""
        start = time.perf_counter()
        state = self._executor.run(state)
        elapsed = time.perf_counter() - start

        times = state.get("execution_times", {})
        times["executor"] = round(elapsed, 3)
        state["execution_times"] = times

        return state

    def _run_visualization(self, state: AgentState) -> AgentState:
        """Run the visualization agent."""
        start = time.perf_counter()
        state = self._visualization.run(state)
        elapsed = time.perf_counter() - start

        times = state.get("execution_times", {})
        times["visualization"] = round(elapsed, 3)
        state["execution_times"] = times

        return state

    def _run_insight(self, state: AgentState) -> AgentState:
        """Run the insight agent."""
        start = time.perf_counter()
        state = self._insight.run(state)
        elapsed = time.perf_counter() - start

        times = state.get("execution_times", {})
        times["insight"] = round(elapsed, 3)
        state["execution_times"] = times

        return state

    # ========================================
    # Public API
    # ========================================

    def run(
        self,
        question: str,
        schema_info: str,
        database_dialect: str = "postgresql",
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> AgentState:
        """
        Execute the full analysis pipeline.

        Args:
            question: User's natural language question.
            schema_info: LLM-formatted database schema.
            database_dialect: Database dialect name.
            conversation_history: Previous conversation messages.

        Returns:
            Final AgentState with all results.
        """
        logger.info("=" * 60)
        logger.info("WORKFLOW START: '%s'", question[:100])
        logger.info("=" * 60)

        initial_state: AgentState = {
            "user_question": question,
            "schema_info": schema_info,
            "database_dialect": database_dialect,
            "conversation_history": conversation_history or [],
            "retry_count": 0,
            "max_retries": self._max_retries,
            "execution_times": {},
            "token_usage": {},
            "metadata": {
                "start_time": time.time(),
            },
        }

        try:
            # Execute the compiled graph
            final_state = self._compiled.invoke(initial_state)

            # Add total execution time
            if isinstance(final_state, dict):
                total_time = time.time() - initial_state["metadata"]["start_time"]
                final_state.setdefault("execution_times", {})["total"] = round(total_time, 3)
                final_state.setdefault("metadata", {})["end_time"] = time.time()

                logger.info(
                    "WORKFLOW COMPLETE: %.2fs total, %d rows, %s",
                    total_time,
                    final_state.get("result_row_count", 0),
                    "success" if not final_state.get("error") else f"error: {final_state.get('error', '')[:80]}",
                )

            return final_state

        except Exception as e:
            logger.exception("WORKFLOW FAILED")
            return {
                **initial_state,
                "error": f"Workflow execution failed: {str(e)}",
                "execution_times": {"total": time.time() - initial_state["metadata"]["start_time"]},
            }

    def get_graph_visualization(self) -> str:
        """
        Generate a Mermaid diagram of the workflow graph.

        Returns:
            Mermaid diagram string.
        """
        return """
graph TD
    A[🧠 Planner Agent] --> B[⚙️ SQL Generator]
    B --> C{🛡️ Guardrail}
    C -->|Valid| D[🗄️ SQL Executor]
    C -->|Invalid & Retries Left| B
    C -->|Max Retries| E[❌ End with Error]
    D --> F[📊 Visualization]
    F --> G[💡 Insight Agent]
    G --> H[✅ Final Response]
"""
