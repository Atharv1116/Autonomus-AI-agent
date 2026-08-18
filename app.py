"""
Autonomous Data Analyst Agent — Streamlit Dashboard.

A modern, dark-themed dashboard with chat interface, SQL panel,
data tables, Plotly visualizations, executive insights, and export capabilities.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

# Load environment before any project imports
load_dotenv()

from agents.executor import ExecutorAgent
from agents.guardrail import GuardrailAgent
from agents.insights import InsightAgent
from agents.planner import PlannerAgent
from agents.sql_generator import SQLGeneratorAgent
from agents.visualization import VisualizationAgent
from config.logging_config import get_logger, setup_logging
from config.settings import LLMProvider, get_settings, reload_settings
from database.connection import DatabaseManager
from database.csv_handler import CSVHandler
from database.reflection import SchemaReflector
from graphs.workflow import AnalystWorkflow
from utils.export import ExportManager
from utils.llm_provider import LLMProviderFactory

logger = get_logger("app")


# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Autonomous Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# Custom CSS — Premium Dark Theme
# ============================================
def apply_custom_css() -> None:
    """Apply custom dark theme CSS for a premium look."""
    st.markdown("""
    <style>
        /* --- Global Dark Theme --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* --- Header --- */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .main-header h1 {
            color: white;
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }
        .main-header p {
            color: rgba(255,255,255,0.85);
            font-size: 1rem;
            margin: 0.5rem 0 0;
        }

        /* --- Metric Cards --- */
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }
        .metric-card .label {
            color: #8b8fa3;
            font-size: 0.78rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card .value {
            color: #e0e0e0;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        /* --- SQL Code Block --- */
        .sql-block {
            background: #1e1e2f;
            border: 1px solid #2d2d44;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.85rem;
            color: #a6e22e;
            overflow-x: auto;
            white-space: pre-wrap;
        }

        /* --- Status Badges --- */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .status-success {
            background: rgba(0, 204, 150, 0.15);
            color: #00CC96;
            border: 1px solid rgba(0, 204, 150, 0.3);
        }
        .status-error {
            background: rgba(239, 85, 59, 0.15);
            color: #EF553B;
            border: 1px solid rgba(239, 85, 59, 0.3);
        }
        .status-processing {
            background: rgba(99, 110, 250, 0.15);
            color: #636EFA;
            border: 1px solid rgba(99, 110, 250, 0.3);
        }

        /* --- Pipeline Steps --- */
        .step-indicator {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            margin: 0.25rem 0;
        }
        .step-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .step-active { background: #636EFA; }
        .step-done { background: #00CC96; }
        .step-pending { background: #4a4a5a; }
        .step-error { background: #EF553B; }

        /* --- Sidebar Styling --- */
        section[data-testid="stSidebar"] {
            background: #0E1117;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* --- Hide default Streamlit elements --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* --- Tab Styling --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================
# Session State Initialization
# ============================================
def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "sql_history": [],
        "db_connected": False,
        "db_manager": None,
        "schema_reflector": None,
        "workflow": None,
        "csv_handler": CSVHandler(),
        "export_manager": ExportManager(),
        "current_results": None,
        "current_fig": None,
        "current_sql": None,
        "current_insights": None,
        "execution_times": {},
        "token_usage": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================
# Database Connection
# ============================================
def connect_database(db_url: str) -> bool:
    """
    Establish database connection and reflect schema.

    Args:
        db_url: SQLAlchemy database URL.

    Returns:
        True if connection successful.
    """
    try:
        db_manager = DatabaseManager(db_url)
        if not db_manager.test_connection():
            st.error("❌ Database connection failed. Check your connection URL.")
            return False

        schema_reflector = SchemaReflector(db_manager.engine)
        schema_reflector.reflect(force=True)

        st.session_state.db_manager = db_manager
        st.session_state.schema_reflector = schema_reflector
        st.session_state.db_connected = True

        logger.info("Database connected and schema reflected")
        return True

    except Exception as e:
        st.error(f"❌ Connection error: {str(e)}")
        logger.exception("Database connection failed")
        return False


def build_workflow(
    provider: str,
    model: str,
    api_key: str,
    base_url: str = "",
) -> Optional[AnalystWorkflow]:
    """
    Build the analysis workflow with the selected LLM provider.

    Args:
        provider: LLM provider name.
        model: Model name.
        api_key: API key.
        base_url: Base URL for self-hosted providers.

    Returns:
        Configured AnalystWorkflow or None on failure.
    """
    try:
        llm = LLMProviderFactory.create(
            provider=provider,
            model=model,
            api_key=api_key if api_key else None,
            base_url=base_url if base_url else None,
            temperature=0.0,
            streaming=False,
        )

        settings = get_settings()
        db_manager = st.session_state.db_manager

        workflow = AnalystWorkflow(
            planner=PlannerAgent(llm),
            sql_generator=SQLGeneratorAgent(llm, max_rows=settings.max_query_rows),
            guardrail=GuardrailAgent(),
            executor=ExecutorAgent(
                engine=db_manager.engine,
                max_rows=settings.max_query_rows,
                timeout_seconds=settings.query_timeout_seconds,
            ),
            visualization=VisualizationAgent(llm),
            insight=InsightAgent(llm),
            max_retries=settings.max_retries,
        )

        st.session_state.workflow = workflow
        logger.info("Workflow built: provider=%s, model=%s", provider, model)
        return workflow

    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {str(e)}")
        logger.exception("Workflow build failed")
        return None


# ============================================
# Sidebar
# ============================================
def render_sidebar() -> None:
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # --- Database Connection ---
        st.markdown("### 🗄️ Database")
        db_url = st.text_input(
            "Connection URL",
            value=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/analyst_db"),
            type="password",
            help="SQLAlchemy connection URL",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔌 Connect", use_container_width=True):
                with st.spinner("Connecting..."):
                    if connect_database(db_url):
                        st.success("✅ Connected!")
                        st.rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                if st.session_state.schema_reflector:
                    st.session_state.schema_reflector.reflect(force=True)
                    st.success("✅ Refreshed!")

        if st.session_state.db_connected:
            st.markdown('<span class="status-badge status-success">● Connected</span>', unsafe_allow_html=True)
            reflector = st.session_state.schema_reflector
            if reflector:
                tables = reflector.get_table_names()
                st.caption(f"📋 {len(tables)} tables detected")
                with st.expander("View Tables"):
                    for t in tables:
                        cols = reflector.get_column_names(t)
                        st.markdown(f"**{t}** ({len(cols)} cols)")
        else:
            st.markdown('<span class="status-badge status-error">● Disconnected</span>', unsafe_allow_html=True)

        st.divider()

        # --- LLM Configuration ---
        st.markdown("### 🤖 LLM Provider")
        provider = st.selectbox(
            "Provider",
            options=["nvidia_nim", "openai", "groq", "ollama"],
            index=0,
            help="Select your LLM provider",
        )

        default_models = {
            "nvidia_nim": "meta/llama-3.1-70b-instruct",
            "openai": "gpt-4o",
            "groq": "llama-3.1-70b-versatile",
            "ollama": "llama3.1",
        }

        model = st.text_input("Model", value=default_models.get(provider, ""))

        api_key = ""
        base_url = ""
        if provider != "ollama":
            api_key = st.text_input(
                "API Key",
                type="password",
                value=os.getenv(f"{provider.upper().replace('_NIM', '')}_API_KEY", ""),
            )
        else:
            base_url = st.text_input(
                "Ollama URL",
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

        if st.button("🚀 Initialize Agent", use_container_width=True):
            if not st.session_state.db_connected:
                st.warning("⚠️ Connect to database first!")
            else:
                with st.spinner("Initializing..."):
                    workflow = build_workflow(provider, model, api_key, base_url)
                    if workflow:
                        st.success("✅ Agent ready!")
                        st.rerun()

        if st.session_state.workflow:
            st.markdown('<span class="status-badge status-success">● Agent Ready</span>', unsafe_allow_html=True)

        st.divider()

        # --- CSV Upload ---
        st.markdown("### 📁 CSV Upload")
        uploaded_file = st.file_uploader("Upload CSV for analysis", type=["csv"])
        if uploaded_file:
            csv_handler = st.session_state.csv_handler
            info = csv_handler.load_csv(
                file_content=uploaded_file.getvalue(),
                file_name=uploaded_file.name,
            )
            st.success(f"✅ Loaded: `{info['table_name']}` ({info['row_count']} rows)")

        st.divider()

        # --- SQL History ---
        st.markdown("### 📜 SQL History")
        if st.session_state.sql_history:
            for i, entry in enumerate(reversed(st.session_state.sql_history[-10:])):
                with st.expander(f"#{len(st.session_state.sql_history) - i}: {entry['question'][:40]}..."):
                    st.code(entry["sql"], language="sql")
                    st.caption(f"⏱️ {entry.get('time', 'N/A')}s | 📊 {entry.get('rows', 0)} rows")
        else:
            st.caption("No queries yet")

        st.divider()

        # --- Clear History ---
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sql_history = []
            st.session_state.current_results = None
            st.session_state.current_fig = None
            st.session_state.current_sql = None
            st.session_state.current_insights = None
            st.rerun()


# ============================================
# Main Chat Interface
# ============================================
def render_main_content() -> None:
    """Render the main content area with chat and results."""

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Autonomous Data Analyst Agent</h1>
        <p>Ask questions about your data in natural language — no SQL required</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Example Questions ---
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        example_cols = st.columns(3)
        examples = [
            "What are the top 10 selling products?",
            "Show monthly revenue trend",
            "Which city generated the highest sales?",
            "What is the average order value by customer segment?",
            "Show customer distribution by age group",
            "Compare sales performance across regions",
        ]
        for i, example in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(f"💬 {example}", key=f"example_{i}", use_container_width=True):
                    st.session_state.pending_question = example
                    st.rerun()

    # --- Chat Messages ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat Input ---
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask a question about your data...") or pending

    if user_input:
        process_question(user_input)


def process_question(question: str) -> None:
    """
    Process a user's question through the full pipeline.

    Args:
        question: Natural language question.
    """
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Validate state
    if not st.session_state.workflow:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please connect to a database and initialize the agent first (see sidebar).")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Please connect to a database and initialize the agent first.",
        })
        return

    # Run workflow
    with st.chat_message("assistant"):
        # Status indicators
        status = st.status("🔄 Analyzing your question...", expanded=True)

        with status:
            st.write("🧠 **Step 1/6** — Planning analysis strategy...")
            start_time = time.time()

            # Get schema info (include CSV tables if any)
            reflector = st.session_state.schema_reflector
            schema_info = reflector.get_schema_for_llm()

            csv_schema = st.session_state.csv_handler.get_schema_for_llm()
            if csv_schema:
                schema_info += "\n\n" + csv_schema

            # Build conversation history
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-10:]
            ]

            try:
                # Execute workflow
                workflow = st.session_state.workflow
                result = workflow.run(
                    question=question,
                    schema_info=schema_info,
                    database_dialect=st.session_state.db_manager.get_dialect_name(),
                    conversation_history=history,
                )

                elapsed = time.time() - start_time
                exec_times = result.get("execution_times", {})

                # Update status with step timings
                for step_name, step_time in exec_times.items():
                    if step_name != "total":
                        st.write(f"✅ **{step_name.replace('_', ' ').title()}**: {step_time:.2f}s")

                status.update(label=f"✅ Analysis complete ({elapsed:.1f}s)", state="complete")

            except Exception as e:
                status.update(label="❌ Analysis failed", state="error")
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Analysis failed: {str(e)}",
                })
                return

        # --- Display Results ---
        error = result.get("error")
        if error and not result.get("query_results"):
            st.error(f"⚠️ {error}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {error}",
            })
            return

        # Build response
        response_parts: list[str] = []

        # --- Tabs for results ---
        tabs = st.tabs(["📊 Results", "🔍 SQL", "📈 Chart", "💡 Insights", "⏱️ Metrics"])

        # Tab 1: Data Table
        with tabs[0]:
            results_data = result.get("query_results", [])
            if results_data:
                df = pd.DataFrame(results_data)
                st.session_state.current_results = df

                row_count = result.get("result_row_count", len(df))
                st.markdown(f"**{row_count:,} rows** × **{len(df.columns)} columns** returned")
                st.dataframe(df, use_container_width=True, height=400)

                # Export buttons
                export_cols = st.columns(3)
                with export_cols[0]:
                    csv_bytes = st.session_state.export_manager.export_csv(df, to_bytes=True)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv_bytes,
                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                response_parts.append(f"📊 Query returned **{row_count:,} rows**")
            else:
                st.info("No results returned for this query.")
                response_parts.append("Query returned no results.")

        # Tab 2: Generated SQL
        with tabs[1]:
            sql = result.get("generated_sql", "")
            if sql:
                st.session_state.current_sql = sql
                st.code(sql, language="sql")

                guardrail = result.get("guardrail_result", {})
                if guardrail.get("is_valid"):
                    st.markdown('<span class="status-badge status-success">✅ Guardrail Passed</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-badge status-error">❌ Guardrail Failed</span>', unsafe_allow_html=True)
                    st.warning(guardrail.get("reason", ""))

                # SQL download
                sql_bytes = st.session_state.export_manager.export_sql(
                    sql, to_bytes=True,
                    metadata={"question": question, "timestamp": datetime.now().isoformat()},
                )
                st.download_button(
                    "📥 Download SQL",
                    data=sql_bytes,
                    file_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql",
                    mime="text/plain",
                    use_container_width=True,
                )

                # Add to history
                st.session_state.sql_history.append({
                    "question": question,
                    "sql": sql,
                    "time": round(elapsed, 2),
                    "rows": result.get("result_row_count", 0),
                    "timestamp": datetime.now().isoformat(),
                })

        # Tab 3: Visualization
        with tabs[2]:
            viz_json = result.get("visualization")
            if viz_json:
                try:
                    fig = pio.from_json(viz_json)
                    st.session_state.current_fig = fig
                    st.plotly_chart(fig, use_container_width=True, theme=None)

                    # PNG export
                    try:
                        png_bytes = st.session_state.export_manager.export_chart_png(fig, to_bytes=True)
                        st.download_button(
                            "📥 Download Chart (PNG)",
                            data=png_bytes,
                            file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    except Exception:
                        pass  # Kaleido might not be available

                    response_parts.append("📈 Visualization generated")
                except Exception as e:
                    st.warning(f"Could not render chart: {str(e)}")
            else:
                st.info("No visualization generated for this query.")

        # Tab 4: Insights
        with tabs[3]:
            insights = result.get("insights", "")
            if insights:
                st.session_state.current_insights = insights
                st.markdown(insights)
                response_parts.append("💡 Insights generated")
            else:
                st.info("No insights generated.")

        # Tab 5: Metrics
        with tabs[4]:
            metrics_cols = st.columns(4)
            exec_times = result.get("execution_times", {})

            with metrics_cols[0]:
                st.metric("Total Time", f"{exec_times.get('total', 0):.2f}s")
            with metrics_cols[1]:
                st.metric("Rows", f"{result.get('result_row_count', 0):,}")
            with metrics_cols[2]:
                st.metric("Retries", str(result.get("retry_count", 0)))
            with metrics_cols[3]:
                st.metric("Steps", str(len(exec_times) - 1 if "total" in exec_times else len(exec_times)))

            # Step-by-step timing
            st.markdown("#### ⏱️ Step Execution Times")
            for step, duration in exec_times.items():
                if step != "total":
                    pct = (duration / exec_times.get("total", 1)) * 100
                    st.progress(min(pct / 100, 1.0), text=f"{step.replace('_', ' ').title()}: {duration:.3f}s ({pct:.0f}%)")

        # Add assistant summary to chat
        summary = " | ".join(response_parts) if response_parts else "Analysis complete."
        st.session_state.messages.append({"role": "assistant", "content": summary})


# ============================================
# Main Entry Point
# ============================================
def main() -> None:
    """Main application entry point."""
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), enable_file_logging=False)
    apply_custom_css()
    init_session_state()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
