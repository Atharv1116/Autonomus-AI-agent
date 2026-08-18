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
        /* --- Import Premium Google Fonts --- */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

        /* --- Global Reset & Typography --- */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            background: radial-gradient(circle at 15% 15%, rgba(90, 85, 240, 0.07) 0%, transparent 45%), 
                        radial-gradient(circle at 85% 85%, rgba(0, 242, 254, 0.05) 0%, transparent 45%), 
                        #0b0c10 !important;
            color: #e0e6ed !important;
        }

        h1, h2, h3, h4, h5, h6, .main-header h1 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700;
        }

        code, pre, .sql-block {
            font-family: 'Fira Code', monospace !important;
        }

        /* --- Animations --- */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(16px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes shine {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* --- Custom Scrollbar --- */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(15, 17, 28, 0.3);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(90, 85, 240, 0.3);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 242, 254, 0.5);
        }

        /* --- Header --- */
        .main-header {
            background: linear-gradient(135deg, rgba(20, 24, 43, 0.8) 0%, rgba(10, 12, 22, 0.8) 100%) !important;
            border: 1px solid rgba(90, 85, 240, 0.2) !important;
            box-shadow: 0 8px 32px 0 rgba(90, 85, 240, 0.08) !important;
            backdrop-filter: blur(12px) !important;
            padding: 2rem !important;
            border-radius: 16px !important;
            margin-bottom: 2rem !important;
            text-align: center;
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both !important;
        }
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #5a55f0, #00f2fe, #5a55f0);
            background-size: 200% auto;
            animation: shine 4s linear infinite;
        }
        .main-header h1 {
            color: #ffffff !important;
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            margin: 0 !important;
            background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 100%);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        .main-header p {
            color: #94a3b8 !important;
            font-size: 1.1rem !important;
            font-weight: 400 !important;
            margin: 0.75rem 0 0 !important;
        }

        /* --- Streamlit Metric (Bento Style) --- */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(22, 25, 41, 0.7) 0%, rgba(31, 35, 58, 0.7) 100%) !important;
            border: 1px solid rgba(90, 85, 240, 0.15) !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25) !important;
            backdrop-filter: blur(10px) !important;
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease, box-shadow 0.3s ease !important;
            animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both !important;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-4px) !important;
            border-color: rgba(0, 242, 254, 0.4) !important;
            box-shadow: 0 12px 40px 0 rgba(90, 85, 240, 0.25) !important;
        }
        div[data-testid="stMetric"] label {
            color: #8b8fa3 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.9rem !important;
            font-weight: 800 !important;
            font-family: 'Outfit', sans-serif !important;
        }

        /* --- SQL Code Block --- */
        .sql-block {
            background: #08090f !important;
            border: 1px solid rgba(90, 85, 240, 0.35) !important;
            border-radius: 10px !important;
            padding: 1.25rem !important;
            font-family: 'Fira Code', monospace !important;
            font-size: 0.9rem !important;
            color: #00f2fe !important;
            box-shadow: inset 0 2px 12px rgba(0,0,0,0.6) !important;
            overflow-x: auto;
            white-space: pre-wrap;
        }

        /* --- Status Badges --- */
        .status-badge {
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            padding: 0.35rem 0.85rem !important;
            border-radius: 20px !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.02em !important;
        }
        .status-success {
            background: rgba(0, 242, 254, 0.08) !important;
            color: #00f2fe !important;
            border: 1px solid rgba(0, 242, 254, 0.25) !important;
            box-shadow: 0 0 10px rgba(0, 242, 254, 0.1) !important;
        }
        .status-error {
            background: rgba(239, 85, 59, 0.08) !important;
            color: #EF553B !important;
            border: 1px solid rgba(239, 85, 59, 0.25) !important;
            box-shadow: 0 0 10px rgba(239, 85, 59, 0.1) !important;
        }
        .status-processing {
            background: rgba(90, 85, 240, 0.08) !important;
            color: #a5b4fc !important;
            border: 1px solid rgba(90, 85, 240, 0.25) !important;
            box-shadow: 0 0 10px rgba(90, 85, 240, 0.1) !important;
        }

        /* --- Buttons --- */
        div.stButton > button {
            background: linear-gradient(135deg, #5a55f0 0%, #4640de 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            box-shadow: 0 4px 12px rgba(90, 85, 240, 0.2) !important;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(90, 85, 240, 0.45), 0 0 0 2px rgba(0, 242, 254, 0.2) !important;
            border-color: rgba(0, 242, 254, 0.5) !important;
            background: linear-gradient(135deg, #6b66f2 0%, #524ce6 100%) !important;
        }
        div.stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Secondary buttons (like example question buttons) */
        div[data-testid="column"] div.stButton > button {
            background: rgba(22, 25, 41, 0.5) !important;
            border: 1px solid rgba(90, 85, 240, 0.2) !important;
            color: #a5b4fc !important;
            box-shadow: none !important;
        }
        div[data-testid="column"] div.stButton > button:hover {
            background: rgba(90, 85, 240, 0.15) !important;
            color: #ffffff !important;
            border-color: #5a55f0 !important;
        }

        /* --- Inputs, Selectboxes, & Textareas --- */
        div[data-baseweb="input"] {
            background-color: rgba(15, 17, 28, 0.7) !important;
            border: 1px solid rgba(90, 85, 240, 0.2) !important;
            border-radius: 8px !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #00f2fe !important;
            box-shadow: 0 0 0 3px rgba(0, 242, 254, 0.15) !important;
        }
        div[data-baseweb="select"] {
            background-color: rgba(15, 17, 28, 0.7) !important;
            border: 1px solid rgba(90, 85, 240, 0.2) !important;
            border-radius: 8px !important;
        }

        /* --- Chat Input --- */
        div[data-testid="stChatInput"] {
            background-color: rgba(20, 23, 38, 0.9) !important;
            border: 1px solid rgba(90, 85, 240, 0.25) !important;
            border-radius: 12px !important;
            padding: 0.2rem !important;
            backdrop-filter: blur(12px) !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35) !important;
        }
        div[data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: #ffffff !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        /* --- Sidebar Styling --- */
        section[data-testid="stSidebar"] {
            background: #08090d !important;
            border-right: 1px solid rgba(90, 85, 240, 0.15) !important;
        }
        section[data-testid="stSidebar"] .stMarkdown h2 {
            color: #ffffff;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
        }

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(15, 17, 28, 0.6) !important;
            padding: 6px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(90, 85, 240, 0.15) !important;
            gap: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            padding: 8px 18px !important;
            color: #8b8fa3 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease !important;
            border-bottom: none !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #5a55f0 !important;
            color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(90, 85, 240, 0.25) !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #ffffff !important;
        }

        /* --- Dataframe Wrapper --- */
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(90, 85, 240, 0.15) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            background-color: rgba(15, 17, 28, 0.4) !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        }

        /* --- Chat Messages --- */
        div[data-testid="stChatMessage"] {
            background-color: rgba(20, 22, 37, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
            margin-bottom: 0.8rem !important;
            transition: transform 0.2s ease, border-color 0.2s ease !important;
            animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both !important;
        }
        div[data-testid="stChatMessage"]:hover {
            border-color: rgba(90, 85, 240, 0.2) !important;
            transform: translateY(-2px) !important;
        }

        /* --- Expanders --- */
        div[data-testid="stExpander"] {
            background-color: rgba(15, 17, 28, 0.45) !important;
            border: 1px solid rgba(90, 85, 240, 0.15) !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
        }
        
        /* --- Progress Bar --- */
        div[data-testid="stProgress"] > div > div > div {
            background: linear-gradient(90deg, #5a55f0, #00f2fe) !important;
        }

        /* --- Hide default Streamlit elements --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
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
            "groq": "groq/compound",
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
