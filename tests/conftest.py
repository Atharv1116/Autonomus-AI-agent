"""
Shared test fixtures and configuration.

Provides mock LLM, test database engine (SQLite in-memory),
sample schema, and reusable test data across all test modules.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine

from agents.guardrail import GuardrailAgent
from agents.state import AgentState


# ============================================
# Mock LLM
# ============================================
class MockLLM:
    """
    Mock LangChain LLM for testing.

    Returns configurable responses for different prompts.
    """

    def __init__(self, response: str = "") -> None:
        self._response = response
        self._call_count = 0
        self._last_messages = None

    def invoke(self, messages: list, **kwargs) -> MagicMock:
        """Mock invoke that returns a configurable response."""
        self._call_count += 1
        self._last_messages = messages
        mock_response = MagicMock()
        mock_response.content = self._response
        mock_response.response_metadata = {}
        mock_response.usage_metadata = None
        return mock_response

    def set_response(self, response: str) -> None:
        """Set the response for the next invoke call."""
        self._response = response

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_messages(self) -> list:
        return self._last_messages


@pytest.fixture
def mock_llm() -> MockLLM:
    """Provide a mock LLM instance."""
    return MockLLM()


# ============================================
# Test Database (SQLite in-memory)
# ============================================
@pytest.fixture
def test_engine() -> Engine:
    """Create an in-memory SQLite database with sample tables."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    # Create test tables
    Table(
        "products", metadata,
        Column("product_id", Integer, primary_key=True),
        Column("product_name", String(200)),
        Column("category", String(100)),
        Column("unit_price", Float),
        Column("stock_quantity", Integer),
    )

    Table(
        "customers", metadata,
        Column("customer_id", Integer, primary_key=True),
        Column("first_name", String(100)),
        Column("last_name", String(100)),
        Column("email", String(255)),
        Column("city", String(100)),
        Column("segment", String(50)),
    )

    Table(
        "orders", metadata,
        Column("order_id", Integer, primary_key=True),
        Column("customer_id", Integer),
        Column("order_date", String(50)),
        Column("total_amount", Float),
        Column("status", String(50)),
    )

    Table(
        "sales", metadata,
        Column("sale_id", Integer, primary_key=True),
        Column("order_id", Integer),
        Column("product_id", Integer),
        Column("quantity", Integer),
        Column("unit_price", Float),
        Column("total_price", Float),
    )

    metadata.create_all(engine)

    # Insert sample data
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO products (product_id, product_name, category, unit_price, stock_quantity)
            VALUES
            (1, 'Laptop Pro', 'Electronics', 1299.99, 50),
            (2, 'Wireless Mouse', 'Electronics', 29.99, 500),
            (3, 'Running Shoes', 'Sports', 89.99, 200),
            (4, 'Coffee Maker', 'Home', 79.99, 150),
            (5, 'Python Book', 'Books', 39.99, 300)
        """))

        conn.execute(text("""
            INSERT INTO customers (customer_id, first_name, last_name, email, city, segment)
            VALUES
            (1, 'John', 'Doe', 'john@example.com', 'New York', 'Consumer'),
            (2, 'Jane', 'Smith', 'jane@example.com', 'Los Angeles', 'Corporate'),
            (3, 'Bob', 'Wilson', 'bob@example.com', 'Chicago', 'Consumer'),
            (4, 'Alice', 'Brown', 'alice@example.com', 'Houston', 'Small Business'),
            (5, 'Charlie', 'Davis', 'charlie@example.com', 'Phoenix', 'Enterprise')
        """))

        conn.execute(text("""
            INSERT INTO orders (order_id, customer_id, order_date, total_amount, status)
            VALUES
            (1, 1, '2024-01-15', 1329.98, 'Completed'),
            (2, 2, '2024-01-20', 89.99, 'Shipped'),
            (3, 3, '2024-02-01', 119.98, 'Completed'),
            (4, 4, '2024-02-10', 79.99, 'Processing'),
            (5, 5, '2024-03-01', 39.99, 'Completed')
        """))

        conn.execute(text("""
            INSERT INTO sales (sale_id, order_id, product_id, quantity, unit_price, total_price)
            VALUES
            (1, 1, 1, 1, 1299.99, 1299.99),
            (2, 1, 2, 1, 29.99, 29.99),
            (3, 2, 3, 1, 89.99, 89.99),
            (4, 3, 2, 2, 29.99, 59.98),
            (5, 3, 4, 1, 79.99, 79.99),
            (6, 4, 4, 1, 79.99, 79.99),
            (7, 5, 5, 1, 39.99, 39.99)
        """))

        conn.commit()

    return engine


@pytest.fixture
def guardrail() -> GuardrailAgent:
    """Provide a GuardrailAgent instance."""
    return GuardrailAgent()


@pytest.fixture
def sample_state() -> AgentState:
    """Provide a sample AgentState for testing."""
    return AgentState(
        user_question="What are the top 5 selling products?",
        schema_info="TABLE: products (product_id, product_name, category, unit_price)",
        database_dialect="postgresql",
        conversation_history=[],
        retry_count=0,
        max_retries=3,
        execution_times={},
        token_usage={},
        metadata={},
    )


@pytest.fixture
def sample_plan() -> dict[str, Any]:
    """Provide a sample analysis plan."""
    return {
        "understanding": "Find the top 5 best-selling products",
        "analysis_type": "ranking",
        "tables_needed": ["products", "sales"],
        "columns_needed": {
            "products": ["product_id", "product_name"],
            "sales": ["product_id", "quantity"],
        },
        "joins": [{
            "left_table": "products",
            "right_table": "sales",
            "left_column": "product_id",
            "right_column": "product_id",
            "join_type": "INNER",
        }],
        "filters": [],
        "aggregations": [
            {"function": "SUM", "column": "quantity", "alias": "total_sold"},
        ],
        "group_by": ["product_id", "product_name"],
        "order_by": [{"column": "total_sold", "direction": "DESC"}],
        "limit": 5,
        "notes": "",
    }
