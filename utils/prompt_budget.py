"""
Prompt budget manager for LLM requests.

Groq's groq/compound model enforces a hard request-size limit (~8 KB body).
This module provides utilities that trim/compress prompt components so that
the total payload always fits within that budget.

Strategy (applied in order until budget is satisfied):
  1. Filter schema to only the tables mentioned in the plan
  2. Compress the plan JSON (no indentation)
  3. Drop few-shot examples
  4. Truncate conversation history to the last message only
  5. Hard-truncate the schema itself as a last resort

All functions are pure (no side-effects) and return trimmed copies.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Maximum total prompt characters we allow before sending to Groq.
# Groq's compound model fails at extremely low sizes on this endpoint.
# We stay strictly under 3,000 characters to guarantee it never fails.
MAX_PROMPT_CHARS: int = 3_000

# Maximum characters we allow for the schema section of any prompt.
MAX_SCHEMA_CHARS: int = 1_000

# Maximum characters for the plan JSON section.
MAX_PLAN_CHARS: int = 400

# Maximum characters for conversation history.
MAX_HISTORY_CHARS: int = 200


# ---------------------------------------------------------------------------
# Schema filtering
# ---------------------------------------------------------------------------

def filter_schema_for_tables(schema: str, tables_needed: list[str]) -> str:
    """
    Extract only the TABLE blocks needed by the plan from the full schema string.

    The schema produced by SchemaReflector.get_schema_for_llm() looks like:

        DATABASE SCHEMA (SQLITE)
        Total Tables: 7
        ============================================================

        TABLE: customers (~15000 rows)
        ----------------------------------------
          • customer_id: INTEGER [PK] NOT NULL
          ...

        TABLE: orders (~30000 rows)
        ...

    We split on "TABLE:" lines and keep only the ones whose name appears in
    ``tables_needed``.  The header block is always preserved.

    Args:
        schema: Full schema string from SchemaReflector.
        tables_needed: List of table names the planner identified.

    Returns:
        Filtered schema string.  If ``tables_needed`` is empty or no match is
        found, the original schema is returned (truncated to MAX_SCHEMA_CHARS).
    """
    if not tables_needed or not schema:
        return _truncate(schema, MAX_SCHEMA_CHARS)

    normalised = {t.lower().strip() for t in tables_needed}

    # Split into blocks: header + one block per TABLE
    # Each block starts with a line like "\nTABLE: tablename"
    parts = re.split(r'(?=\nTABLE:\s)', schema)

    # Always keep the header (first part before any TABLE block)
    header = parts[0]
    table_blocks: list[str] = []

    for part in parts[1:]:
        # Extract table name from the first line of this block
        first_line = part.strip().split("\n")[0]  # e.g. "TABLE: customers (~15000 rows)"
        name_match = re.match(r'TABLE:\s+(\w+)', first_line, re.IGNORECASE)
        if name_match:
            table_name = name_match.group(1).lower()
            if table_name in normalised:
                table_blocks.append(part)

    if not table_blocks:
        # No match found — return the original, truncated
        return _truncate(schema, MAX_SCHEMA_CHARS)

    filtered = header + "".join(table_blocks)

    # Also append a short relationship summary if present
    rel_match = re.search(r'TABLE RELATIONSHIPS:.*', schema, re.DOTALL)
    if rel_match:
        filtered += "\n\n" + rel_match.group(0)

    return _truncate(filtered, MAX_SCHEMA_CHARS)


# ---------------------------------------------------------------------------
# Plan compression
# ---------------------------------------------------------------------------

def compress_plan(plan: dict[str, Any]) -> str:
    """
    Serialise the plan to compact JSON, keeping only fields the SQL generator
    actually needs.  Drops verbose fields like 'understanding' and 'notes'.

    Args:
        plan: Plan dictionary from PlannerAgent.

    Returns:
        Compact JSON string.
    """
    keep = {
        "analysis_type", "tables_needed", "columns_needed",
        "joins", "filters", "aggregations", "group_by", "order_by", "limit",
    }
    slim = {k: v for k, v in plan.items() if k in keep and v not in (None, [], {}, "")}
    return _truncate(json.dumps(slim, separators=(",", ":")), MAX_PLAN_CHARS)


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------

def compress_history(history: list[dict]) -> str:
    """
    Format conversation history within the character budget.

    Starts from the most recent messages and keeps adding until the budget
    is exhausted.

    Args:
        history: List of {role, content} dicts.

    Returns:
        Formatted string or "No previous conversation."
    """
    if not history:
        return "No previous conversation."

    lines: list[str] = []
    budget = MAX_HISTORY_CHARS

    for msg in reversed(history):
        role = msg.get("role", "user").upper()
        content = str(msg.get("content", ""))[:200]  # cap each message
        line = f"{role}: {content}"
        if budget - len(line) - 1 < 0:
            break
        lines.insert(0, line)
        budget -= len(line) + 1

    return "\n".join(lines) if lines else "No previous conversation."


# ---------------------------------------------------------------------------
# Full prompt budget check
# ---------------------------------------------------------------------------

def fits_in_budget(prompt: str, system_msg: str = "") -> bool:
    """Return True if the combined prompt fits within MAX_PROMPT_CHARS."""
    return len(prompt) + len(system_msg) <= MAX_PROMPT_CHARS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    """Hard-truncate ``text`` to ``limit`` characters, appending an ellipsis."""
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


# Public alias for use in other modules
truncate_text = _truncate
