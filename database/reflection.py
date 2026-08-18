"""
Database schema reflection.

Automatically discovers tables, columns, types, primary keys,
and foreign key relationships using SQLAlchemy MetaData reflection.
Provides LLM-readable schema descriptions for prompt injection.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from sqlalchemy import MetaData, inspect
from sqlalchemy.engine import Engine

from config.logging_config import get_logger

logger = get_logger("database.reflection")


class SchemaReflector:
    """
    Reflects database schema and provides structured descriptions.

    Caches reflected metadata with TTL to avoid repeated introspection.
    Outputs schema in a format optimized for LLM consumption.
    """

    def __init__(self, engine: Engine, cache_ttl: int = 3600) -> None:
        """
        Initialize the schema reflector.

        Args:
            engine: SQLAlchemy engine connected to the target database.
            cache_ttl: Cache time-to-live in seconds.
        """
        self._engine = engine
        self._cache_ttl = cache_ttl
        self._metadata: Optional[MetaData] = None
        self._cached_schema: Optional[dict[str, Any]] = None
        self._cache_timestamp: float = 0.0
        self._schema_hash: str = ""

    def reflect(self, force: bool = False) -> dict[str, Any]:
        """
        Reflect the database schema.

        Args:
            force: If True, bypass cache and re-reflect.

        Returns:
            Dictionary with complete schema information.
        """
        now = time.time()
        if (
            not force
            and self._cached_schema is not None
            and (now - self._cache_timestamp) < self._cache_ttl
        ):
            logger.debug("Returning cached schema (age: %.0fs)", now - self._cache_timestamp)
            return self._cached_schema

        logger.info("Reflecting database schema...")
        self._metadata = MetaData()
        self._metadata.reflect(bind=self._engine)

        schema = self._build_schema_dict()
        self._cached_schema = schema
        self._cache_timestamp = now
        self._schema_hash = self._compute_hash(schema)

        logger.info(
            "Schema reflected: %d tables, hash=%s",
            len(schema["tables"]),
            self._schema_hash[:8],
        )
        return schema

    def _build_schema_dict(self) -> dict[str, Any]:
        """
        Build a structured dictionary from reflected metadata.

        Returns:
            Schema dictionary with tables, columns, keys, and relationships.
        """
        inspector = inspect(self._engine)
        tables_info: list[dict[str, Any]] = []

        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": str(col.get("default", "")) if col.get("default") else None,
                    "primary_key": False,  # Will be updated below
                })

            # Mark primary key columns
            pk_constraint = inspector.get_pk_constraint(table_name)
            pk_columns = set(pk_constraint.get("constrained_columns", []))
            for col in columns:
                if col["name"] in pk_columns:
                    col["primary_key"] = True

            # Foreign keys
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"],
                })

            # Indexes
            indexes = []
            for idx in inspector.get_indexes(table_name):
                indexes.append({
                    "name": idx["name"],
                    "columns": idx["column_names"],
                    "unique": idx.get("unique", False),
                })

            # Row count estimate
            try:
                with self._engine.connect() as conn:
                    from sqlalchemy import text
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = result.scalar()
            except Exception:
                row_count = None

            tables_info.append({
                "name": table_name,
                "columns": columns,
                "primary_keys": list(pk_columns),
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "row_count": row_count,
            })

        return {
            "dialect": self._engine.dialect.name,
            "tables": tables_info,
            "table_names": [t["name"] for t in tables_info],
            "total_tables": len(tables_info),
        }

    def get_schema_for_llm(self, force: bool = False) -> str:
        """
        Generate an LLM-optimized text representation of the schema.

        This format is designed to be injected into prompts for SQL
        generation, providing maximum context with minimal tokens.

        Args:
            force: If True, bypass cache and re-reflect.

        Returns:
            Human-readable schema string.
        """
        schema = self.reflect(force=force)
        lines: list[str] = []

        lines.append(f"DATABASE SCHEMA ({schema['dialect'].upper()})")
        lines.append(f"Total Tables: {schema['total_tables']}")
        lines.append("=" * 60)

        for table in schema["tables"]:
            row_info = f" (~{table['row_count']:,} rows)" if table["row_count"] else ""
            lines.append(f"\nTABLE: {table['name']}{row_info}")
            lines.append("-" * 40)

            for col in table["columns"]:
                pk_marker = " [PK]" if col["primary_key"] else ""
                nullable = " NULL" if col["nullable"] and not col["primary_key"] else " NOT NULL"
                lines.append(f"  • {col['name']}: {col['type']}{pk_marker}{nullable}")

            if table["foreign_keys"]:
                lines.append("  Foreign Keys:")
                for fk in table["foreign_keys"]:
                    lines.append(
                        f"    → {', '.join(fk['constrained_columns'])} "
                        f"REFERENCES {fk['referred_table']}({', '.join(fk['referred_columns'])})"
                    )

        lines.append("\n" + "=" * 60)

        # Add relationship summary
        lines.append("\nTABLE RELATIONSHIPS:")
        for table in schema["tables"]:
            for fk in table["foreign_keys"]:
                lines.append(
                    f"  {table['name']}.{', '.join(fk['constrained_columns'])} "
                    f"→ {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                )

        return "\n".join(lines)

    def get_table_names(self) -> list[str]:
        """Return list of all table names in the database."""
        schema = self.reflect()
        return schema["table_names"]

    def get_column_names(self, table_name: str) -> list[str]:
        """
        Return list of column names for a specific table.

        Args:
            table_name: Name of the table.

        Returns:
            List of column names.
        """
        schema = self.reflect()
        for table in schema["tables"]:
            if table["name"] == table_name:
                return [col["name"] for col in table["columns"]]
        return []

    def get_all_columns(self) -> dict[str, list[str]]:
        """
        Return a mapping of table names to their column names.

        Returns:
            Dictionary mapping table_name -> list of column names.
        """
        schema = self.reflect()
        return {
            table["name"]: [col["name"] for col in table["columns"]]
            for table in schema["tables"]
        }

    def search_columns(self, query: str) -> list[dict[str, str]]:
        """
        Semantic search for columns matching a natural language query.

        Performs fuzzy matching of the query against table and column names.

        Args:
            query: Natural language search term.

        Returns:
            List of matching columns with table context.
        """
        schema = self.reflect()
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        matches: list[dict[str, str]] = []

        for table in schema["tables"]:
            table_name = table["name"].lower()

            for col in table["columns"]:
                col_name = col["name"].lower()
                score = 0

                # Exact match
                if query_lower in col_name or query_lower in table_name:
                    score += 10

                # Partial term matching
                for term in query_terms:
                    if term in col_name:
                        score += 5
                    if term in table_name:
                        score += 3

                # Common synonyms
                synonyms = {
                    "revenue": ["amount", "total", "price", "sales", "income"],
                    "customer": ["client", "buyer", "user"],
                    "product": ["item", "sku", "goods"],
                    "date": ["time", "created", "updated", "timestamp"],
                    "city": ["location", "region", "area", "address"],
                    "name": ["title", "label", "description"],
                    "quantity": ["qty", "count", "number", "amount"],
                }

                for key, syns in synonyms.items():
                    if key in query_terms:
                        for syn in syns:
                            if syn in col_name or syn in table_name:
                                score += 4

                if score > 0:
                    matches.append({
                        "table": table["name"],
                        "column": col["name"],
                        "type": col["type"],
                        "score": score,
                    })

        # Sort by score descending
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:20]

    @property
    def schema_hash(self) -> str:
        """Return the hash of the current schema (for cache invalidation)."""
        if not self._schema_hash:
            self.reflect()
        return self._schema_hash

    @staticmethod
    def _compute_hash(schema: dict[str, Any]) -> str:
        """Compute a stable hash of the schema dictionary."""
        # Use table names + column names for hash stability
        content = ""
        for table in sorted(schema["tables"], key=lambda t: t["name"]):
            content += table["name"]
            for col in sorted(table["columns"], key=lambda c: c["name"]):
                content += f"{col['name']}{col['type']}"
        return hashlib.sha256(content.encode()).hexdigest()
