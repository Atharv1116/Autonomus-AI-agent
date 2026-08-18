"""
Database connection manager.

Handles SQLAlchemy engine creation, session management, and connection
pooling for PostgreSQL, MySQL, and SQLite. Uses connection URL from
environment configuration.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from config.logging_config import get_logger

logger = get_logger("database.connection")


class DatabaseManager:
    """
    Manages database connections with SQLAlchemy.

    Supports PostgreSQL, MySQL, and SQLite with appropriate
    connection pooling and dialect-specific configuration.
    """

    def __init__(self, database_url: str, echo: bool = False) -> None:
        """
        Initialize the database manager.

        Args:
            database_url: SQLAlchemy connection URL.
            echo: If True, log all SQL statements (debug mode).
        """
        self._database_url = database_url
        self._echo = echo
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        logger.info("DatabaseManager initialized for: %s", self._mask_url(database_url))

    @staticmethod
    def _mask_url(url: str) -> str:
        """Mask password in database URL for safe logging."""
        if "@" in url and ":" in url:
            # Mask the password portion
            parts = url.split("@")
            credentials = parts[0]
            if ":" in credentials:
                scheme_user = credentials.rsplit(":", 1)[0]
                return f"{scheme_user}:****@{parts[-1]}"
        return url

    @property
    def engine(self) -> Engine:
        """Get or create the SQLAlchemy engine (lazy initialization)."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> Engine:
        """
        Create a SQLAlchemy engine with dialect-specific settings.

        Returns:
            Configured SQLAlchemy engine.
        """
        url = self._database_url.lower()

        # --- SQLite Configuration ---
        if url.startswith("sqlite"):
            engine = create_engine(
                self._database_url,
                echo=self._echo,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
            # Enable foreign keys for SQLite
            @event.listens_for(engine, "connect")
            def _set_sqlite_pragma(dbapi_conn: Any, _: Any) -> None:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

            logger.info("SQLite engine created")
            return engine

        # --- PostgreSQL / MySQL Configuration ---
        pool_kwargs: dict[str, Any] = {
            "echo": self._echo,
            "poolclass": QueuePool,
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 1800,
            "pool_pre_ping": True,
        }

        engine = create_engine(self._database_url, **pool_kwargs)
        dialect_name = engine.dialect.name
        logger.info("Engine created for dialect: %s", dialect_name)
        return engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
            )
        return self._session_factory

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Provide a transactional database session.

        Yields:
            A SQLAlchemy Session instance.

        Usage:
            with db_manager.get_session() as session:
                result = session.execute(text("SELECT 1"))
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Session rollback due to error")
            raise
        finally:
            session.close()

    def execute_raw_sql(
        self,
        sql: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a raw SQL query and return results as list of dicts.

        Args:
            sql: The SQL query string.
            params: Optional bind parameters.
            timeout: Query timeout in seconds.

        Returns:
            List of dictionaries, one per row.

        Raises:
            Exception: If query execution fails.
        """
        logger.debug("Executing SQL: %s", sql[:200])

        with self.engine.connect() as conn:
            if timeout and self.engine.dialect.name == "postgresql":
                conn.execute(text(f"SET statement_timeout = {timeout * 1000}"))

            result = conn.execute(text(sql), params or {})

            if result.returns_rows:
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                logger.info("Query returned %d rows", len(rows))
                return rows
            else:
                logger.info("Query executed successfully (no rows returned)")
                return []

    def test_connection(self) -> bool:
        """
        Test the database connection.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test: SUCCESS")
            return True
        except Exception as e:
            logger.error("Database connection test: FAILED - %s", str(e))
            return False

    def get_dialect_name(self) -> str:
        """Return the name of the database dialect (e.g., 'postgresql')."""
        return self.engine.dialect.name

    def dispose(self) -> None:
        """Dispose of the engine and release all connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database engine disposed")
