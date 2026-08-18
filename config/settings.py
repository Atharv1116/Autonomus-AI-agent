"""
Centralized application settings.

Loads configuration from environment variables and .env file
using pydantic-settings for validation and type coercion.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    NVIDIA_NIM = "nvidia_nim"
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"


class DatabaseDialect(str, Enum):
    """Supported database dialects."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/analyst_db",
        description="SQLAlchemy database connection URL",
    )

    # --- LLM Provider ---
    llm_provider: LLMProvider = Field(
        default=LLMProvider.NVIDIA_NIM,
        description="Active LLM provider",
    )

    # --- NVIDIA NIM ---
    nvidia_api_key: Optional[str] = Field(default=None, description="NVIDIA NIM API key")
    nvidia_model: str = Field(default="meta/llama-3.1-70b-instruct", description="NVIDIA NIM model name")

    # --- OpenAI ---
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")

    # --- Groq ---
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    groq_model: str = Field(default="groq/compound", description="Groq model name")

    # --- Ollama ---
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama3.1", description="Ollama model name")

    # --- Application ---
    log_level: str = Field(default="INFO", description="Logging level")
    max_query_rows: int = Field(default=10000, description="Maximum rows returned per query")
    query_timeout_seconds: int = Field(default=30, description="SQL query timeout in seconds")
    cache_ttl_seconds: int = Field(default=3600, description="Cache time-to-live in seconds")
    max_retries: int = Field(default=3, description="Maximum SQL generation retries")

    # --- Streamlit ---
    streamlit_theme: str = Field(default="dark", description="Streamlit theme")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper

    @property
    def database_dialect(self) -> DatabaseDialect:
        """Detect database dialect from connection URL."""
        url_lower = self.database_url.lower()
        if url_lower.startswith("postgresql") or url_lower.startswith("postgres"):
            return DatabaseDialect.POSTGRESQL
        elif url_lower.startswith("mysql"):
            return DatabaseDialect.MYSQL
        elif url_lower.startswith("sqlite"):
            return DatabaseDialect.SQLITE
        else:
            raise ValueError(f"Unsupported database dialect in URL: {self.database_url}")

    def get_active_api_key(self) -> Optional[str]:
        """Return the API key for the currently active LLM provider."""
        key_map = {
            LLMProvider.NVIDIA_NIM: self.nvidia_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.OLLAMA: None,  # Ollama doesn't need an API key
        }
        return key_map.get(self.llm_provider)

    def get_active_model(self) -> str:
        """Return the model name for the currently active LLM provider."""
        model_map = {
            LLMProvider.NVIDIA_NIM: self.nvidia_model,
            LLMProvider.OPENAI: self.openai_model,
            LLMProvider.GROQ: self.groq_model,
            LLMProvider.OLLAMA: self.ollama_model,
        }
        return model_map[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns a singleton Settings instance. The first call loads from
    environment; subsequent calls return the cached instance.
    """
    return Settings()


def reload_settings(**overrides) -> Settings:
    """
    Create a new Settings instance with optional overrides.

    Useful for runtime reconfiguration (e.g., changing LLM provider
    from the Streamlit sidebar).

    Args:
        **overrides: Key-value pairs to override default settings.

    Returns:
        A new Settings instance with the specified overrides.
    """
    # Clear the cached instance
    get_settings.cache_clear()

    # Set overrides as environment variables so pydantic picks them up
    for key, value in overrides.items():
        os.environ[key.upper()] = str(value)

    return get_settings()
