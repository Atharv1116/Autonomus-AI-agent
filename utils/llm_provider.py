"""
Modular LLM provider factory.

Supports NVIDIA NIM, OpenAI, Groq, and Ollama through a unified
factory pattern. Each provider returns a LangChain BaseChatModel
instance for seamless integration.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from config.logging_config import get_logger
from config.settings import LLMProvider

logger = get_logger("utils.llm_provider")


class LLMProviderFactory:
    """
    Factory for creating LLM instances from different providers.

    Supports runtime switching between providers without code changes.
    Each provider is lazily imported to avoid unnecessary dependencies.
    """

    _PROVIDER_REGISTRY: dict[LLMProvider, str] = {
        LLMProvider.NVIDIA_NIM: "nvidia_nim",
        LLMProvider.OPENAI: "openai",
        LLMProvider.GROQ: "groq",
        LLMProvider.OLLAMA: "ollama",
    }

    @classmethod
    def create(
        cls,
        provider: LLMProvider | str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        streaming: bool = False,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Create an LLM instance for the specified provider.

        Args:
            provider: LLM provider name or enum.
            model: Model name (uses provider default if None).
            api_key: API key (uses env var if None).
            base_url: Base URL for self-hosted providers.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            streaming: Enable streaming responses.
            **kwargs: Additional provider-specific arguments.

        Returns:
            A LangChain BaseChatModel instance.

        Raises:
            ValueError: If provider is not supported.
            ImportError: If provider dependencies are not installed.
        """
        if isinstance(provider, str):
            try:
                provider = LLMProvider(provider)
            except ValueError:
                raise ValueError(
                    f"Unsupported LLM provider: '{provider}'. "
                    f"Choose from: {[p.value for p in LLMProvider]}"
                )

        method_name = f"_create_{cls._PROVIDER_REGISTRY[provider]}"
        create_method = getattr(cls, method_name)

        logger.info(
            "Creating LLM: provider=%s, model=%s, temperature=%.1f, streaming=%s",
            provider.value, model, temperature, streaming,
        )

        return create_method(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def _create_nvidia_nim(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        streaming: bool = False,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create NVIDIA NIM LLM instance."""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError:
            raise ImportError(
                "langchain-nvidia-ai-endpoints is required for NVIDIA NIM. "
                "Install with: pip install langchain-nvidia-ai-endpoints"
            )

        params: dict[str, Any] = {
            "model": model or "meta/llama-3.1-70b-instruct",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "streaming": streaming,
        }

        if api_key:
            params["nvidia_api_key"] = api_key

        params.update(kwargs)
        logger.info("NVIDIA NIM LLM created: %s", params["model"])
        return ChatNVIDIA(**params)

    @staticmethod
    def _create_openai(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        streaming: bool = False,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create OpenAI LLM instance."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenAI. "
                "Install with: pip install langchain-openai"
            )

        params: dict[str, Any] = {
            "model": model or "gpt-4o",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "streaming": streaming,
        }

        if api_key:
            params["openai_api_key"] = api_key

        params.update(kwargs)
        logger.info("OpenAI LLM created: %s", params["model"])
        return ChatOpenAI(**params)

    @staticmethod
    def _create_groq(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        streaming: bool = False,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create Groq LLM instance."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq is required for Groq. "
                "Install with: pip install langchain-groq"
            )

        params: dict[str, Any] = {
            "model_name": model or "llama-3.1-70b-versatile",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "streaming": streaming,
        }

        if api_key:
            params["groq_api_key"] = api_key

        params.update(kwargs)
        logger.info("Groq LLM created: %s", params["model_name"])
        return ChatGroq(**params)

    @staticmethod
    def _create_ollama(
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        streaming: bool = False,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create Ollama LLM instance."""
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-community is required for Ollama. "
                "Install with: pip install langchain-community"
            )

        params: dict[str, Any] = {
            "model": model or "llama3.1",
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        if base_url:
            params["base_url"] = base_url

        params.update(kwargs)
        logger.info("Ollama LLM created: %s", params["model"])
        return ChatOllama(**params)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Return list of all supported provider names."""
        return [p.value for p in LLMProvider]

    @classmethod
    def validate_provider(cls, provider: str) -> bool:
        """Check if a provider name is valid."""
        try:
            LLMProvider(provider)
            return True
        except ValueError:
            return False
