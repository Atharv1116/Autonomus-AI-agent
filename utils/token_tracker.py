"""
Token usage tracking.

Tracks prompt and completion tokens across agent steps
for cost estimation and monitoring.
"""

from __future__ import annotations

from typing import Any, Optional

from config.logging_config import get_logger

logger = get_logger("utils.token_tracker")


class TokenTracker:
    """
    Tracks token usage across all agent invocations.

    Accumulates prompt/completion tokens per step and provides
    aggregate totals for the entire query pipeline.
    """

    def __init__(self) -> None:
        """Initialize the token tracker."""
        self._steps: list[dict[str, Any]] = []
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def record(
        self,
        step_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "",
    ) -> None:
        """
        Record token usage for a single agent step.

        Args:
            step_name: Name of the agent step (e.g., 'planner', 'sql_generator').
            prompt_tokens: Number of prompt/input tokens.
            completion_tokens: Number of completion/output tokens.
            model: Model name used for this step.
        """
        total = prompt_tokens + completion_tokens
        self._steps.append({
            "step": step_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "model": model,
        })
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        logger.debug(
            "Token usage [%s]: prompt=%d, completion=%d, total=%d",
            step_name, prompt_tokens, completion_tokens, total,
        )

    def record_from_response(self, step_name: str, response: Any, model: str = "") -> None:
        """
        Extract and record token usage from a LangChain response.

        Handles various response formats from different LLM providers.

        Args:
            step_name: Name of the agent step.
            response: LangChain AIMessage or response object.
            model: Model name used.
        """
        prompt_tokens = 0
        completion_tokens = 0

        # Try to extract usage from response_metadata
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if isinstance(metadata, dict):
                # OpenAI format
                usage = metadata.get("token_usage", metadata.get("usage", {}))
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

        # Try usage_metadata (newer LangChain format)
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            if isinstance(usage, dict):
                prompt_tokens = usage.get("input_tokens", prompt_tokens)
                completion_tokens = usage.get("output_tokens", completion_tokens)

        self.record(
            step_name=step_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all steps."""
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens used."""
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens used."""
        return self._total_completion_tokens

    @property
    def step_breakdown(self) -> list[dict[str, Any]]:
        """Return per-step token breakdown."""
        return self._steps.copy()

    def get_summary(self) -> dict[str, Any]:
        """
        Return a complete usage summary.

        Returns:
            Dictionary with total and per-step token usage.
        """
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self.total_tokens,
            "num_steps": len(self._steps),
            "steps": self._steps.copy(),
        }

    def reset(self) -> None:
        """Reset all tracked usage."""
        self._steps.clear()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        logger.debug("Token tracker reset")
