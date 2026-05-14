import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin async wrapper around the Anthropic SDK.
    Uses ephemeral prompt caching on system prompts to reduce cost when the
    same system context is reused across multiple calls in one tick.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self._usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_calls": 0,
        }

    async def ask(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        cache_system: bool = True,
    ) -> str:
        system_block: dict[str, Any] = {"type": "text", "text": system}
        if cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}

        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[system_block],  # type: ignore[list-item]
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIError as exc:
            logger.warning("LLM call failed: %s — returning empty string", exc)
            return ""

        usage = response.usage
        self._usage["input_tokens"] += usage.input_tokens
        self._usage["output_tokens"] += usage.output_tokens
        self._usage["cache_read_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        self._usage["cache_write_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self._usage["total_calls"] += 1

        return response.content[0].text  # type: ignore[union-attr]

    def usage_report(self) -> dict[str, int]:
        return dict(self._usage)

    def reset_usage(self) -> None:
        for k in self._usage:
            self._usage[k] = 0
