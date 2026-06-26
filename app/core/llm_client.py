"""Provider-agnostic LLM client supporting OpenAI and Anthropic."""

from __future__ import annotations

from typing import Any, Literal

from anthropic import Anthropic
from openai import OpenAI

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

Provider = Literal["openai", "anthropic"]


class LLMClient:
    """Dispatch chat completions to OpenAI or Anthropic based on provider.

    API keys are accepted at call time and are never persisted or logged.
    """

    def __init__(self, provider: Provider | None = None, default_model: str | None = None) -> None:
        settings = get_settings()
        self._provider: Provider = provider or settings.llm_provider
        self._default_model = default_model or (
            settings.llm_model if self._provider == "openai" else settings.anthropic_model
        )

    @property
    def provider(self) -> Provider:
        return self._provider

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        api_key: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return assistant text.

        Args:
            messages: OpenAI-style role/content message list.
            api_key: Provider API key (required; never stored on the instance).
            model: Optional model override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            Assistant message content as a string.
        """
        if not api_key:
            raise ValueError("api_key is required for LLM calls")

        resolved_model = model or self._default_model
        logger.debug("LLM chat request provider=%s model=%s", self._provider, resolved_model)

        if self._provider == "openai":
            return self._chat_openai(
                messages,
                api_key=api_key,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        if self._provider == "anthropic":
            return self._chat_anthropic(
                messages,
                api_key=api_key,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        raise ValueError(f"Unsupported LLM provider: {self._provider}")

    def _chat_openai(
        self,
        messages: list[dict[str, str]],
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def _chat_anthropic(
        self,
        messages: list[dict[str, str]],
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> str:
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            system="\n".join(system_parts) if system_parts else None,
            messages=anthropic_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "".join(text_blocks)
