"""Tests for LLM client dispatch logic."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.llm_client import LLMClient


def test_chat_requires_api_key() -> None:
    client = LLMClient(provider="openai")
    with pytest.raises(ValueError, match="api_key"):
        client.chat([{"role": "user", "content": "hi"}], api_key="")


@patch("app.core.llm_client.OpenAI")
def test_openai_dispatch(mock_openai_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="hello"))
    ]

    client = LLMClient(provider="openai")
    result = client.chat([{"role": "user", "content": "hi"}], api_key="sk-test")
    assert result == "hello"
    mock_openai_cls.assert_called_once_with(api_key="sk-test")


@patch("app.core.llm_client.Anthropic")
def test_anthropic_dispatch(mock_anthropic_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    text_block = MagicMock(type="text", text="bonjour")
    mock_client.messages.create.return_value.content = [text_block]

    client = LLMClient(provider="anthropic")
    result = client.chat([{"role": "user", "content": "hi"}], api_key="sk-test")
    assert result == "bonjour"
