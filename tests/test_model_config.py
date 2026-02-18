import types

import pytest

from core.config import AIMode, Config
from services.ai import model_config
from services.ai.ai_settings import AgentRole
from services.ai.model_config import ModelSelector


class _StubSettings:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_model_for_role(self, _: AgentRole) -> str:
        return self.model_name


@pytest.mark.parametrize(
    ("model_name", "api_key_field", "expected_client"),
    [
        ("claude-4", "anthropic_api_key", "ChatAnthropic"),
        ("gpt-4o", "openai_api_key", "ChatOpenAI"),
    ],
)
def test_prefers_direct_api_when_key_available(
    monkeypatch, model_name, api_key_field, expected_client
):
    api_key_values = {
        "anthropic_api_key": "sk-ant-api03-test",
        "openai_api_key": "sk-test",
    }
    config_dict = {
        api_key_field: api_key_values[api_key_field],
        "ai_mode": AIMode.STANDARD,
    }
    from typing import Any, cast

    config = Config(**cast("dict[str, Any]", config_dict))
    monkeypatch.setattr(model_config, "get_config", lambda: config)
    monkeypatch.setattr(model_config, "ai_settings", _StubSettings(model_name))

    captured = {}

    def fake_chat_anthropic(**kwargs):
        captured.update(kwargs)
        captured["client"] = "ChatAnthropic"
        return types.SimpleNamespace(**kwargs)

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        captured["client"] = "ChatOpenAI"
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(model_config, "ChatAnthropic", fake_chat_anthropic)
    monkeypatch.setattr(model_config, "ChatOpenAI", fake_chat_openai)

    ModelSelector.get_llm(AgentRole.SUMMARIZER)

    assert captured["api_key"] == api_key_values[api_key_field]
    assert captured["client"] == expected_client
    if expected_client == "ChatOpenAI":
        assert captured["base_url"] == "https://api.openai.com/v1"


def test_missing_direct_key_raises(monkeypatch):
    config = Config(ai_mode=AIMode.STANDARD)
    monkeypatch.setattr(model_config, "get_config", lambda: config)
    monkeypatch.setattr(model_config, "ai_settings", _StubSettings("claude-4"))

    monkeypatch.setattr(model_config, "ChatOpenAI", lambda **_kwargs: None)
    monkeypatch.setattr(model_config, "ChatAnthropic", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is required"):
        ModelSelector.get_llm(AgentRole.SUMMARIZER)


def test_missing_openai_key_raises(monkeypatch):
    config = Config(ai_mode=AIMode.STANDARD)
    monkeypatch.setattr(model_config, "get_config", lambda: config)
    monkeypatch.setattr(model_config, "ai_settings", _StubSettings("gpt-4o"))

    monkeypatch.setattr(model_config, "ChatOpenAI", lambda **_kwargs: None)
    monkeypatch.setattr(model_config, "ChatAnthropic", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        ModelSelector.get_llm(AgentRole.SUMMARIZER)


