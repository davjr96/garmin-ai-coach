import logging
from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from core.config import get_config

from .ai_settings import AgentRole, ai_settings

logger = logging.getLogger(__name__)


@dataclass
class ModelConfiguration:
    name: str
    base_url: str


class ModelSelector:

    @staticmethod
    def _detect_provider(base_url: str) -> str:
        if "anthropic" in base_url:
            return "anthropic"
        elif "openai.com" in base_url:
            return "openai"
        elif "googleapis.com" in base_url:
            return "google"
        else:
            return "unknown"

    CONFIGURATIONS: dict[str, ModelConfiguration] = {
        # OpenAI Models
        "gpt-4o": ModelConfiguration(
            name="gpt-4o",
            base_url="https://api.openai.com/v1",
        ),
        "gpt-4.1": ModelConfiguration(
            name="gpt-4.1",
            base_url="https://api.openai.com/v1",
        ),
        "gpt-4.5": ModelConfiguration(
            name="gpt-4.5-preview",
            base_url="https://api.openai.com/v1",
        ),
        "gpt-4o-mini": ModelConfiguration(
            name="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        ),
        "o1": ModelConfiguration(
            name="o1-preview",
            base_url="https://api.openai.com/v1",
        ),
        "o1-mini": ModelConfiguration(
            name="o1-mini",
            base_url="https://api.openai.com/v1",
        ),
        "o3": ModelConfiguration(
            name="o3",
            base_url="https://api.openai.com/v1",
        ),
        "o3-mini": ModelConfiguration(
            name="o3-mini",
            base_url="https://api.openai.com/v1",
        ),
        "o4-mini": ModelConfiguration(
            name="o4-mini",
            base_url="https://api.openai.com/v1",
        ),
        "gpt-5": ModelConfiguration(
            name="gpt-5.2",
            base_url="https://api.openai.com/v1",
        ),
        "gpt-5.2-pro": ModelConfiguration(
            name="gpt-5.2-pro",
            base_url="https://api.openai.com/v1",
        ),
        "gpt-5-mini": ModelConfiguration(
            name="gpt-5-mini",
            base_url="https://api.openai.com/v1",
        ),
        "gpt-5-search": ModelConfiguration(
            name="gpt-5.2",
            base_url="https://api.openai.com/v1",
        ),
        "gpt-5.2-pro-search": ModelConfiguration(
            name="gpt-5.2-pro",
            base_url="https://api.openai.com/v1",
        ),
        # Anthropic Models
        "claude-4": ModelConfiguration(
            name="claude-sonnet-4-5-20250929",
            base_url="https://api.anthropic.com",
        ),
        "claude-4-thinking": ModelConfiguration(
            name="claude-sonnet-4-5-20250929",
            base_url="https://api.anthropic.com",
        ),
        "claude-sonnet-4.6": ModelConfiguration(
            name="claude-sonnet-4-6",
            base_url="https://api.anthropic.com",
        ),
        "claude-sonnet-4.6-thinking": ModelConfiguration(
            name="claude-sonnet-4-6",
            base_url="https://api.anthropic.com",
        ),
        "claude-opus": ModelConfiguration(
            name="claude-opus-4-1-20250805",
            base_url="https://api.anthropic.com",
        ),
        "claude-opus-thinking": ModelConfiguration(
            name="claude-opus-4-1-20250805",
            base_url="https://api.anthropic.com",
        ),
        "claude-3-haiku": ModelConfiguration(
            name="claude-3-haiku-20240307",
            base_url="https://api.anthropic.com",
        ),
        # Google AI Studio Models (direct)
        "gemini-2.5-pro-direct": ModelConfiguration(
            name="gemini-2.5-pro",
            base_url="https://generativelanguage.googleapis.com",
        ),
        "gemini-2.5-flash-direct": ModelConfiguration(
            name="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com",
        ),
    }

    MODEL_CONFIGS: dict[str, dict[str, Any]] = {
        "claude-opus-thinking": {
            "max_tokens": 32000,
            "thinking": {"type": "enabled", "budget_tokens": 16000},
            "log": "Using extended thinking mode for {role} (max_tokens: 32000, budget_tokens: 16000)",
        },
        "claude-4-thinking": {
            "max_tokens": 64000,
            "thinking": {"type": "enabled", "budget_tokens": 16000},
            "log": "Using extended thinking mode for {role} (max_tokens: 64000, budget_tokens: 16000)",
        },
        "claude-4": {
            "max_tokens": 64000,
            "log": "Using extended output tokens for {role} (max_tokens: 64000)",
        },
        "claude-sonnet-4.6": {
            "max_tokens": 64000,
            "betas": ["context-1m-2025-08-07"],
            "log": "Using Claude Sonnet 4.6 for {role} (max_tokens: 64000, context: 1M)",
        },
        "claude-sonnet-4.6-thinking": {
            "max_tokens": 64000,
            "thinking": {"type": "enabled", "budget_tokens": 16000},
            "betas": ["context-1m-2025-08-07"],
            "log": "Using Claude Sonnet 4.6 with extended thinking for {role} (max_tokens: 64000, budget_tokens: 16000, context: 1M)",
        },
        "claude-opus": {
            "max_tokens": 32000,
            "log": "Using extended output tokens for {role} (max_tokens: 32000)",
        },
        "gpt-5": {
            "use_responses_api": True,
            "reasoning": {"effort": "xhigh"},
            "model_kwargs": {"text": {"verbosity": "high"}},
            "log": "Using GPT-5 with Responses API for {role} (verbosity: high, reasoning_effort: xhigh)",
        },
        "gpt-5.2-pro": {
            "use_responses_api": True,
            "reasoning": {"effort": "xhigh"},
            "model_kwargs": {"text": {"verbosity": "high"}},
            "log": "Using GPT-5.2 Pro with Responses API for {role} (verbosity: high, reasoning_effort: xhigh)",
        },
        "gpt-5-mini": {
            "use_responses_api": True,
            "reasoning": {"effort": "high"},
            "model_kwargs": {"text": {"verbosity": "high"}},
            "log": "Using GPT-5-mini with Responses API for {role} (verbosity: high, reasoning_effort: high)",
        },
        "gpt-5-search": {
            "use_responses_api": True,
            "reasoning": {"effort": "xhigh"},
            "model_kwargs": {
                "text": {"verbosity": "high"},
                "tools": [{"type": "web_search"}],
                "include": ["web_search_call.action.sources"],
            },
            "log": "Using GPT-5.2 with web search + Responses API for {role} (verbosity: high, reasoning_effort: xhigh)",
        },
        "gpt-5.2-pro-search": {
            "use_responses_api": True,
            "reasoning": {"effort": "xhigh"},
            "model_kwargs": {
                "text": {"verbosity": "high"},
                "tools": [{"type": "web_search"}],
                "include": ["web_search_call.action.sources"],
            },
            "log": "Using GPT-5.2 Pro with web search + Responses API for {role} (verbosity: high, reasoning_effort: xhigh)",
        },
        "gemini-2.5-flash-direct": {
            "thinking_budget": 0,
            "log": "Using Gemini 2.5 Flash for {role} (thinking disabled, max_output: 65536)",
        },
        "gemini-2.5-pro-direct": {
            "thinking_budget": 8192,
            "log": "Using Gemini 2.5 Pro for {role} (thinking_budget: 8192, max_output: 65536)",
        },
    }

    @classmethod
    def _apply_model_config(cls, model_name: str, role: AgentRole, llm_params: dict[str, Any]):
        if model_name not in cls.MODEL_CONFIGS:
            return

        config_data = cls.MODEL_CONFIGS[model_name].copy()
        log_msg = config_data.pop("log", None)
        llm_params.update(config_data)
        if log_msg:
            logger.info(str(log_msg).format(role=role.value))

    @classmethod
    def get_llm(cls, role: AgentRole):
        model_name = ai_settings.get_model_for_role(role)
        selected_config = cls.CONFIGURATIONS.get(model_name)
        if not selected_config:
            raise RuntimeError(f"Unknown model '{model_name}' in configuration")
        config = get_config()

        base_url = selected_config.base_url
        final_model_name = selected_config.name

        provider = cls._detect_provider(base_url)

        # Google AI Studio uses its own client — handle early
        if provider == "google":
            google_api_key = config.google_api_key
            if not google_api_key:
                raise RuntimeError("GOOGLE_API_KEY is required for Google AI Studio models")
            logger.info("Configuring LLM for role %s with model %s (Google AI Studio)", role.value, final_model_name)
            google_params: dict[str, Any] = {
                "model": final_model_name,
                "google_api_key": google_api_key,
                "max_output_tokens": 65536,
            }
            cls._apply_model_config(model_name, role, google_params)
            return ChatGoogleGenerativeAI(**google_params)

        key_map = {
            "anthropic": config.anthropic_api_key,
            "openai": config.openai_api_key,
        }

        api_key = key_map.get(provider)
        if not api_key:
            raise RuntimeError(f"{provider.upper()}_API_KEY is required for {provider.title()} models")

        logger.info("Configuring LLM for role %s with model %s", role.value, final_model_name)

        llm_params: dict[str, Any] = {"model": final_model_name, "api_key": api_key}

        cls._apply_model_config(model_name, role, llm_params)

        if provider == "anthropic":
            return ChatAnthropic(**llm_params)

        llm_params["base_url"] = base_url
        return ChatOpenAI(**llm_params)
