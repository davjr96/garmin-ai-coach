import logging
from dataclasses import dataclass
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

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
        elif "googleapis.com" in base_url:
            return "google"
        else:
            return "unknown"

    CONFIGURATIONS: dict[str, ModelConfiguration] = {
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
            name="claude-opus-4-6",
            base_url="https://api.anthropic.com",
        ),
        "claude-opus-thinking": ModelConfiguration(
            name="claude-opus-4-6",
            base_url="https://api.anthropic.com",
        ),
        "claude-haiku-4-5": ModelConfiguration(
            name="claude-haiku-4-5-20251001",
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
        "gemini-3-flash-direct": ModelConfiguration(
            name="gemini-3-flash-preview",
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
        "gemini-2.5-flash-direct": {
            "thinking_budget": 0,
            "log": "Using Gemini 2.5 Flash for {role} (thinking disabled, max_output: 65536)",
        },
        "gemini-3-flash-direct": {
            "thinking_budget": 0,
            "log": "Using Gemini 3 Flash Preview for {role} (thinking disabled, max_output: 65536)",
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

        if provider != "anthropic":
            raise RuntimeError(f"Unsupported provider '{provider}' for model '{model_name}'")

        api_key = config.anthropic_api_key
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic models")

        logger.info("Configuring LLM for role %s with model %s", role.value, final_model_name)

        llm_params: dict[str, Any] = {"model": final_model_name, "api_key": api_key}

        cls._apply_model_config(model_name, role, llm_params)

        return ChatAnthropic(**llm_params)
