from dataclasses import dataclass, field
from enum import Enum

from core.config import AIMode, get_config


class AgentRole(Enum):
    SUMMARIZER = "summarizer"
    METRICS_EXPERT = "metrics_expert"
    PHYSIOLOGY_EXPERT = "physiology_expert"
    ACTIVITY_EXPERT = "activity_expert"
    SYNTHESIS = "synthesis"
    WORKOUT = "workout"
    SEASON_PLANNER = "season_planner"
    FORMATTER = "formatter"
    CHATBOT = "chatbot"


@dataclass
class AISettings:
    mode: AIMode

    model_assignments: dict[AIMode, dict[AgentRole, str]] = field(
        default_factory=lambda: {
            AIMode.STANDARD: {
                AgentRole.SUMMARIZER: "gemini-3-flash-direct",
                AgentRole.FORMATTER: "claude-sonnet-4.6",
                AgentRole.METRICS_EXPERT: "claude-sonnet-4.6",
                AgentRole.PHYSIOLOGY_EXPERT: "claude-sonnet-4.6",
                AgentRole.ACTIVITY_EXPERT: "claude-sonnet-4.6",
                AgentRole.SYNTHESIS: "claude-sonnet-4.6",
                AgentRole.WORKOUT: "claude-sonnet-4.6",
                AgentRole.SEASON_PLANNER: "claude-sonnet-4.6",
                AgentRole.CHATBOT: "claude-sonnet-4.6",
            },
            AIMode.COST_EFFECTIVE: {
                AgentRole.SUMMARIZER: "claude-haiku-4-5",
                AgentRole.FORMATTER: "claude-haiku-4-5",
                AgentRole.METRICS_EXPERT: "claude-haiku-4-5",
                AgentRole.PHYSIOLOGY_EXPERT: "claude-haiku-4-5",
                AgentRole.ACTIVITY_EXPERT: "claude-haiku-4-5",
                AgentRole.SYNTHESIS: "claude-haiku-4-5",
                AgentRole.WORKOUT: "claude-haiku-4-5",
                AgentRole.SEASON_PLANNER: "claude-haiku-4-5",
                AgentRole.CHATBOT: "claude-haiku-4-5",
            },
            AIMode.DEVELOPMENT: {
                AgentRole.SUMMARIZER: "gemini-3-flash-direct",
                AgentRole.FORMATTER: "claude-sonnet-4.6",
                AgentRole.METRICS_EXPERT: "claude-sonnet-4.6",
                AgentRole.PHYSIOLOGY_EXPERT: "claude-sonnet-4.6",
                AgentRole.ACTIVITY_EXPERT: "claude-sonnet-4.6",
                AgentRole.SYNTHESIS: "claude-sonnet-4.6",
                AgentRole.WORKOUT: "claude-sonnet-4.6",
                AgentRole.SEASON_PLANNER: "claude-sonnet-4.6",
                AgentRole.CHATBOT: "claude-sonnet-4.6",
            },
            AIMode.PRO: {
                AgentRole.SUMMARIZER: "gemini-3-flash-direct",
                AgentRole.FORMATTER: "claude-sonnet-4.6",
                AgentRole.METRICS_EXPERT: "claude-opus",
                AgentRole.PHYSIOLOGY_EXPERT: "claude-opus",
                AgentRole.ACTIVITY_EXPERT: "claude-opus",
                AgentRole.SYNTHESIS: "claude-opus",
                AgentRole.WORKOUT: "claude-opus",
                AgentRole.SEASON_PLANNER: "claude-opus",
                AgentRole.CHATBOT: "claude-opus",
            },
        }
    )

    def get_model_for_role(self, role: AgentRole) -> str:
        return self.model_assignments[self.mode][role]

    @classmethod
    def load_settings(cls) -> "AISettings":
        return cls(mode=get_config().ai_mode)

    def reload(self) -> None:
        self.mode = get_config().ai_mode


# Global settings instance
ai_settings = AISettings.load_settings()
