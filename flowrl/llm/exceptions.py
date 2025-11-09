"""Shared exception types for Flow LLM orchestration."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ContinueSkillDecision:
    """Structured payload describing a continue-training directive."""

    action: str
    skill_name: str
    extra_timesteps: int = 0
    metadata: Dict[str, Any] | None = None


class ContinueSkillException(Exception):
    """Signal that the LLM elected to continue training an existing skill."""

    def __init__(self, decision: Dict[str, Any]):
        self.decision_dict = dict(decision or {})
        # Normalize defaults
        self.decision = ContinueSkillDecision(
            action=self.decision_dict.get("action", "continue_training"),
            skill_name=self.decision_dict.get("skill_name", ""),
            extra_timesteps=int(self.decision_dict.get("extra_timesteps", 0) or 0),
            metadata=self.decision_dict.get("metadata"),
        )
        message = self.decision_dict.get(
            "message",
            f"Continue training skill '{self.decision.skill_name}'",
        )
        super().__init__(message)


__all__ = ["ContinueSkillException", "ContinueSkillDecision"]
