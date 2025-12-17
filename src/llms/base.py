"""Unified LLM client interfaces and router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.config import AppSettings


class BaseLLMClient:
    """Minimal interface for provider clients."""

    name: str

    def generate(self, messages: List[Dict[str, str]], **_: object) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    name = "mock"

    def generate(self, messages: List[Dict[str, str]], **_: object) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"[mock-response] {last[:120]}"


class OpenAIClient(BaseLLMClient):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    def generate(self, messages: List[Dict[str, str]], **_: object) -> str:
        # Placeholder: would call OpenAI SDK. Kept deterministic for tests.
        last = messages[-1]["content"] if messages else ""
        return f"[openai:{self.model}] {last[:200]}"


class DeepSeekClient(BaseLLMClient):
    name = "deepseek"

    def __init__(self, model: str = "deepseek-chat") -> None:
        self.model = model

    def generate(self, messages: List[Dict[str, str]], **_: object) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"[deepseek:{self.model}] {last[:200]}"


class WanAIClient(BaseLLMClient):
    name = "wanai"

    def __init__(self, model: str = "wanai-chat") -> None:
        self.model = model

    def generate(self, messages: List[Dict[str, str]], **_: object) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"[wanai:{self.model}] {last[:200]}"


@dataclass
class RoutingDecision:
    task_type: str
    provider: str


class LLMRouter:
    """Route requests to providers based on task type."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.clients: Dict[str, BaseLLMClient] = {
            "mock": MockLLMClient(),
            "openai": OpenAIClient(model=settings.model_for("openai")),
            "deepseek": DeepSeekClient(model=settings.model_for("deepseek")),
            "wanai": WanAIClient(model=settings.model_for("wanai")),
        }

    def choose_provider(self, task_type: str) -> RoutingDecision:
        # Simple policy configurable later
        mapping = {
            "dialogue": self.settings.default_provider,
            "retrieval": "deepseek",
            "scoring": "openai",
            "longitudinal": "openai",
        }
        provider = mapping.get(task_type, self.settings.default_provider)
        if provider not in self.clients:
            provider = "mock"
        return RoutingDecision(task_type=task_type, provider=provider)

    def generate(self, task_type: str, messages: List[Dict[str, str]]) -> str:
        decision = self.choose_provider(task_type)
        client = self.clients[decision.provider]
        return client.generate(messages)

    def available_providers(self) -> List[str]:
        return list(self.clients.keys())


__all__ = [
    "BaseLLMClient",
    "MockLLMClient",
    "OpenAIClient",
    "DeepSeekClient",
    "WanAIClient",
    "LLMRouter",
    "RoutingDecision",
]
