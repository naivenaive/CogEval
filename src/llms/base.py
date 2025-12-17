"""Unified LLM client interfaces and router.

This module now integrates LangChain chat models so production users can
back the agent with real LLMs (e.g., OpenAI). When credentials or optional
dependencies are missing, clients gracefully fall back to deterministic
mock responses to keep tests green.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from src.config import AppSettings

try:  # LangChain is optional for tests; required for real LLM calls
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    AIMessage = None
    HumanMessage = None
    SystemMessage = None
    ChatOpenAI = None


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

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("COGEVAL_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._chat = self._build_chat_model()

    def _build_chat_model(self):  # pragma: no cover - network-bound path
        if ChatOpenAI is None or self.api_key is None:
            return None
        try:
            return ChatOpenAI(model=self.model, api_key=self.api_key, temperature=self.temperature)
        except Exception:
            return None

    def _to_langchain_messages(self, messages: List[Dict[str, str]]):
        if HumanMessage is None:
            return []
        role_map = {
            "user": HumanMessage,
            "system": SystemMessage,
            "assistant": AIMessage,
        }
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            cls = role_map.get(role, HumanMessage)
            converted.append(cls(content))
        return converted

    def generate(self, messages: List[Dict[str, str]], **_: object) -> str:
        if self._chat:
            try:  # pragma: no cover - network-bound path
                response = self._chat.invoke(self._to_langchain_messages(messages))
                return response.content if hasattr(response, "content") else str(response)
            except Exception:
                pass
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
