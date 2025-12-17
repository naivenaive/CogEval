"""Cognitive agent orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from src.config import AppSettings
from src.llms.base import LLMRouter, MockLLMClient
from src.memory.atom_memory import AtomMemory, MemoryAtom
from src.rag.agentic_rag import AgenticRAG, Retrieval


@dataclass
class ProviderConfig:
    name: str
    call_fn: Callable[[str, Sequence[str]], str]
    model: str


@dataclass
class AgentConfig:
    providers: List[ProviderConfig]
    default_provider: str


@dataclass
class TurnPlan:
    intent: str
    needs_tool: bool
    retrieval_hints: List[str] = field(default_factory=list)
    cognitive_domain: str = "executive"


class CognitiveAgent:
    """Simplified agent that demonstrates planning + retrieval + generation."""

    def __init__(self, rag: AgenticRAG, memory: AtomMemory, router: LLMRouter) -> None:
        self.rag = rag
        self.memory = memory
        self.router = router

    def plan(self, message: str) -> TurnPlan:
        needs_tool = "图形" in message or "figure" in message.lower()
        hints = ["cognitive assessment", "executive function", "scoring rubric"]
        domain = "visuospatial" if needs_tool else "language"
        return TurnPlan(intent="assess_cognition", needs_tool=needs_tool, retrieval_hints=hints, cognitive_domain=domain)

    def _compose_prompt(
        self, user_message: str, retrievals: List[Retrieval], events: List[MemoryAtom], plan: TurnPlan
    ) -> List[Dict[str, str]]:
        context_lines = [f"Source: {r.source} -> {r.text}" for r in retrievals]
        memory_lines = [f"[{e.atom_type}] {e.raw_content}" for e in events]
        content = "\n".join(
            [
                "You are a cognitive assessment assistant.",
                "Ground your answers in provided sources.",
                f"Target domain: {plan.cognitive_domain}.",
                "\nMemories:",
                *memory_lines,
                "\nDocuments:",
                *context_lines,
                "\nUser:",
                user_message,
            ]
        )
        return [{"role": "user", "content": content}]

    def run_turn(self, user_id: str, session_id: str, message: str) -> str:
        plan = self.plan(message)
        retrievals = self.rag.agentic_retrieve(plan.retrieval_hints)
        memories = self.memory.recall(user_id=user_id, limit=4)
        prompt = self._compose_prompt(message, retrievals, memories, plan)
        response = self.router.generate(task_type="dialogue", messages=prompt)
        self.memory.store_dialogue(user_id=user_id, session_id=session_id, role="user", content=message)
        self.memory.store_dialogue(user_id=user_id, session_id=session_id, role="assistant", content=response)
        if plan.needs_tool:
            # log a placeholder task atom for downstream processing
            self.memory.store_task_result(
                user_id=user_id,
                session_id=session_id,
                domain=plan.cognitive_domain,
                content="Pending external task result",
                score=70.0,
                confidence=0.5,
                source="mcp_stub",
            )
        return response


def fake_call_fn(prompt: str, _: Sequence[str]) -> str:
    """Fake model call for tests and demos."""
    return "[demo-response]" + prompt[:40]


def build_demo_agent() -> CognitiveAgent:
    settings = AppSettings()
    rag = AgenticRAG(index_path=settings.rag_index_path)
    memory = AtomMemory(db_path=settings.database_path)
    router = LLMRouter(settings=settings)
    return CognitiveAgent(rag=rag, memory=memory, router=router)


__all__ = [
    "AgentConfig",
    "CognitiveAgent",
    "ProviderConfig",
    "TurnPlan",
    "build_demo_agent",
    "fake_call_fn",
]
