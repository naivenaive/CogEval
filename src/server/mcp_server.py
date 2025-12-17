"""MCP-like server exposing assessment tasks and chat endpoints."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.agents.cognitive_agent import build_demo_agent
from src.config import get_logger
from src.rag.agentic_rag import AgenticRAG

LOGGER = get_logger(__name__)
app = FastAPI(title="CogEval API")
agent = build_demo_agent()


class Response(BaseModel):
    success: bool
    data: object | None
    error: str | None


@dataclass
class Task:
    task_id: str
    task_type: str
    prompt: str
    state: str = "CREATED"
    user_answer: Optional[str] = None
    score: Optional[float] = None
    domain: str = "executive"

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "state": self.state,
            "user_answer": self.user_answer,
            "score": self.score,
            "domain": self.domain,
        }


tasks: Dict[str, Task] = {}


class ChatMessage(BaseModel):
    user_id: str
    session_id: str
    message: str


@app.post("/api/chat/start", response_model=Response)
def chat_start(payload: ChatMessage) -> Response:
    agent.memory.store_dialogue(payload.user_id, payload.session_id, role="system", content="session-start")
    return Response(success=True, data={"session_id": payload.session_id}, error=None)


@app.post("/api/chat/message", response_model=Response)
def chat_message(payload: ChatMessage) -> Response:
    reply = agent.run_turn(user_id=payload.user_id, session_id=payload.session_id, message=payload.message)
    return Response(success=True, data={"reply": reply}, error=None)


@app.post("/api/chat/end", response_model=Response)
def chat_end(payload: ChatMessage) -> Response:
    agent.memory.store_dialogue(payload.user_id, payload.session_id, role="system", content="session-end")
    return Response(success=True, data={"status": "ended"}, error=None)


class IngestRequest(BaseModel):
    paths: List[str]


@app.post("/api/rag/ingest", response_model=Response)
def rag_ingest(req: IngestRequest) -> Response:
    try:
        agent.rag.ingest_documents(req.paths)
        return Response(success=True, data={"count": len(agent.rag.docs)}, error=None)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Ingest failed")
        return Response(success=False, data=None, error=str(exc))


@app.get("/api/rag/status", response_model=Response)
def rag_status() -> Response:
    return Response(success=True, data={"documents": len(agent.rag.docs)}, error=None)


@app.get("/api/memory/user/{user_id}/timeline", response_model=Response)
def memory_timeline(user_id: str) -> Response:
    atoms = [asdict(atom) for atom in agent.memory.recall(user_id=user_id, limit=50)]
    return Response(success=True, data={"atoms": atoms}, error=None)


class TaskCreateRequest(BaseModel):
    task_type: str


@app.post("/api/mcp/task/create", response_model=Response)
def create_task(req: TaskCreateRequest) -> Response:
    prompt_map = {
        "visual_reasoning": "Please choose the shape that completes the pattern.",
        "digit_span": "Repeat the following numbers backward: 7 2 9.",
        "word_recall": "Remember these words and repeat after delay: tree, house, river.",
    }
    task_id = str(uuid.uuid4())
    prompt = prompt_map.get(req.task_type, "Follow the instruction")
    task = Task(task_id=task_id, task_type=req.task_type, prompt=prompt, state="PRESENTED")
    tasks[task_id] = task
    return Response(success=True, data=task.to_dict(), error=None)


class TaskAnswer(BaseModel):
    answer: str


@app.get("/api/mcp/task/{task_id}", response_model=Response)
def get_task(task_id: str) -> Response:
    task = tasks.get(task_id)
    if not task:
        return Response(success=False, data=None, error="not found")
    return Response(success=True, data=task.to_dict(), error=None)


@app.post("/api/mcp/task/{task_id}/answer", response_model=Response)
def answer_task(task_id: str, payload: TaskAnswer) -> Response:
    task = tasks.get(task_id)
    if not task:
        return Response(success=False, data=None, error="not found")
    task.user_answer = payload.answer
    task.state = "ANSWERED"
    task.score = 80.0 if payload.answer else 50.0
    task.state = "SCORED"
    return Response(success=True, data=task.to_dict(), error=None)


__all__ = ["app", "Task", "Response"]
