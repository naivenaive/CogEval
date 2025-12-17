"""Lightweight FastAPI stub for offline testing."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Callable, Dict, Tuple


class FastAPI:
    def __init__(self, title: str | None = None) -> None:
        self.title = title
        self.routes: Dict[Tuple[str, str], Callable[..., Any]] = {}

    def get(self, path: str, response_model: Any | None = None):  # noqa: ANN401
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes[("GET", path)] = func
            return func

        return decorator

    def post(self, path: str, response_model: Any | None = None):  # noqa: ANN401
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.routes[("POST", path)] = func
            return func

        return decorator


class Form:
    def __init__(self, default: Any | None = None) -> None:  # noqa: ANN401
        self.default = default


# responses module shim
class HTMLResponse(str):
    def __new__(cls, content: str, status_code: int = 200):  # noqa: D401
        obj = str.__new__(cls, content)
        obj.status_code = status_code
        return obj


__all__ = ["FastAPI", "Form", "HTMLResponse"]
