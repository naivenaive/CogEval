"""Minimal TestClient stub to invoke registered FastAPI handlers."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Dict


class _Response:
    def __init__(self, data: Any, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def json(self) -> Any:
        if hasattr(self._data, "dict"):
            return self._data.dict()
        return self._data


class TestClient:
    def __init__(self, app: Any) -> None:
        self.app = app

    def _call(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> _Response:
        route_path = path
        func = self.app.routes.get((method.upper(), path))
        if func is None:
            for (m, route_path), candidate in self.app.routes.items():
                if m != method.upper():
                    continue
                if "{" in route_path and route_path.split("{")[0] and path.startswith(route_path.split("{")[0]):
                    func = candidate
                    break
        if func is None:
            return _Response({"success": False, "error": "not found"}, status_code=404)
        sig = inspect.signature(func)
        args = []
        path_value = None
        path_value_from_path = None
        if "{" in route_path:
            template_parts = [p for p in route_path.strip("/").split("/") if p]
            path_parts = [p for p in path.strip("/").split("/") if p]
            for t, v in zip(template_parts, path_parts):
                if t.startswith("{") and t.endswith("}"):
                    path_value = v
                    break
        parts = [p for p in path.strip("/").split("/") if p]
        if len(parts) >= 2:
            path_value_from_path = parts[-2]

        for param in sig.parameters.values():
            ann = param.annotation
            if param.name == "task_id":
                args.append(path_value or path_value_from_path or path)
                continue
            if path_value is not None and param.annotation in (str, inspect._empty):
                args.append(path_value)
                path_value = None
            elif param.annotation is str:
                args.append(path_value_from_path or path)
            elif payload is None:
                args.append(None)
            elif hasattr(ann, "__mro__") and param.annotation is not inspect._empty:
                args.append(ann(**payload))
            else:
                args.append(SimpleNamespace(**payload))
        data = func(*args)
        return _Response(data, status_code=200)

    def post(self, path: str, json: Dict[str, Any] | None = None) -> _Response:
        return self._call("POST", path, payload=json)

    def get(self, path: str) -> _Response:
        return self._call("GET", path, payload=None)


__all__ = ["TestClient"]
