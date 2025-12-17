"""Minimal pydantic BaseModel stub for tests."""

from __future__ import annotations

from typing import Any, Dict


class BaseModel:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self) -> Dict[str, Any]:
        return self.__dict__

    def dict(self) -> Dict[str, Any]:
        return self.__dict__


__all__ = ["BaseModel"]
