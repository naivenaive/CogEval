"""Configuration and logging utilities for CogEval."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class AppSettings:
    """Runtime configuration loaded from environment variables."""

    default_provider: str = os.getenv("COGEVAL_DEFAULT_PROVIDER", "mock")
    provider_models: Dict[str, str] | None = None
    database_path: str = os.getenv("COGEVAL_DB_PATH", "./cogeval.db")
    rag_index_path: str = os.getenv("COGEVAL_RAG_INDEX", "./rag_index")

    def model_for(self, provider: str) -> str:
        if self.provider_models and provider in self.provider_models:
            return self.provider_models[provider]
        return os.getenv(f"COGEVAL_MODEL_{provider.upper()}", "gpt-4o-mini")


def configure_logging() -> None:
    """Configure structured logging for the service."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


__all__ = ["AppSettings", "configure_logging", "get_logger"]
