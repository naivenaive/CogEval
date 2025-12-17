"""Agentic RAG utilities for cognitive assessment backed by FAISS or a lightweight fallback."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from src.config import get_logger

_FAISS_SPEC = importlib.util.find_spec("faiss")
_FAISS_AVAILABLE = _FAISS_SPEC is not None
if _FAISS_AVAILABLE:  # pragma: no cover - exercised when faiss is available
    import faiss  # type: ignore
    import numpy as np
else:
    faiss = None  # type: ignore
    np = None  # type: ignore


LOGGER = get_logger(__name__)


@dataclass
class Retrieval:
    source: str
    text: str


class _PythonIndex:
    """Pure-Python similarity search used when FAISS is unavailable."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors: List[List[float]] = []

    def add(self, vectors: Sequence[Sequence[float]]) -> None:
        for vec in vectors:
            self.vectors.append(list(vec))

    def search(self, queries: Sequence[Sequence[float]], k: int) -> List[tuple[List[float], List[int]]]:
        results: List[tuple[List[float], List[int]]] = []
        for query in queries:
            scored: List[tuple[int, float]] = []
            for idx, vector in enumerate(self.vectors):
                scored.append((idx, self._cosine_similarity(query, vector)))
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:k]
            scores = [score for _, score in top]
            ids = [i for i, _ in top]
            results.append((scores, ids))
        return results

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)


class AgenticRAG:
    """FAISS-backed retriever with deterministic text embeddings."""

    def __init__(self, index_path: str, dim: int = 64) -> None:
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.docs: List[Retrieval] = []
        self.index = self._create_index(dim)

    def _create_index(self, dim: int):
        if _FAISS_AVAILABLE:  # pragma: no cover - depends on optional dependency
            return faiss.IndexFlatL2(dim)
        return _PythonIndex(dim)

    def _embed_text(self, text: str) -> List[float]:
        """Generate a deterministic embedding using token hashes."""

        tokens = text.lower().split()
        vec = [0.0 for _ in range(self.dim)]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(self.dim):
                vec[i] += digest[i % len(digest)] / 255.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks or [text]

    def _load_content(self, path: Path) -> str:
        if path.suffix.lower() in {".txt", ".md"}:
            return path.read_text(encoding="utf-8")
        LOGGER.warning("Unsupported file type %s, skipping", path)
        return ""

    def ingest_documents(self, paths: Iterable[str], chunk_size: int = 300) -> None:
        embeddings: List[Sequence[float]] = []
        for path_str in paths:
            path = Path(path_str)
            content = self._load_content(path)
            for chunk in self._chunk_text(content, chunk_size=chunk_size):
                self.docs.append(Retrieval(source=path_str, text=chunk))
                embeddings.append(self._embed_text(chunk))
        if not embeddings:
            return
        if _FAISS_AVAILABLE:  # pragma: no cover - depends on optional dependency
            vectors = np.array(embeddings, dtype="float32")
            self.index.add(vectors)
        else:
            self.index.add(embeddings)

    def _search(self, query_embedding: Sequence[float], k: int) -> List[int]:
        if _FAISS_AVAILABLE:  # pragma: no cover - depends on optional dependency
            vectors = np.array([query_embedding], dtype="float32")
            distances, indices = self.index.search(vectors, k)
            return list(indices[0])
        _, hits = self.index.search([query_embedding], k)[0]
        return hits

    def retrieve(self, hints: List[str], k: int = 2) -> List[Retrieval]:
        if not self.docs:
            return [
                Retrieval(source="seed", text="Use standard cognitive evaluation heuristics."),
            ]
        query_embedding = self._embed_text(" ".join(hints))
        hits = self._search(query_embedding, k)
        return [self.docs[i] for i in hits if i < len(self.docs)]

    def agentic_retrieve(self, hints: List[str], k: int = 2) -> List[Retrieval]:
        """Run a two-iteration plan + retrieve loop."""

        first = self.retrieve(hints, k)
        if len(first) >= k:
            return first
        # identify missing aspects: add keywords
        expanded_hints = hints + ["scoring rubric", "domain mapping"]
        second = self.retrieve(expanded_hints, k)
        merged = {r.text: r for r in [*first, *second]}
        return list(merged.values())[:k]

    def reset(self) -> None:
        self.docs = []
        self.index = self._create_index(self.dim)


__all__ = ["AgenticRAG", "Retrieval"]
