"""Cognitive scoring and longitudinal analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Dict, List, Tuple

from src.memory.atom_memory import AtomMemory, MemoryAtom

DOMAIN_BANDS = [
    (85, 100, "Within expected range"),
    (70, 84, "Mild concern"),
    (50, 69, "Moderate concern"),
    (0, 49, "High concern"),
]


@dataclass
class DomainScore:
    domain: str
    raw_scores: List[float]
    normalized: float
    confidence: float
    evidence_atoms: List[int]

    @property
    def interpretation(self) -> str:
        for low, high, label in DOMAIN_BANDS:
            if low <= self.normalized <= high:
                return label
        return "Uninterpreted"


@dataclass
class LongitudinalTrend:
    domain: str
    slope_per_day: float
    label: str


class ScoringAnalyzer:
    """Aggregate MemoryAtoms into domain scores and trends."""

    def __init__(self, memory: AtomMemory) -> None:
        self.memory = memory

    def score_session(self, user_id: str, session_id: str) -> Dict[str, DomainScore]:
        atoms = self.memory.query_by_session(user_id, session_id)
        domains: Dict[str, List[MemoryAtom]] = {}
        for atom in atoms:
            domains.setdefault(atom.cognitive_domain, []).append(atom)
        scores: Dict[str, DomainScore] = {}
        for domain, items in domains.items():
            raw_scores = [a.score for a in items if a.score is not None]
            normalized = mean(raw_scores) if raw_scores else 75.0
            evidence = [a.id for a in items if a.id is not None]
            scores[domain] = DomainScore(
                domain=domain,
                raw_scores=raw_scores,
                normalized=normalized,
                confidence=0.6 + 0.1 * min(len(raw_scores), 3),
                evidence_atoms=evidence,
            )
        return scores

    def trend(self, user_id: str) -> List[LongitudinalTrend]:
        sessions = self.memory.sessions_for_user(user_id)
        trends: List[LongitudinalTrend] = []
        for domain in self.memory.domains_for_user(user_id):
            points: List[Tuple[datetime, float]] = []
            for session in sessions:
                scores = self.score_session(user_id, session)
                if domain in scores:
                    # assume session_id encodes order; using stored timestamps would be better
                    ts = self.memory.session_time(user_id, session)
                    points.append((ts, scores[domain].normalized))
            if len(points) < 2:
                continue
            points.sort(key=lambda p: p[0])
            slope = self._slope(points)
            label = self._label_trend(slope)
            trends.append(LongitudinalTrend(domain=domain, slope_per_day=slope, label=label))
        return trends

    def _slope(self, points: List[Tuple[datetime, float]]) -> float:
        x = [(p[0] - points[0][0]).days or 1 for p in points]
        y = [p[1] for p in points]
        n = len(points)
        sum_xy = sum(a * b for a, b in zip(x, y))
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(a * a for a in x)
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom

    def _label_trend(self, slope: float) -> str:
        if slope > 0.5:
            return "Improving"
        if slope < -0.5:
            return "Declining"
        return "Stable"


__all__ = ["ScoringAnalyzer", "DomainScore", "LongitudinalTrend", "DOMAIN_BANDS"]
