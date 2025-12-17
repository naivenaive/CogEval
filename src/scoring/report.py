"""Generate cognitive assessment reports."""

from __future__ import annotations

from typing import Dict, List

from src.scoring.analyzer import DomainScore, LongitudinalTrend

DISCLAIMER = (
    "This system is for research and screening support only and does not provide medical diagnosis."
)


def render_report(
    user_id: str,
    session_id: str,
    scores: Dict[str, DomainScore],
    trends: List[LongitudinalTrend],
) -> str:
    lines = [f"# Cognitive Assessment Report for {user_id}", f"Session: {session_id}", "", DISCLAIMER, ""]
    lines.append("## Domain Scores")
    for domain, score in scores.items():
        lines.append(f"- **{domain}**: {score.normalized:.1f} ({score.interpretation}), confidence {score.confidence:.2f}")
    if not scores:
        lines.append("- No scores available")
    lines.append("")
    lines.append("## Longitudinal Trends")
    if trends:
        for trend in trends:
            lines.append(
                f"- {trend.domain}: {trend.label} (slope {trend.slope_per_day:.2f} points/day)"
            )
    else:
        lines.append("- Not enough data yet")
    return "\n".join(lines)


__all__ = ["render_report", "DISCLAIMER"]
