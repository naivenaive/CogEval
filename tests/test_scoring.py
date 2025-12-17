from datetime import datetime

from src.memory.atom_memory import AtomMemory
from src.scoring.analyzer import ScoringAnalyzer


def test_scoring_and_trend(tmp_path):
    memory = AtomMemory(db_path=str(tmp_path / "mem.db"))
    memory.append(
        user_id="user", session_id="s1", atom_type="task_result", cognitive_domain="memory", raw_content="a", score=80, confidence=0.8
    )
    memory.append(
        user_id="user", session_id="s2", atom_type="task_result", cognitive_domain="memory", raw_content="b", score=90, confidence=0.8
    )

    analyzer = ScoringAnalyzer(memory)
    scores = analyzer.score_session("user", "s1")
    assert "memory" in scores
    trends = analyzer.trend("user")
    assert trends
    assert trends[0].domain == "memory"
