"""Atom memory captures per-turn events with metadata persisted in SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class MemoryAtom:
    id: Optional[int]
    user_id: str
    session_id: str
    timestamp: datetime
    atom_type: str
    cognitive_domain: str
    raw_content: str
    score: Optional[float]
    confidence: float
    source: str


class AtomMemory:
    def __init__(self, db_path: str = "./cogeval.db") -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS atoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                timestamp TEXT,
                atom_type TEXT,
                cognitive_domain TEXT,
                raw_content TEXT,
                score REAL,
                confidence REAL,
                source TEXT
            )
            """
        )
        self.conn.commit()

    def append(
        self,
        user_id: str,
        session_id: str,
        atom_type: str,
        cognitive_domain: str,
        raw_content: str,
        score: Optional[float] = None,
        confidence: float = 0.5,
        source: str = "chat",
    ) -> int:
        ts = datetime.utcnow().isoformat()
        cur = self.conn.execute(
            """
            INSERT INTO atoms (user_id, session_id, timestamp, atom_type, cognitive_domain, raw_content, score, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, session_id, ts, atom_type, cognitive_domain, raw_content, score, confidence, source),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recall(self, user_id: str, limit: int = 5) -> List[MemoryAtom]:
        rows = self.conn.execute(
            """SELECT * FROM atoms WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def query_by_domain(self, user_id: str, domain: str, limit: int = 20) -> List[MemoryAtom]:
        rows = self.conn.execute(
            """SELECT * FROM atoms WHERE user_id = ? AND cognitive_domain = ? ORDER BY timestamp DESC LIMIT ?""",
            (user_id, domain, limit),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def query_by_session(self, user_id: str, session_id: str) -> List[MemoryAtom]:
        rows = self.conn.execute(
            """SELECT * FROM atoms WHERE user_id = ? AND session_id = ? ORDER BY timestamp ASC""",
            (user_id, session_id),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def sessions_for_user(self, user_id: str) -> List[str]:
        rows = self.conn.execute(
            """SELECT DISTINCT session_id FROM atoms WHERE user_id = ? ORDER BY timestamp""",
            (user_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def domains_for_user(self, user_id: str) -> List[str]:
        rows = self.conn.execute(
            """SELECT DISTINCT cognitive_domain FROM atoms WHERE user_id = ?""",
            (user_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def session_time(self, user_id: str, session_id: str) -> datetime:
        row = self.conn.execute(
            """SELECT timestamp FROM atoms WHERE user_id = ? AND session_id = ? ORDER BY timestamp LIMIT 1""",
            (user_id, session_id),
        ).fetchone()
        if not row:
            return datetime.utcnow()
        return datetime.fromisoformat(row[0])

    def store_dialogue(self, user_id: str, session_id: str, role: str, content: str) -> int:
        domain = "language" if role == "user" else "executive"
        return self.append(
            user_id=user_id,
            session_id=session_id,
            atom_type="dialogue",
            cognitive_domain=domain,
            raw_content=content,
            score=None,
            confidence=0.5,
            source="chat",
        )

    def store_task_result(
        self,
        user_id: str,
        session_id: str,
        domain: str,
        content: str,
        score: float,
        confidence: float = 0.7,
        source: str = "tool",
    ) -> int:
        return self.append(
            user_id=user_id,
            session_id=session_id,
            atom_type="task_result",
            cognitive_domain=domain,
            raw_content=content,
            score=score,
            confidence=confidence,
            source=source,
        )

    def _row_to_atom(self, row: sqlite3.Row) -> MemoryAtom:
        return MemoryAtom(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            atom_type=row["atom_type"],
            cognitive_domain=row["cognitive_domain"],
            raw_content=row["raw_content"],
            score=row["score"],
            confidence=row["confidence"],
            source=row["source"],
        )

    def close(self) -> None:
        self.conn.close()


__all__ = ["AtomMemory", "MemoryAtom"]
