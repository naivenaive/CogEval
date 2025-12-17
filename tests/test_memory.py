from src.memory.atom_memory import AtomMemory


def test_memory_append_and_recall(tmp_path):
    db_path = tmp_path / "mem.db"
    memory = AtomMemory(db_path=str(db_path))
    memory.append(
        user_id="user1",
        session_id="sess1",
        atom_type="dialogue",
        cognitive_domain="language",
        raw_content="hello",
        confidence=0.8,
        score=None,
    )
    memory.append(
        user_id="user1",
        session_id="sess1",
        atom_type="task_result",
        cognitive_domain="memory",
        raw_content="score",
        confidence=0.9,
        score=85.0,
        source="tool",
    )

    atoms = memory.recall(user_id="user1", limit=2)
    assert len(atoms) == 2
    assert {a.atom_type for a in atoms} == {"dialogue", "task_result"}
