from src.rag.agentic_rag import AgenticRAG


def test_retrieve_returns_ranked_documents(tmp_path):
    doc_path = tmp_path / "guide.txt"
    doc_path.write_text("executive function guidance for clinicians", encoding="utf-8")

    rag = AgenticRAG(index_path=tmp_path / "index_store")
    rag.ingest_documents([str(doc_path)])

    results = rag.retrieve(["executive"], k=1)

    assert results
    assert results[0].source == str(doc_path)
