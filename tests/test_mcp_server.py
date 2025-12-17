from fastapi.testclient import TestClient

from src.server.mcp_server import app


client = TestClient(app)


def test_task_creation_and_answer():
    create_resp = client.post("/api/mcp/task/create", json={"task_type": "visual_reasoning"})
    assert create_resp.status_code == 200
    payload = create_resp.json()
    assert payload["success"]
    task_id = payload["data"]["task_id"]

    answer_resp = client.post(f"/api/mcp/task/{task_id}/answer", json={"answer": "triangle"})
    assert answer_resp.status_code == 200
    assert answer_resp.json()["data"]["state"] == "SCORED"
