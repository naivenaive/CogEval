from src.agents.cognitive_agent import build_demo_agent


def test_run_turn_returns_response():
    agent = build_demo_agent()
    reply = agent.run_turn(user_id="u1", session_id="s1", message="你好，做一个简单的图形推理题")
    assert "mock-response" in reply

