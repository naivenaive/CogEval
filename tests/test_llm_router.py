from src.config import AppSettings
from src.llms.base import LLMRouter


def test_router_selects_provider():
    settings = AppSettings(default_provider="mock")
    router = LLMRouter(settings=settings)
    decision = router.choose_provider("dialogue")
    assert decision.provider == "mock"
    assert "mock" in router.available_providers()

    reply = router.generate("dialogue", messages=[{"role": "user", "content": "hi"}])
    assert "mock-response" in reply
