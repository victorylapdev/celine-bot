from types import SimpleNamespace

from celine.ai.providers.deepseek import ChatMessage, DeepSeekClient
from celine.core.settings import Settings


class FakeCompletions:
    def create(self, **kwargs: object) -> object:
        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["messages"] == [{"role": "user", "content": "Olá, Celine!"}]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Olá! Como posso ajudar?"))]
        )


def test_deepseek_client_returns_completion_text() -> None:
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    settings = Settings(deepseek_api_key="test-key")
    client = DeepSeekClient(settings=settings, client=fake_client)  # type: ignore[arg-type]

    assert client.reply([ChatMessage(role="user", content="Olá, Celine!")]) == "Olá! Como posso ajudar?"
