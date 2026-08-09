from types import SimpleNamespace

import pytest

from celine.ai.providers.deepseek import (
    ChatMessage,
    DeepSeekClient,
    ProviderCommunicationError,
    UnexpectedProviderResponseError,
)
from celine.core.settings import Settings


class FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return self.response


def build_client(response: object) -> tuple[DeepSeekClient, FakeCompletions]:
    completions = FakeCompletions(response)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = Settings(deepseek_api_key="test-key")
    return DeepSeekClient(settings=settings, client=fake_client), completions  # type: ignore[arg-type]


def test_deepseek_client_returns_completion_text() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Olá! Como posso ajudar?", tool_calls=[]))]
    )
    client, _ = build_client(response)

    assert client.reply([ChatMessage(role="user", content="Olá, Celine!")]) == "Olá! Como posso ajudar?"


def test_deepseek_client_sends_tools_and_extracts_tool_call() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="system_info", arguments="{}"),
                        )
                    ],
                )
            )
        ]
    )
    client, completions = build_client(response)
    tools = [{"type": "function", "function": {"name": "system_info", "parameters": {}}}]

    result = client.complete([ChatMessage(role="user", content="Qual sistema? ")], tools)

    assert result.tool_calls[0].name == "system_info"
    assert result.tool_calls[0].arguments == "{}"
    assert completions.requests[0]["tools"] == tools


def test_deepseek_client_wraps_communication_error() -> None:
    class FailingCompletions:
        def create(self, **kwargs: object) -> object:
            raise RuntimeError("network unavailable")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    client = DeepSeekClient(settings=Settings(deepseek_api_key="test-key"), client=fake_client)  # type: ignore[arg-type]

    with pytest.raises(ProviderCommunicationError):
        client.complete([ChatMessage(role="user", content="Olá")])


def test_deepseek_client_rejects_unexpected_response() -> None:
    client, _ = build_client(SimpleNamespace(choices=[]))

    with pytest.raises(UnexpectedProviderResponseError):
        client.complete([ChatMessage(role="user", content="Olá")])
