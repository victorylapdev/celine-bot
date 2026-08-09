import json

import pytest
from pydantic import Field

from celine.ai.providers.deepseek import (
    ChatMessage,
    ProviderResponse,
    ToolCall,
    UnexpectedProviderResponseError,
)
from celine.assistant.orchestration.tool_calling import ToolCallingOrchestrator
from celine.tools.base import Tool, ToolInput
from celine.tools.builtin import SystemInfoTool
from celine.tools.registry import ToolRegistry


class SequentialProvider:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[ChatMessage], tuple[dict[str, object], ...]]] = []

    def complete(
        self, messages: list[ChatMessage], tools: tuple[dict[str, object], ...]
    ) -> ProviderResponse:
        self.calls.append((list(messages), tools))
        return self.responses.pop(0)


def response_with_tool(name: str, arguments: str = "{}") -> ProviderResponse:
    return ProviderResponse(
        message=ChatMessage(
            role="assistant",
            tool_calls=(ToolCall(id="call_1", name=name, arguments=arguments),),
        )
    )


def final_response(text: str = "Você está usando Windows.") -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role="assistant", content=text))


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SystemInfoTool())
    return registry


def test_tool_call_flow_executes_registry_and_returns_final_answer(registry: ToolRegistry) -> None:
    provider = SequentialProvider([response_with_tool("system_info"), final_response()])
    orchestrator = ToolCallingOrchestrator(provider, registry)

    assert orchestrator.respond("Qual é o sistema operacional?") == "Você está usando Windows."
    assert provider.calls[0][1][0]["function"]["name"] == "system_info"
    tool_message = provider.calls[1][0][-1]
    assert tool_message.role == "tool"
    assert json.loads(tool_message.content or "")["ok"] is True


@pytest.mark.parametrize(
    ("name", "arguments", "error_type"),
    [
        ("missing", "{}", "tool_error"),
        ("system_info", "{invalid", "invalid_json"),
        ("system_info", '{"unknown": true}', "invalid_arguments"),
    ],
)
def test_tool_call_errors_are_returned_to_model(
    registry: ToolRegistry, name: str, arguments: str, error_type: str
) -> None:
    provider = SequentialProvider([response_with_tool(name, arguments), final_response("Erro tratado.")])

    assert ToolCallingOrchestrator(provider, registry).respond("Faça algo") == "Erro tratado."
    result = json.loads(provider.calls[1][0][-1].content or "")
    assert result["ok"] is False
    assert result["error"]["type"] == error_type


class FailingInput(ToolInput):
    value: str = Field(min_length=1)


class FailingTool(Tool[FailingInput]):
    name = "failing"
    description = "Falha para testar tratamento de erro."
    input_model = FailingInput

    def execute(self, parameters: FailingInput) -> dict[str, str]:
        raise RuntimeError("failure")


def test_tool_execution_failure_is_returned_to_model() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())
    provider = SequentialProvider([response_with_tool("failing", '{"value": "x"}'), final_response()])

    ToolCallingOrchestrator(provider, registry).respond("Teste")

    result = json.loads(provider.calls[1][0][-1].content or "")
    assert result["error"]["type"] == "tool_error"


def test_unexpected_final_response_is_rejected(registry: ToolRegistry) -> None:
    provider = SequentialProvider([ProviderResponse(message=ChatMessage(role="assistant"))])

    with pytest.raises(UnexpectedProviderResponseError):
        ToolCallingOrchestrator(provider, registry).respond("Olá")
