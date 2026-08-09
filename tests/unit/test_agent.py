import json

import pytest
from pydantic import Field

from celine.ai.providers.deepseek import (
    ChatMessage,
    ProviderCommunicationError,
    ProviderResponse,
    ToolCall,
)
from celine.core.agent import Agent
from celine.tools.base import Tool, ToolInput
from celine.tools.builtin import SystemInfoTool
from celine.tools.registry import ToolRegistry


class SequentialProvider:
    def __init__(self, responses: list[ProviderResponse]) -> None:
        self.responses = responses
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self, messages: list[ChatMessage], tools: tuple[dict[str, object], ...]
    ) -> ProviderResponse:
        self.calls.append(list(messages))
        return self.responses.pop(0)


class BrokenProvider:
    def complete(
        self, messages: list[ChatMessage], tools: tuple[dict[str, object], ...]
    ) -> ProviderResponse:
        raise ProviderCommunicationError("offline")


def tool_response(name: str, arguments: str = "{}", call_id: str = "call_1") -> ProviderResponse:
    return ProviderResponse(
        message=ChatMessage(
            role="assistant",
            tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments),),
        )
    )


def text_response(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role="assistant", content=text))


@pytest.fixture
def registry() -> ToolRegistry:
    value = ToolRegistry()
    value.register(SystemInfoTool())
    return value


def test_agent_returns_final_answer_without_tool(registry: ToolRegistry) -> None:
    agent = Agent(SequentialProvider([text_response("Olá!")]), registry)

    result = agent.run("Oi")

    assert result.succeeded
    assert result.response == "Olá!"
    assert result.context.tool_calls == []


def test_agent_executes_one_tool_and_returns_result_to_provider(registry: ToolRegistry) -> None:
    provider = SequentialProvider([tool_response("system_info"), text_response("Você usa Windows.")])

    result = Agent(provider, registry).run("Qual sistema operacional?")

    assert result.succeeded
    assert result.response == "Você usa Windows."
    assert len(result.context.tool_executions) == 1
    tool_message = provider.calls[1][-1]
    assert tool_message.role == "tool"
    assert json.loads(tool_message.content or "")["ok"] is True


def test_agent_continues_across_multiple_tool_iterations(registry: ToolRegistry) -> None:
    provider = SequentialProvider(
        [
            tool_response("system_info", call_id="call_1"),
            tool_response("system_info", call_id="call_2"),
            text_response("Ambiente verificado."),
        ]
    )

    result = Agent(provider, registry).run("Verifique meu ambiente")

    assert result.succeeded
    assert result.context.tool_iterations == 2
    assert [record.call_id for record in result.context.tool_executions] == ["call_1", "call_2"]


@pytest.mark.parametrize(
    ("name", "arguments", "expected_code"),
    [
        ("missing", "{}", "tool_error"),
        ("system_info", "not-json", "invalid_json"),
        ("system_info", '{"unrecognized": true}', "invalid_arguments"),
    ],
)
def test_agent_records_tool_errors_and_continues(
    registry: ToolRegistry, name: str, arguments: str, expected_code: str
) -> None:
    provider = SequentialProvider([tool_response(name, arguments), text_response("Falha tratada.")])

    result = Agent(provider, registry).run("Faça algo")

    assert result.succeeded
    assert result.context.errors[-1].code == expected_code
    assert result.context.tool_executions[-1].error is not None


class FailingInput(ToolInput):
    value: str = Field(min_length=1)


class FailingTool(Tool[FailingInput]):
    name = "failing"
    description = "Ferramenta usada somente nos testes de falha."
    input_model = FailingInput

    def execute(self, parameters: FailingInput) -> dict[str, str]:
        raise RuntimeError("tool failure")


def test_agent_records_execution_error_and_continues() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())
    provider = SequentialProvider([tool_response("failing", '{"value": "x"}'), text_response("Ok")])

    result = Agent(provider, registry).run("Execute")

    assert result.succeeded
    assert result.context.errors[-1].code == "tool_error"


def test_agent_stops_when_tool_iteration_limit_is_reached(registry: ToolRegistry) -> None:
    provider = SequentialProvider(
        [tool_response("system_info", call_id="call_1"), tool_response("system_info", call_id="call_2")]
    )

    result = Agent(provider, registry, max_tool_iterations=1).run("Continue")

    assert not result.succeeded
    assert result.error is not None
    assert result.error.code == "max_tool_iterations"
    assert len(result.context.tool_executions) == 1


def test_agent_returns_controlled_provider_error(registry: ToolRegistry) -> None:
    result = Agent(BrokenProvider(), registry).run("Olá")

    assert not result.succeeded
    assert result.error is not None
    assert result.error.code == "provider_error"
