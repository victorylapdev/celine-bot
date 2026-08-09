"""Agent Core: controlled orchestration of LLM calls and registered tools."""

import json
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from celine.ai.providers.deepseek import (
    ChatMessage,
    ProviderError,
    ProviderResponse,
    ToolCall,
)
from celine.core.context import AgentErrorRecord, ExecutionContext, ToolExecutionRecord
from celine.tools.errors import ToolError
from celine.tools.registry import ToolRegistry


class LLMProvider(Protocol):
    """Provider contract used by Agent Core."""

    def complete(
        self,
        messages: list[ChatMessage],
        tools: tuple[dict[str, Any], ...],
    ) -> ProviderResponse: ...


class AgentResult(BaseModel):
    """Structured outcome of one Agent execution."""

    response: str | None = None
    context: ExecutionContext
    error: AgentErrorRecord | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.response is not None


class Agent:
    """Orchestrates provider, tool registry, and execution context."""

    def __init__(self, provider: LLMProvider, registry: ToolRegistry, max_tool_iterations: int = 5) -> None:
        if max_tool_iterations < 1:
            raise ValueError("max_tool_iterations deve ser pelo menos 1.")
        self.provider = provider
        self.registry = registry
        self.max_tool_iterations = max_tool_iterations

    def run(self, user_message: str) -> AgentResult:
        """Process a user request through the complete LLM-and-tools loop."""
        context = ExecutionContext.start(user_message)
        tool_definitions = self.registry.as_function_definitions()

        while True:
            try:
                provider_response = self.provider.complete(context.messages, tool_definitions)
            except ProviderError as error:
                return self._failed(context, "provider_error", "Falha ao comunicar com o provedor de IA.", error)
            except Exception as error:  # noqa: BLE001 - Agent boundary returns controlled failures.
                return self._failed(context, "unexpected_error", "Falha inesperada no Agent.", error)

            context.messages.append(provider_response.message)
            if not provider_response.tool_calls:
                if provider_response.message.content:
                    return AgentResult(response=provider_response.message.content, context=context)
                return self._failed(
                    context,
                    "invalid_provider_response",
                    "O provedor não retornou uma resposta final em texto.",
                )

            if context.tool_iterations >= self.max_tool_iterations:
                return self._failed(
                    context,
                    "max_tool_iterations",
                    "O limite de iterações de ferramentas foi atingido.",
                )

            context.tool_iterations += 1
            for tool_call in provider_response.tool_calls:
                context.tool_calls.append(tool_call)
                context.messages.append(self._execute_tool_call(context, tool_call))

    def _execute_tool_call(self, context: ExecutionContext, tool_call: ToolCall) -> ChatMessage:
        """Execute a Tool only via the registry and return a tool protocol message."""
        record = ToolExecutionRecord(
            call_id=tool_call.id,
            name=tool_call.name,
            raw_arguments=tool_call.arguments,
        )
        context.tool_executions.append(record)

        try:
            arguments = json.loads(tool_call.arguments)
            if not isinstance(arguments, Mapping):
                raise TypeError("Os argumentos da ferramenta devem ser um objeto JSON.")
            record.result = self.registry.execute(tool_call.name, arguments)
            payload: dict[str, Any] = {"ok": True, "result": record.result}
        except json.JSONDecodeError as error:
            payload = self._tool_error(context, record, "invalid_json", "JSON de argumentos inválido.", error)
        except ValidationError as error:
            payload = self._tool_error(
                context, record, "invalid_arguments", "Argumentos inválidos para a ferramenta.", error
            )
        except TypeError as error:
            payload = self._tool_error(context, record, "invalid_arguments", str(error), error)
        except ToolError as error:
            payload = self._tool_error(context, record, "tool_error", str(error), error)
        except Exception as error:  # noqa: BLE001 - Tools must not terminate the Agent loop.
            payload = self._tool_error(context, record, "unexpected_tool_error", "Falha inesperada na ferramenta.", error)

        return ChatMessage(
            role="tool",
            tool_call_id=tool_call.id,
            content=json.dumps(payload, ensure_ascii=False, default=str),
        )

    def _tool_error(
        self,
        context: ExecutionContext,
        record: ToolExecutionRecord,
        code: str,
        message: str,
        details: Exception,
    ) -> dict[str, Any]:
        error = context.add_error(code, message, details)
        record.error = error
        return {"ok": False, "error": error.model_dump(exclude_none=True)}

    @staticmethod
    def _failed(
        context: ExecutionContext,
        code: str,
        message: str,
        details: Exception | None = None,
    ) -> AgentResult:
        return AgentResult(context=context, error=context.add_error(code, message, details))
