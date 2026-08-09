"""Coordinates LLM tool calls without granting the LLM direct system access."""

import json
from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from celine.ai.providers.deepseek import (
    ChatMessage,
    ProviderResponse,
    ToolCall,
    UnexpectedProviderResponseError,
)
from celine.tools.errors import ToolError
from celine.tools.registry import ToolRegistry


class ToolCallingProvider(Protocol):
    """Minimum provider interface required by the tool-calling flow."""

    def complete(
        self,
        messages: list[ChatMessage],
        tools: tuple[dict[str, Any], ...],
    ) -> ProviderResponse: ...


class ToolCallingLoopError(RuntimeError):
    """Raised when the provider exceeds the permitted tool-call rounds."""


class ToolCallingOrchestrator:
    """Coordinates provider requests and registry-controlled tool execution."""

    def __init__(
        self,
        provider: ToolCallingProvider,
        registry: ToolRegistry,
        max_tool_rounds: int = 5,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.max_tool_rounds = max_tool_rounds

    def respond(self, user_message: str) -> str:
        """Return a final answer, executing only requested registered tools."""
        messages = [ChatMessage(role="user", content=user_message)]
        tool_definitions = self.registry.as_function_definitions()

        for _ in range(self.max_tool_rounds + 1):
            response = self.provider.complete(messages, tool_definitions)
            messages.append(response.message)
            if not response.tool_calls:
                if not response.message.content:
                    raise UnexpectedProviderResponseError(
                        "A DeepSeek não retornou uma resposta final em texto."
                    )
                return response.message.content

            for tool_call in response.tool_calls:
                messages.append(self._execute_tool_call(tool_call))

        raise ToolCallingLoopError("A DeepSeek excedeu o limite de chamadas de ferramentas.")

    def _execute_tool_call(self, tool_call: ToolCall) -> ChatMessage:
        """Validate, run, and serialize one request through the registry only."""
        try:
            arguments = json.loads(tool_call.arguments)
            if not isinstance(arguments, Mapping):
                raise TypeError("Os argumentos da ferramenta devem ser um objeto JSON.")
            result = {"ok": True, "result": self.registry.execute(tool_call.name, arguments)}
        except json.JSONDecodeError:
            result = self._error_result("invalid_json", "Argumentos da ferramenta não contêm JSON válido.")
        except ValidationError as error:
            result = self._error_result("invalid_arguments", "Argumentos inválidos para a ferramenta.", error)
        except ToolError as error:
            result = self._error_result("tool_error", str(error))
        except TypeError as error:
            result = self._error_result("invalid_arguments", str(error))

        return ChatMessage(
            role="tool",
            tool_call_id=tool_call.id,
            content=json.dumps(result, ensure_ascii=False, default=str),
        )

    @staticmethod
    def _error_result(error_type: str, message: str, details: Exception | None = None) -> dict[str, Any]:
        error: dict[str, str] = {"type": error_type, "message": message}
        if details:
            error["details"] = str(details)
        return {"ok": False, "error": error}
