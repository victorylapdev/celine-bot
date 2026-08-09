"""Agent Core: controlled orchestration of LLM calls and registered tools."""

import json
from collections.abc import Mapping
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ValidationError

from celine.ai.providers.deepseek import (
    ChatMessage,
    ProviderError,
    ProviderResponse,
    ToolCall,
)
from celine.core.context import AgentErrorRecord, ExecutionContext, ToolExecutionRecord
from celine.memory.manager import MemoryManager
from celine.memory.models import HistoryMessage
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

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_tool_iterations: int = 5,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        if max_tool_iterations < 1:
            raise ValueError("max_tool_iterations deve ser pelo menos 1.")
        self.provider = provider
        self.registry = registry
        self.max_tool_iterations = max_tool_iterations
        self.memory_manager = memory_manager

    def run(self, user_message: str, session_id: UUID | None = None) -> AgentResult:
        """Process a user request through the complete LLM-and-tools loop."""
        context = ExecutionContext.start(user_message)
        self._add_relevant_memories(context, user_message, session_id)
        tool_definitions = self.registry.as_function_definitions()

        while True:
            try:
                provider_response = self.provider.complete(context.messages, tool_definitions)
            except ProviderError as error:
                return self._finish(self._failed(context, "provider_error", "Falha ao comunicar com o provedor de IA.", error))
            except Exception as error:  # noqa: BLE001 - Agent boundary returns controlled failures.
                return self._finish(self._failed(context, "unexpected_error", "Falha inesperada no Agent.", error))

            context.messages.append(provider_response.message)
            if not provider_response.tool_calls:
                if provider_response.message.content:
                    return self._finish(AgentResult(response=provider_response.message.content, context=context))
                return self._finish(self._failed(
                    context,
                    "invalid_provider_response",
                    "O provedor não retornou uma resposta final em texto.",
                ))

            if context.tool_iterations >= self.max_tool_iterations:
                return self._finish(self._failed(
                    context,
                    "max_tool_iterations",
                    "O limite de iterações de ferramentas foi atingido.",
                ))

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

    def _add_relevant_memories(
        self, context: ExecutionContext, user_message: str, session_id: UUID | None
    ) -> None:
        if self.memory_manager is None:
            return
        try:
            session = self.memory_manager.create_session() if session_id is None else None
            context.session_id = session.id if session else session_id
            memories = self.memory_manager.recall(user_message, limit=5, min_score=0.15)
            context.retrieved_memory_ids = [item.memory.id for item in memories]
            memory_context = self.memory_manager.format_context(memories)
            if memory_context:
                context.messages.insert(0, ChatMessage(role="system", content=memory_context))
        except Exception as error:  # noqa: BLE001 - memory must not prevent a user request.
            context.add_error("memory_retrieval_error", "Falha ao recuperar memórias relevantes.", error)

    def _finish(self, result: AgentResult) -> AgentResult:
        if self.memory_manager is None or result.context.session_id is None:
            return result
        try:
            history = [
                HistoryMessage(
                    session_id=result.context.session_id,
                    role=message.role,
                    content=message.content,
                    metadata={
                        **({"tool_call_id": message.tool_call_id} if message.tool_call_id else {}),
                        **(
                            {"tool_calls": [call.model_dump() for call in message.tool_calls]}
                            if message.tool_calls
                            else {}
                        ),
                    },
                )
                for message in result.context.messages
            ]
            self.memory_manager.record_history(result.context.session_id, history)
        except Exception as error:  # noqa: BLE001 - persistence failure is recorded, not fatal.
            result.context.add_error("history_persistence_error", "Falha ao registrar o histórico.", error)
        return result

    @staticmethod
    def _failed(
        context: ExecutionContext,
        code: str,
        message: str,
        details: Exception | None = None,
    ) -> AgentResult:
        return AgentResult(context=context, error=context.add_error(code, message, details))
