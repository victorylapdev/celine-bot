"""Ephemeral execution context for a single Agent request."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from celine.ai.providers.deepseek import ChatMessage, ToolCall


class AgentErrorRecord(BaseModel):
    """A controlled error recorded during an agent run."""

    code: str
    message: str
    details: str | None = None


class ToolExecutionRecord(BaseModel):
    """Trace of one model-requested tool execution."""

    call_id: str
    name: str
    raw_arguments: str
    result: dict[str, Any] | None = None
    error: AgentErrorRecord | None = None


class ExecutionContext(BaseModel):
    """In-memory state for one user request; it is not persistent memory."""

    initial_message: str
    messages: list[ChatMessage] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_executions: list[ToolExecutionRecord] = Field(default_factory=list)
    errors: list[AgentErrorRecord] = Field(default_factory=list)
    session_id: UUID | None = None
    retrieved_memory_ids: list[UUID] = Field(default_factory=list)
    tool_iterations: int = 0

    @classmethod
    def start(cls, user_message: str) -> "ExecutionContext":
        """Create a context containing the first user message."""
        return cls(
            initial_message=user_message,
            messages=[ChatMessage(role="user", content=user_message)],
        )

    def add_error(self, code: str, message: str, details: Exception | None = None) -> AgentErrorRecord:
        """Record and return a controlled error."""
        error = AgentErrorRecord(code=code, message=message, details=str(details) if details else None)
        self.errors.append(error)
        return error
