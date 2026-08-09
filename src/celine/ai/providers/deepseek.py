"""DeepSeek provider using its OpenAI-compatible chat-completions API."""

from collections.abc import Sequence
from typing import Any, Literal

from openai import APIError, OpenAI
from pydantic import BaseModel, Field, ValidationError

from celine.core.settings import Settings, get_settings


class ProviderError(Exception):
    """Base exception for errors returned by an LLM provider."""


class ProviderCommunicationError(ProviderError):
    """Raised when the DeepSeek API cannot be reached or rejects a request."""


class UnexpectedProviderResponseError(ProviderError):
    """Raised when a provider response does not match the expected protocol."""


class ToolCall(BaseModel):
    """A function call requested by the language model."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str = Field(min_length=1)


class ChatMessage(BaseModel):
    """Provider-neutral conversation message, serializable for DeepSeek."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def as_api_message(self) -> dict[str, Any]:
        """Convert the message to the OpenAI-compatible request format."""
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        return {key: value for key, value in message.items() if value is not None}


class ProviderResponse(BaseModel):
    """Normalized result of one DeepSeek completion request."""

    message: ChatMessage

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.message.tool_calls


class DeepSeekClient:
    """Adapter responsible only for communication with the DeepSeek API."""

    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.deepseek_api_key:
            raise ValueError(
                "CELINE_DEEPSEEK_API_KEY não foi configurada. Adicione sua chave ao arquivo .env."
            )
        self.client = client or OpenAI(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
        )

    def reply(self, messages: Sequence[ChatMessage]) -> str:
        """Send a regular chat request and return its text response."""
        response = self.complete(messages)
        if response.tool_calls or not response.message.content:
            raise UnexpectedProviderResponseError("A DeepSeek não retornou uma resposta textual final.")
        return response.message.content

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict[str, Any]] = (),
    ) -> ProviderResponse:
        """Send messages and optional tool definitions to DeepSeek."""
        request: dict[str, Any] = {
            "model": self.settings.deepseek_model,
            "messages": [message.as_api_message() for message in messages],
        }
        if tools:
            request["tools"] = list(tools)

        try:
            completion = self.client.chat.completions.create(**request)
        except APIError as error:
            raise ProviderCommunicationError("Falha de comunicação com a DeepSeek.") from error
        except Exception as error:
            raise ProviderCommunicationError("Falha inesperada ao chamar a DeepSeek.") from error

        try:
            choice = completion.choices[0]
            raw_message = choice.message
            raw_tool_calls = raw_message.tool_calls or []
            tool_calls = tuple(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in raw_tool_calls
            )
            return ProviderResponse(
                message=ChatMessage(
                    role="assistant",
                    content=raw_message.content,
                    tool_calls=tool_calls,
                )
            )
        except (AttributeError, IndexError, TypeError, ValidationError) as error:
            raise UnexpectedProviderResponseError("Resposta inesperada recebida da DeepSeek.") from error
