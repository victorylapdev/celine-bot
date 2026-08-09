"""DeepSeek chat-completions provider."""

from collections.abc import Sequence

from openai import OpenAI
from pydantic import BaseModel, Field

from celine.core.settings import Settings, get_settings


class ChatMessage(BaseModel):
    """A message accepted by the DeepSeek chat-completions API."""

    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1)


class DeepSeekClient:
    """Small, testable wrapper around DeepSeek's OpenAI-compatible API."""

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
        """Send messages to DeepSeek and return the assistant's text response."""
        completion = self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=[message.model_dump() for message in messages],
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("A DeepSeek retornou uma resposta vazia.")
        return content
