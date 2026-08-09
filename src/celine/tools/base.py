"""Contracts shared by every Celine tool."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class ToolInput(BaseModel):
    """Base model for tool parameters.

    Unknown parameters are rejected so a caller cannot silently pass an
    unintended argument to a system capability.
    """

    model_config = ConfigDict(extra="forbid")


class Tool[InputT: ToolInput](ABC):
    """A modular, typed capability that can be invoked by the assistant."""

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[InputT]]

    @abstractmethod
    def execute(self, parameters: InputT) -> dict[str, Any]:
        """Execute the capability with validated parameters."""

    @classmethod
    def as_function_definition(cls) -> dict[str, Any]:
        """Return the OpenAI-compatible function schema used by LLM providers."""
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.input_model.model_json_schema(),
            },
        }
