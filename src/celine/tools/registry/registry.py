"""Registry and dispatcher for Celine tools."""

from collections.abc import Mapping
from typing import Any

from celine.tools.base import Tool, ToolInput
from celine.tools.errors import DuplicateToolError, ToolNotFoundError


class ToolRegistry:
    """Central catalog of tools, independent from their implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool[ToolInput]] = {}

    def register(self, tool: Tool[ToolInput]) -> None:
        """Register a tool under its stable unique name."""
        if tool.name in self._tools:
            raise DuplicateToolError(f"A ferramenta '{tool.name}' já está registrada.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[ToolInput]:
        """Retrieve a tool by name."""
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError(f"A ferramenta '{name}' não está registrada.") from error

    def list(self) -> tuple[Tool[ToolInput], ...]:
        """List all registered tools in registration order."""
        return tuple(self._tools.values())

    def execute(self, name: str, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Validate input and execute a registered tool by name."""
        tool = self.get(name)
        validated_parameters = tool.input_model.model_validate(parameters or {})
        return tool.execute(validated_parameters)
