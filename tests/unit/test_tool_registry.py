import platform

import pytest

from celine.tools.builtin import SystemInfoTool
from celine.tools.errors import DuplicateToolError, ToolNotFoundError
from celine.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


def test_register_and_find_tool(registry: ToolRegistry) -> None:
    tool = SystemInfoTool()

    registry.register(tool)

    assert registry.get("system_info") is tool


def test_registering_duplicate_name_raises_error(registry: ToolRegistry) -> None:
    registry.register(SystemInfoTool())

    with pytest.raises(DuplicateToolError, match="system_info"):
        registry.register(SystemInfoTool())


def test_list_tools_returns_registered_tools(registry: ToolRegistry) -> None:
    registry.register(SystemInfoTool())

    assert [tool.name for tool in registry.list()] == ["system_info"]


def test_execute_system_info(registry: ToolRegistry) -> None:
    registry.register(SystemInfoTool())

    result = registry.execute("system_info")

    assert result["operating_system"] == platform.system()
    assert result["python_version"] == platform.python_version()


def test_unknown_tool_raises_error(registry: ToolRegistry) -> None:
    with pytest.raises(ToolNotFoundError, match="missing"):
        registry.get("missing")

    with pytest.raises(ToolNotFoundError, match="missing"):
        registry.execute("missing")
