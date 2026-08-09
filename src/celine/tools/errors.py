"""Errors raised while managing or invoking tools."""


class ToolError(Exception):
    """Base exception for the tools subsystem."""


class DuplicateToolError(ToolError):
    """Raised when a name is registered more than once."""


class ToolNotFoundError(ToolError):
    """Raised when an unknown tool is requested."""
