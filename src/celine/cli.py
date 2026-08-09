"""Command-line utilities for Celine."""

import argparse

from celine.ai.providers.deepseek import DeepSeekClient
from celine.core.agent import Agent
from celine.core.settings import get_settings
from celine.memory.factory import build_postgres_memory_manager
from celine.tools.builtin import SystemInfoTool
from celine.tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    """Compose the tools enabled for the local Celine runtime."""
    registry = ToolRegistry()
    registry.register(SystemInfoTool())
    return registry


def main() -> None:
    """Send one prompt to DeepSeek and print its reply."""
    parser = argparse.ArgumentParser(description="Envie uma mensagem para a DeepSeek.")
    parser.add_argument("message", help="Mensagem que será enviada à DeepSeek.")
    args = parser.parse_args()

    settings = get_settings()
    memory_manager = (
        build_postgres_memory_manager(settings.database_url) if settings.database_url else None
    )
    result = Agent(DeepSeekClient(), build_default_registry(), memory_manager=memory_manager).run(
        args.message
    )
    if not result.succeeded:
        raise RuntimeError(result.error.message if result.error else "Falha desconhecida no Agent.")
    print(result.response)


if __name__ == "__main__":
    main()
