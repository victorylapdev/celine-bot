"""Command-line utilities for Celine."""

import argparse

from celine.ai.providers.deepseek import DeepSeekClient
from celine.core.agent import Agent
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

    result = Agent(DeepSeekClient(), build_default_registry()).run(args.message)
    if not result.succeeded:
        raise RuntimeError(result.error.message if result.error else "Falha desconhecida no Agent.")
    print(result.response)


if __name__ == "__main__":
    main()
