"""Command-line utilities for Celine."""

import argparse

from celine.ai.providers.deepseek import ChatMessage, DeepSeekClient


def main() -> None:
    """Send one prompt to DeepSeek and print its reply."""
    parser = argparse.ArgumentParser(description="Envie uma mensagem para a DeepSeek.")
    parser.add_argument("message", help="Mensagem que será enviada à DeepSeek.")
    args = parser.parse_args()

    answer = DeepSeekClient().reply([ChatMessage(role="user", content=args.message)])
    print(answer)


if __name__ == "__main__":
    main()
