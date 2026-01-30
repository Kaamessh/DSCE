from __future__ import annotations

from typing import Optional
import logging

from config import get_settings

logger = logging.getLogger("food-ai-system.chatbot")


def is_claude_sonnet_enabled() -> bool:
    settings = get_settings()
    return (
        settings.CHAT_PROVIDER.lower() == "anthropic"
        and settings.CHAT_MODEL.lower() in {"claude-sonnet-4.5", "claude_sonnet_4_5"}
        and settings.ENABLE_CLAUDE_SONNET_4_5 == "1"
    )


def get_chat_response(prompt: str) -> str:
    """Placeholder chatbot hook.

    This function is intentionally a stub. It returns a clear message indicating
    what would happen if integrated. Real API calls are not performed here.
    """
    if not prompt or not prompt.strip():
        return "Please enter a prompt."

    if is_claude_sonnet_enabled():
        # In a future integration, this is where the SDK call would occur.
        # Example (pseudo-code):
        # from anthropic import Anthropic
        # client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        # resp = client.messages.create(model=settings.CHAT_MODEL, messages=[{"role":"user","content":prompt}])
        # return resp.content[0].text
        return (
            "Claude Sonnet 4.5 is marked enabled. "
            "Add the Anthropic SDK and API key to perform real calls."
        )

    return (
        "Chatbot not enabled. Set CHAT_PROVIDER=anthropic, "
        "CHAT_MODEL=claude-sonnet-4.5, and ENABLE_CLAUDE_SONNET_4_5=1 to enable."
    )
