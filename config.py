import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Chatbot configuration (future-ready)
    CHAT_PROVIDER: str  # e.g., "anthropic"
    CHAT_MODEL: str     # e.g., "claude-sonnet-4.5"
    ENABLE_CLAUDE_SONNET_4_5: str

    # API keys (future use)
    ANTHROPIC_API_KEY: str

def get_settings() -> Settings:
    """Build settings from current environment on each call.
    This allows runtime toggling (e.g., in tests or via setx/new shells).
    """
    return Settings(
        CHAT_PROVIDER=os.getenv("CHAT_PROVIDER", "none"),
        CHAT_MODEL=os.getenv("CHAT_MODEL", ""),
        ENABLE_CLAUDE_SONNET_4_5=os.getenv("ENABLE_CLAUDE_SONNET_4_5", "0"),
        ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY", ""),
    )
