import os
from services.chatbot import is_claude_sonnet_enabled

print("CHAT_PROVIDER=", os.getenv("CHAT_PROVIDER"))
print("CHAT_MODEL=", os.getenv("CHAT_MODEL"))
print("ENABLE_CLAUDE_SONNET_4_5=", os.getenv("ENABLE_CLAUDE_SONNET_4_5"))
print("enabled=", is_claude_sonnet_enabled())
