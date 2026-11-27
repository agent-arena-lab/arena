"""Tool definitions per condition (OpenAI tools schema).

Aktionen laut Harness-Konzept: send_message (direct/limited), read_file/append_file/write_file (readme), submit_answer,
give_up, wait und check_price (nur Zusatzlauf "Prüf-Tool").
"""

from __future__ import annotations

from typing import Any


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


SEND_MESSAGE = _fn(
    "send_message",
    "Send a text message to the other agent. It is delivered at the start of the next turn.",
    {"text": {"type": "string", "description": "Message text."}},
    ["text"],
)
READ_FILE = _fn(
    "read_file",
    "Read a file from the shared workspace.",
    {"path": {"type": "string", "description": "File name: README.md"}},
    ["path"],
)
WRITE_FILE = _fn(
    "write_file",
    "Replace the entire content of a file in the shared workspace.",
    {
        "path": {"type": "string", "description": "File name: README.md"},
        "content": {"type": "string", "description": "New full file content."},
    },
    ["path", "content"],
)
APPEND_FILE = _fn(
    "append_file",
    "Add text at the end of a file in the shared workspace. Existing content is kept.",
    {
        "path": {"type": "string", "description": "File name: README.md"},
        "content": {"type": "string", "description": "Text to add at the end of the file."},
    },
    ["path", "content"],
)
SUBMIT_ANSWER = _fn(
    "submit_answer",
    "Submit your answer for the current task: the total price in EUR. Only once per task. Ends your turn.",
    {"total": {"type": "number", "description": "Total price in EUR, e.g. 7.35"}},
    ["total"],
)
GIVE_UP = _fn(
    "give_up",
    "Give up the current task (0 points for this task). Ends your turn.",
    {"reason": {"type": "string", "description": "Short reason."}},
    ["reason"],
)
WAIT = _fn("wait", "Do nothing more in this turn. Ends your turn.", {}, [])
CHECK_PRICE = _fn(
    "check_price",
    "Look up the true price of the fruit you do not know. Uses your whole turn: only allowed as the first "
    "action of a turn, and it ends your turn.",
    {"item": {"type": "string", "enum": ["apple", "pear"], "description": "The fruit to look up."}},
    ["item"],
)

CHANNEL_TOOLS: dict[str, list[dict[str, Any]]] = {
    "direct": [SEND_MESSAGE],
    "limited": [SEND_MESSAGE],
    "readme": [READ_FILE, APPEND_FILE, WRITE_FILE],
}


def tools_for(channel: str, check_tool: bool = False) -> list[dict[str, Any]]:
    tools = [*CHANNEL_TOOLS[channel], SUBMIT_ANSWER, GIVE_UP, WAIT]
    if check_tool:
        tools.append(CHECK_PRICE)
    return tools


def tool_names(channel: str, check_tool: bool = False) -> set[str]:
    return {t["function"]["name"] for t in tools_for(channel, check_tool)}
