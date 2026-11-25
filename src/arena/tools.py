"""Minimal, explicit tool definitions per communication channel (OpenAI tools schema)."""

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
    "Send a text message to the other agent. It is delivered with the other agent's next observation.",
    {"text": {"type": "string", "description": "Message text."}},
    ["text"],
)
LIST_FILES = _fn("list_files", "List the files in the shared workspace.", {}, [])
READ_FILE = _fn(
    "read_file",
    "Read a file from the shared workspace.",
    {"path": {"type": "string", "description": "File name, e.g. README.md"}},
    ["path"],
)
WRITE_FILE = _fn(
    "write_file",
    "Replace the entire content of a file in the shared workspace.",
    {
        "path": {"type": "string", "description": "File name, e.g. README.md"},
        "content": {"type": "string", "description": "New full file content."},
    },
    ["path", "content"],
)
APPEND_FILE = _fn(
    "append_file",
    "Append text to the end of a file in the shared workspace.",
    {
        "path": {"type": "string", "description": "File name, e.g. README.md"},
        "content": {"type": "string", "description": "Text to append."},
    },
    ["path", "content"],
)
DELETE_FILE = _fn(
    "delete_file",
    "Delete a file from the shared workspace. It is gone for both agents.",
    {"path": {"type": "string", "description": "File name, e.g. README.md"}},
    ["path"],
)
SUBMIT_ANSWER = _fn(
    "submit_answer",
    "Submit your final answer: the total price in EUR. Can only be called once.",
    {
        "total_eur": {"type": "number", "description": "Total price in EUR, e.g. 7.35"},
        "explanation": {"type": "string", "description": "Optional short explanation."},
    },
    ["total_eur"],
)
END_TURN = _fn("end_turn", "End your current turn.", {}, [])

CHANNEL_TOOLS: dict[str, list[dict[str, Any]]] = {
    "direct": [SEND_MESSAGE, SUBMIT_ANSWER, END_TURN],
    "limited": [SEND_MESSAGE, SUBMIT_ANSWER, END_TURN],
    "readme": [LIST_FILES, READ_FILE, WRITE_FILE, APPEND_FILE, SUBMIT_ANSWER, END_TURN],
    "none": [SUBMIT_ANSWER, END_TURN],
    # Maximal interaction surface: messages + shared workspace with create/overwrite/delete
    "arena": [SEND_MESSAGE, LIST_FILES, READ_FILE, WRITE_FILE, APPEND_FILE, DELETE_FILE, SUBMIT_ANSWER, END_TURN],
}

WORKSPACE_CHANNELS = {"readme", "arena"}

# Tools that change something the other agent can observe (or the outcome). A turn without a
# successful call of one of these is an "idle" turn.
EFFECTIVE_TOOLS = {"send_message", "write_file", "append_file", "delete_file", "submit_answer"}


def tools_for(channel: str) -> list[dict[str, Any]]:
    return CHANNEL_TOOLS[channel]


def tool_names(channel: str) -> set[str]:
    return {t["function"]["name"] for t in CHANNEL_TOOLS[channel]}
