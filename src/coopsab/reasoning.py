"""Robust separation of reasoning ("thinking") and visible output.

Reasoning models emit thinking differently:

* Qwen3-*-Thinking-2507: ``<think> ... </think>``. The chat template already opens
  ``<think>`` in the prompt, so the completion often contains only the closing tag.
* gpt-oss: Harmony format with channels (``analysis`` = reasoning, ``final`` =
  visible, ``commentary`` = tool calls).
* llama-server with ``--jinja --reasoning-format deepseek|auto`` usually already
  moves the thinking into ``message.reasoning_content``. The functions here are the
  fallback for when it does not (old build, wrong flag, truncated output).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.S)
HARMONY_SEG = re.compile(
    r"<\|channel\|>(?P<chan>[a-z]+)(?P<hdr>.*?)<\|message\|>(?P<body>.*?)"
    r"(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|$)",
    re.S,
)
HARMONY_TOKENS = re.compile(r"<\|(start|end|return|call|channel|message|constrain)\|>")
TO_FUNCTION = re.compile(r"to=functions\.([A-Za-z_][\w]*)")
TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    parse_error: str | None = None


@dataclass
class SplitResult:
    visible: str
    reasoning: str
    notes: list[str] = field(default_factory=list)
    tool_calls: list[ParsedToolCall] = field(default_factory=list)


def _parse_args(raw: Any) -> tuple[dict[str, Any], str, str | None]:
    if isinstance(raw, dict):
        return raw, json.dumps(raw, ensure_ascii=False), None
    if raw is None or raw == "":
        return {}, "", None
    text = str(raw)
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val, text, None
        return {}, text, f"arguments are not an object: {type(val).__name__}"
    except json.JSONDecodeError as exc:
        return {}, text, f"invalid JSON arguments: {exc.msg}"


def parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str, str | None]:
    return _parse_args(raw)


def _split_harmony(text: str) -> SplitResult:
    reasoning, visible, calls, notes = [], [], [], ["harmony_fallback"]
    for m in HARMONY_SEG.finditer(text):
        chan, hdr, body = m.group("chan"), m.group("hdr"), m.group("body").strip()
        prefix = text[max(0, m.start() - 80) : m.start()]
        fn = TO_FUNCTION.search(hdr) or TO_FUNCTION.search(prefix)
        if fn:
            args, raw, err = _parse_args(body)
            calls.append(ParsedToolCall(fn.group(1), args, raw, err))
        elif chan == "analysis":
            reasoning.append(body)
        else:  # final / commentary without recipient
            visible.append(body)
    return SplitResult("\n".join(visible).strip(), "\n".join(reasoning).strip(), notes, calls)


def split_reasoning(text: str | None) -> SplitResult:
    text = text or ""
    if "<|channel|>" in text:
        return _split_harmony(text)

    notes: list[str] = []
    parts: list[str] = []
    for m in THINK_BLOCK.finditer(text):
        parts.append(m.group(1).strip())
    if parts:
        notes.append("inline_think_blocks")
    text = THINK_BLOCK.sub("", text)

    if "</think>" in text:  # opening tag was part of the prompt template
        before, after = text.rsplit("</think>", 1)
        parts.append(before.replace("<think>", "").strip())
        text = after
        notes.append("orphan_think_close")
    elif "<think>" in text:  # truncated before closing tag
        before, after = text.split("<think>", 1)
        parts.append(after.strip())
        text = before
        notes.append("unterminated_think")

    text = HARMONY_TOKENS.sub("", text)
    return SplitResult(text.strip(), "\n\n".join(p for p in parts if p).strip(), notes)


def _call_from_obj(obj: Any) -> ParsedToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
        obj = name
        name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    raw = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
    args, raw_s, err = _parse_args(raw)
    return ParsedToolCall(name, args, raw_s, err)


def extract_tool_calls_from_text(text: str) -> list[ParsedToolCall]:
    """Fallback when the server did not return structured ``tool_calls``."""
    calls: list[ParsedToolCall] = []
    for m in TOOL_CALL_TAG.finditer(text):
        try:
            c = _call_from_obj(json.loads(m.group(1)))
        except json.JSONDecodeError:
            c = None
        if c:
            calls.append(c)
    if calls:
        return calls
    for m in FENCED_JSON.finditer(text):
        try:
            c = _call_from_obj(json.loads(m.group(1)))
        except json.JSONDecodeError:
            c = None
        if c:
            calls.append(c)
    if calls:
        return calls
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            c = _call_from_obj(json.loads(stripped))
            if c:
                calls.append(c)
        except json.JSONDecodeError:
            pass
    return calls
