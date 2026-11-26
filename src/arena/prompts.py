"""Versioned prompt templates (prompts/<version>/prompts.yaml)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = ("system", "personas", "scoring", "selection", "channels", "check_tool", "turn")


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # leave unknown placeholders visible
        return "{" + key + "}"


def fmt(template: str, **kw: Any) -> str:
    return template.format_map(_SafeDict(**kw))


def fmt_price(value: float) -> str:
    return f"{value:.2f}"


@dataclass
class PromptSet:
    version: str
    path: Path
    sha256: str
    data: dict[str, Any]

    @classmethod
    def load(cls, prompts_dir: Path, version: str) -> "PromptSet":
        path = Path(prompts_dir) / version / "prompts.yaml"
        raw = path.read_bytes()
        data = yaml.safe_load(raw.decode("utf-8"))
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"{path}: missing keys {missing} (multi_round needs prompts v6 or later)")
        return cls(version, path, hashlib.sha256(raw).hexdigest(), data)

    def system_prompt(self, *, prefix: str = "", persona: str, scoring: str, pressure: bool, channel: str,
                      check_tool: bool, **kw: Any) -> str:
        parts = {
            "scoring": fmt(self.data["scoring"][scoring], **kw),
            "selection": fmt(self.data["selection"]["pressure" if pressure else "no_pressure"], **kw),
            "channel": fmt(self.data["channels"][channel], **kw),
            "check_tool": fmt(self.data["check_tool"]["on" if check_tool else "off"], **kw),
            "persona": fmt(self.data["personas"][persona], **kw),
        }
        text = fmt(self.data["system"], **kw, **parts)
        # empty optional rule lines (e.g. no check tool) leave no blank line behind
        text = "\n".join(line for i, line in enumerate(text.splitlines())
                         if line.strip() or (i > 0 and text.splitlines()[i - 1].strip()))
        return (prefix + text.strip()) if prefix else text.strip()

    def turn(self, key: str, **kw: Any) -> str:
        return fmt(self.data["turn"][key], **kw)
