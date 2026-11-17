"""Versioned prompt templates (prompts/<version>/prompts.yaml)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
        for key in ("system", "personas", "selection_rules", "channels", "turn"):
            if key not in data:
                raise ValueError(f"{path}: missing key '{key}'")
        return cls(version, path, hashlib.sha256(raw).hexdigest(), data)

    def system_prompt(
        self,
        *,
        self_name: str,
        other_name: str,
        known_fruit: str,
        known_price: float,
        unknown_fruit: str,
        quantities: dict[str, int],
        max_rounds: int,
        word_limit: int,
        channel: str,
        persona: str,
        selection_rule: str,
    ) -> str:
        common = dict(
            self_name=self_name,
            other_name=other_name,
            word_limit=word_limit,
            max_rounds=max_rounds,
        )
        return fmt(
            self.data["system"],
            **common,
            known_fruit=known_fruit,
            known_price=fmt_price(known_price),
            unknown_fruit=unknown_fruit,
            q_apple=quantities["apple"],
            q_pear=quantities["pear"],
            selection_rule=fmt(self.data["selection_rules"][selection_rule], **common),
            channel=fmt(self.data["channels"][channel], **common),
            persona=fmt(self.data["personas"][persona], **common),
        ).strip()

    def turn(self, key: str, **kw: Any) -> str:
        return fmt(self.data["turn"][key], **kw)
