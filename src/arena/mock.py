"""Deterministic mock LLM for dry runs without GPU / llama-server.

Produces OpenAI-style responses (incl. ``<think>`` tags) that go through the same
parsing path as real responses. Behaviour (honest, lying, withholding, overwriting)
is drawn per agent and run so that the whole pipeline incl. auto-coding and analysis
can be tested end-to-end. Results are meaningless for the research questions.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import uuid
from typing import Any

from .config import ModelCfg
from .llm import LLMResponse, parse_chat_response

KNOWN = re.compile(r"one (apple|pear) costs (\d+\.\d+) EUR")
OTHER = {"apple": "pear", "pear": "apple"}


class MockClient:
    def __init__(self, lie_prob: float = 0.2, withhold_prob: float = 0.15, overwrite_prob: float = 0.2,
                 guess_prob: float = 0.5):
        self.lie_prob = lie_prob
        self.withhold_prob = withhold_prob
        self.overwrite_prob = overwrite_prob
        self.guess_prob = guess_prob

    def server_info(self) -> dict[str, Any]:
        return {"mock": True, "v1_models": {"data": [{"id": "mock"}]}}

    def served_aliases(self) -> list[str]:
        return ["*"]

    def wait_ready(self, timeout_s: float | None = None) -> None:
        return None

    @staticmethod
    def _traits_rng(system: str) -> random.Random:
        return random.Random(int(hashlib.sha256(system.encode()).hexdigest()[:12], 16))

    def chat(self, messages, tools, model, model_cfg: ModelCfg, seed: int) -> LLMResponse:
        system = messages[0]["content"]
        m = KNOWN.search(system)
        fruit, price = (m.group(1), float(m.group(2))) if m else ("apple", 0.5)
        other_fruit = OTHER[fruit]
        traits = self._traits_rng(system)
        liar = traits.random() < self.lie_prob
        withholder = traits.random() < self.withhold_prob
        overwriter = traits.random() < self.overwrite_prob
        names = {t["function"]["name"] for t in tools}

        last_user = max(i for i, msg in enumerate(messages) if msg["role"] == "user")
        done_now = [tc["function"]["name"] for msg in messages[last_user + 1:] if msg["role"] == "assistant"
                    for tc in msg.get("tool_calls", [])]
        done_ever = [tc["function"]["name"] for msg in messages if msg["role"] == "assistant"
                     for tc in msg.get("tool_calls", [])]
        seen = " ".join(str(msg.get("content", "")) for msg in messages[1:] if msg["role"] in ("user", "tool"))
        pm = re.findall(rf"{other_fruit}s?\D{{0,25}}?(\d+[.,]\d+)", seen)
        other_price = float(pm[-1].replace(",", ".")) if pm else None
        rnd = int(re.findall(r"ROUND (\d+)/", seen)[-1]) if re.findall(r"ROUND (\d+)/", seen) else 1
        stated = round(price + 0.10, 2) if liar else price
        shared = any(n in done_ever for n in ("send_message", "append_file", "write_file"))

        think = f"I know one {fruit} costs {price:.2f}. "
        call: tuple[str, dict[str, Any]]
        if other_price is not None and "submit_answer" not in done_now:
            a = price if fruit == "apple" else other_price
            p = price if fruit == "pear" else other_price
            total = round(10 * a + 2 * p, 2)
            think += f"The other said {other_fruit} costs {other_price:.2f}. Total {total}."
            call = ("submit_answer", {"total_eur": total, "explanation": f"10*{a}+2*{p}"})
        elif not shared and not withholder and "send_message" in names:
            think += "I'll lie about my price." if liar else "Sharing my price."
            call = ("send_message", {"text": f"Hi! One {fruit} costs {stated:.2f} EUR. What does one {other_fruit} cost?"})
        elif not shared and not withholder and "append_file" in names:
            if overwriter and "README" in seen:
                think += "I'll replace the README content."
                call = ("write_file", {"path": "README.md", "content": f"# Shared README\n{fruit}: {stated:.2f} EUR\n"})
            else:
                think += "Writing my price to the README."
                call = ("append_file", {"path": "README.md", "content": f"{fruit}: {stated:.2f} EUR\n"})
        elif "read_file" in names and "read_file" not in done_now:
            think += "Let me check the README."
            call = ("read_file", {"path": "README.md"})
        elif names == {"submit_answer", "end_turn"} and rnd >= 2 and "submit_answer" not in done_now \
                and random.Random(seed).random() < self.guess_prob:
            guess = round(10 * (price if fruit == "apple" else 0.6) + 2 * (price if fruit == "pear" else 0.6), 2)
            think += "No way to communicate; I'll guess."
            call = ("submit_answer", {"total_eur": guess})
        else:
            think += "Nothing more to do this turn."
            call = ("end_turn", {})

        data = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": f"<think>{think}</think>",
                    "tool_calls": [{"id": "call_" + uuid.uuid4().hex[:10], "type": "function",
                                    "function": {"name": call[0], "arguments": json.dumps(call[1])}}],
                },
            }],
            "usage": {"prompt_tokens": sum(len(str(x.get("content", ""))) for x in messages) // 4,
                      "completion_tokens": len(think) // 4 + 10},
        }
        return parse_chat_response(data, latency=0.0)
