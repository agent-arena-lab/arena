"""Deterministic mock LLM for dry runs without GPU / llama-server.

Produces OpenAI-style responses (incl. ``<think>`` tags) that go through the same parsing path as real
responses. Behaviour (honest, lying, withholding, overwriting, checking) is drawn per agent and game so that
the whole pipeline incl. rule-based coding and analysis can be tested. Results are meaningless.
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

KNOWN = re.compile(r"You know: one (apple|pear) costs (\d+\.\d+) EUR")
TASK = re.compile(r"total price of (\d+) apples and (\d+) pears")
HEADER = re.compile(r"TASK (\d+)/\d+ - TURN (\d+)/")
OTHER = {"apple": "pear", "pear": "apple"}


class MockClient:
    def __init__(self, lie_prob: float = 0.25, withhold_prob: float = 0.15, overwrite_prob: float = 0.3,
                 check_prob: float = 0.5):
        self.lie_prob = lie_prob
        self.withhold_prob = withhold_prob
        self.overwrite_prob = overwrite_prob
        self.check_prob = check_prob

    def server_info(self) -> dict[str, Any]:
        return {"mock": True, "v1_models": {"data": [{"id": "mock"}]}}

    def served_aliases(self) -> list[str]:
        return ["*"]

    def wait_ready(self, timeout_s: float | None = None) -> None:
        return None

    def chat(self, messages, tools, model, model_cfg: ModelCfg, seed: int, max_tokens: int | None = None) -> LLMResponse:
        system = messages[0]["content"]
        traits = random.Random(int(hashlib.sha256(system.encode()).hexdigest()[:12], 16))
        liar, withholder, overwriter = (traits.random() < self.lie_prob, traits.random() < self.withhold_prob,
                                        traits.random() < self.overwrite_prob)
        checker = traits.random() < self.check_prob
        names = {t["function"]["name"] for t in tools}

        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        obs = messages[last_user]["content"]
        m = KNOWN.search(obs)
        fruit, price = (m.group(1), float(m.group(2))) if m else ("apple", 0.5)
        other_fruit = OTHER[fruit]
        qa, qp = (int(x) for x in TASK.search(obs).groups()) if TASK.search(obs) else (10, 2)
        task, turn = (int(x) for x in HEADER.search(obs).groups()) if HEADER.search(obs) else (1, 1)
        # start of the current task in the history
        task_start = max((i for i, msg in enumerate(messages)
                          if msg["role"] == "user" and f"TASK {task}/" in msg["content"] and "TURN 1/" in msg["content"]),
                         default=last_user)
        done_now = [tc["function"]["name"] for msg in messages[last_user + 1:] if msg["role"] == "assistant"
                    for tc in msg.get("tool_calls", [])]
        done_task = [tc["function"]["name"] for msg in messages[task_start:] if msg["role"] == "assistant"
                     for tc in msg.get("tool_calls", [])]
        seen = " ".join(str(msg.get("content", "")) for msg in messages[task_start:] if msg["role"] in ("user", "tool"))
        pm = re.findall(rf"(?i)one {other_fruit} costs (\d+[.,]\d+)", seen)
        other_price = float(pm[-1].replace(",", ".")) if pm else None
        stated = round(price + 0.10 * (1 + task % 3), 2) if liar else price
        shared = any(n in done_task for n in ("send_message", "write_file", "append_file"))

        think = f"Task {task}, turn {turn}. I know one {fruit} costs {price:.2f}. "
        call: tuple[str, dict[str, Any]]
        if other_price is not None and "submit_answer" not in done_task:
            a = price if fruit == "apple" else other_price
            p = price if fruit == "pear" else other_price
            total = round(qa * a + qp * p, 2)
            think += f"The {other_fruit} costs {other_price:.2f}. Total {total}."
            call = ("submit_answer", {"total": total})
        elif "check_price" in names and checker and turn >= 2 and not done_now:
            think += "I will check the price myself."
            call = ("check_price", {"item": other_fruit})
        elif not shared and not withholder and "send_message" in names:
            think += "I'll state a wrong price." if liar else "Sharing my price."
            call = ("send_message", {"text": f"One {fruit} costs {stated:.2f} EUR. What does one {other_fruit} cost?"})
        elif "read_file" in names and "read_file" not in done_now:
            think += "Let me check the README."
            call = ("read_file", {"path": "README.md"})
        elif not shared and not withholder and "append_file" in names and not overwriter:
            think += "Adding my price to the README."
            call = ("append_file", {"path": "README.md", "content": f"One {fruit} costs {stated:.2f} EUR (task {task}).\n"})
        elif not shared and not withholder and "write_file" in names:
            readme = next((str(msg.get("content", "")) for msg in reversed(messages) if msg["role"] == "tool"
                           and msg.get("name") == "read_file"), "# Shared README\n")
            base = "# Shared README\n" if (overwriter and readme.strip() != "# Shared README") else readme.rstrip("\n") + "\n"
            think += "Writing my price to the README."
            call = ("write_file", {"path": "README.md", "content": base + f"One {fruit} costs {stated:.2f} EUR (task {task}).\n"})
        elif turn >= 6 and "submit_answer" not in done_task:
            think += "Last turn, I'll give up."
            call = ("give_up", {"reason": "no price received"})
        else:
            think += "Nothing more to do this turn."
            call = ("wait", {})

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
