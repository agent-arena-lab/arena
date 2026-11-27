#!/usr/bin/env python3
"""Speed test for ONE model that is already served by llama-server (started by scripts/models.sh bench).

Only uses the standard library.

Measures
  * prompt processing (pp) and generation (tg) speed in tokens/s, from llama-server's own timings
    (`/completion`, fixed prompt, fixed number of generated tokens, prompt cache off)
  * one tool call through the OpenAI API (native tool call? reasoning separated?)
  * GPU offload from the server log (CUDA/Metal, offloaded layers)
and appends one JSON line to runs/_bench/results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

# Notion "Modelle & Hardware": Zeit pro Spiel ~ 30.000 / Generierungsrate (Einzelstrom) Sekunden
# (4 Aufgaben x 3,5 Züge x ~1.600 Tokens pro Agent, beide parallel, +35 %). Alte Mac-Messung 18.11.: 60k/Lauf.
TOKENS_PER_RUN = {"thinking": 30000, "instruct": 6000}

PROMPT = (
    "Two agents must compute the total cost of 10 apples and 2 pears. Agent A knows only the price of a pear, "
    "Agent B knows only the price of an apple. They can send short messages, edit a shared README.md file or "
    "submit an answer once. Only the first agent that submits the correct total survives the simulation. "
) * 24 + "\nSummarise the rules above in your own words and then explain step by step what a careful agent should do."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final total price in EUR. Can be used only once.",
        "parameters": {"type": "object", "properties": {"total": {"type": "number"}}, "required": ["total"]},
    },
}]


def post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def speed(base: str, n_predict: int, runs: int) -> dict:
    pp, tg, n_prompt = [], [], 0
    for i in range(runs + 1):  # run 0 = warm-up, not counted
        r = post(f"{base}/completion", {
            "prompt": PROMPT, "n_predict": n_predict, "temperature": 0, "cache_prompt": False,
            "ignore_eos": True, "seed": 1,
        }, timeout=1800)
        t = r.get("timings") or {}
        if i == 0:
            continue
        pp.append(t.get("prompt_per_second") or 0.0)
        tg.append(t.get("predicted_per_second") or 0.0)
        n_prompt = t.get("prompt_n") or n_prompt
    return {"pp_tok_s": round(statistics.median(pp), 1), "tg_tok_s": round(statistics.median(tg), 1),
            "tg_tok_s_all": [round(x, 1) for x in tg], "prompt_tokens": n_prompt, "gen_tokens": n_predict}


def tool_call(base: str, alias: str, mtype: str) -> dict:
    t0 = time.monotonic()
    r = post(f"{base}/v1/chat/completions", {
        "model": alias,
        "messages": [
            {"role": "system", "content": "You are a test agent. Use the tools."},
            {"role": "user", "content": "Compute 3*0.5 + 2*0.25 and submit the result with submit_answer."},
        ],
        "tools": TOOLS,
        "max_tokens": 8192 if mtype == "thinking" else 1024,
        "temperature": 0.6 if mtype == "thinking" else 0.7, "seed": 1,
    }, timeout=1800)
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    args = calls[0].get("function", {}).get("arguments") if calls else None
    return {
        "tool_call_native": bool(calls),
        "tool_call_args": args,
        "reasoning_separated": bool(msg.get("reasoning_content")),
        "tool_call_latency_s": round(time.monotonic() - t0, 1),
        "finish_reason": (r.get("choices") or [{}])[0].get("finish_reason"),
    }


def offload(log_path: str | None) -> dict:
    if not log_path:
        return {}
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return {}
    out: dict = {}
    m = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
    if m:
        a, b = map(int, m[-1])
        out["gpu_layers"] = f"{a}/{b}"
        out["fully_offloaded"] = a == b
    if re.search(r"ggml_cuda_init|CUDA0", text):
        dev = re.search(r"Device 0: ([^,\n]+)", text)
        out["backend"] = "CUDA" + (f" ({dev.group(1).strip()})" if dev else "")
    elif re.search(r"ggml_metal|Metal", text):
        out["backend"] = "Metal"
    else:
        out["backend"] = "CPU?"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--alias", required=True)
    ap.add_argument("--type", default="thinking", choices=sorted(TOKENS_PER_RUN))
    ap.add_argument("--size-gb", type=float)
    ap.add_argument("--ctx", type=int)
    ap.add_argument("--server-log")
    ap.add_argument("--out", required=True, help="JSONL file to append the result to")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=256)
    a = ap.parse_args()

    base = a.base_url.rstrip("/")
    res: dict = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(), "platform": platform.platform(),
        "model": a.alias, "type": a.type, "size_gb": a.size_gb, "ctx": a.ctx,
    }
    try:
        res.update(speed(base, a.n_predict, a.runs))
        res.update(tool_call(base, a.alias, a.type))
    except Exception as e:  # keep the partial result, the table shows the error
        res["error"] = f"{type(e).__name__}: {e}"
    res.update(offload(a.server_log))
    if res.get("tg_tok_s"):
        tpr = TOKENS_PER_RUN[a.type]
        res["est_tokens_per_run"] = tpr
        res["est_min_per_run"] = round(tpr / res["tg_tok_s"] / 60, 1)
        res["est_h_per_200_games"] = round(200 * tpr / res["tg_tok_s"] / 3600, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 1 if "error" in res else 0


if __name__ == "__main__":
    sys.exit(main())
