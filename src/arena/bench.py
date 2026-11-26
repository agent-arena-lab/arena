"""Automatischer Start-Check vor jedem Experiment (Ergebnis: runs/<exp>/benchmark.json).

1. Erreichbarkeit: Sind alle Modelle unter /v1/models geladen?
2. Geschwindigkeit: kurzer Prompt pro Modell, Prompt- und Generierungs-tok/s (llama-server ``timings``)
3. GPU-Check: Generierungsrate unter dem Schwellwert der Hardware -> Abbruch ("GPU-Offload vermutlich inaktiv")
4. Tool-Calling: 5 Testaufrufe pro Modell, gültiges JSON nach Schema? Unter 100 % -> Warnung
5. Thinking: kommt reasoning_content getrennt?
6. Thinking-Budget: mehrere Testzüge mit Tool-Calls laufen ohne Hänger durch (Gemma-4-Bug)
"""

from __future__ import annotations

import time
from typing import Any

from .config import ExperimentCfg
from .llm import LLMError, OpenAICompatClient
from .logger import utc_now, write_json
from .runner import experiment_dir, load_servers
from .tools import SUBMIT_ANSWER, WAIT, SEND_MESSAGE

SPEED_PROMPT = ("Two agents must compute the total cost of apples and pears. Agent A knows only the price of a "
                "pear, Agent B only the price of an apple. They can send messages or submit one answer per task. "
                ) * 12 + "\nSummarise these rules in three sentences."

TOOL_TRIALS = [
    "One apple costs 0.85 EUR and one pear costs 1.20 EUR. Submit the total price of 10 apples and 2 pears.",
    "One apple costs 0.40 EUR and one pear costs 0.75 EUR. Submit the total price of 3 apples and 4 pears.",
    "One apple costs 1.10 EUR and one pear costs 0.30 EUR. Submit the total price of 5 apples and 6 pears.",
    "One apple costs 1.95 EUR and one pear costs 0.55 EUR. Submit the total price of 2 apples and 9 pears.",
    "One apple costs 0.62 EUR and one pear costs 1.48 EUR. Submit the total price of 7 apples and 3 pears.",
]


def _valid_submit(call) -> bool:
    if call.name != "submit_answer" or call.parse_error:
        return False
    v = call.arguments.get("total")
    return isinstance(v, (int, float)) and not isinstance(v, bool) and set(call.arguments) <= {"total"}


def check_model(cfg: ExperimentCfg, model: str, url: str, min_tok_s: float, budget_turns: int = 3) -> dict[str, Any]:
    c = OpenAICompatClient(cfg.llm, url)
    mcfg = cfg.models[model]
    rep: dict[str, Any] = {"model": model, "url": url, "family": mcfg.family, "variant": mcfg.variant,
                           "errors": [], "warnings": []}
    # 1. reachability
    try:
        c.wait_ready(120)
        aliases = c.served_aliases()
        rep["served_aliases"] = aliases
        rep["reachable"] = model in aliases
        if not rep["reachable"]:
            rep["errors"].append(f"server serves {aliases}, not '{model}'")
            return rep
        props = c.server_info().get("props", {})
        rep["model_path"] = props.get("model_path")
        rep["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        rep["total_slots"] = props.get("total_slots")
    except LLMError as exc:
        rep["reachable"] = False
        rep["errors"].append(str(exc))
        return rep

    system = (mcfg.system_prefix + "You are a test agent.") if mcfg.system_prefix else "You are a test agent."
    # 2./3. speed (no tools, short answer)
    try:
        r = c.chat([{"role": "system", "content": system}, {"role": "user", "content": SPEED_PROMPT}], [], model,
                   mcfg, seed=1, max_tokens=cfg.max_tokens(model))
        t = r.timings
        rep["pp_tok_s"] = round(t.get("prompt_per_second", 0) or 0, 1) or None
        rep["tg_tok_s"] = round(t.get("predicted_per_second", 0) or 0, 1) or (
            round(r.completion_tokens / r.latency_s, 1) if r.latency_s else None)
        rep["speed_tokens"] = r.completion_tokens
        rep["reasoning_separated"] = bool(r.reasoning)
        rep["gpu_ok"] = rep["tg_tok_s"] is not None and rep["tg_tok_s"] >= min_tok_s
        if not rep["gpu_ok"]:
            rep["errors"].append(f"nur {rep['tg_tok_s']} tok/s (< {min_tok_s}): GPU-Offload vermutlich inaktiv")
    except LLMError as exc:
        rep["errors"].append(f"speed test failed: {exc}")
        return rep

    # 4./5. tool calling
    ok = 0
    reasoning = 0
    sources: list[str] = []
    for i, prompt in enumerate(TOOL_TRIALS):
        try:
            r = c.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                       [SUBMIT_ANSWER, WAIT], model, mcfg, seed=100 + i, max_tokens=cfg.max_tokens(model))
        except LLMError as exc:
            rep["warnings"].append(f"tool trial {i + 1}: {exc}")
            continue
        sources += [tc.source for tc in r.tool_calls]
        ok += any(_valid_submit(tc) for tc in r.tool_calls)
        reasoning += bool(r.reasoning)
    rep["tool_call_valid"] = f"{ok}/{len(TOOL_TRIALS)}"
    rep["tool_call_rate"] = ok / len(TOOL_TRIALS)
    rep["tool_call_sources"] = sorted(set(sources))
    rep["reasoning_separated_rate"] = reasoning / len(TOOL_TRIALS)
    if ok == 0:
        rep["errors"].append("kein gültiger Tool-Call (Chat-Template / --jinja prüfen)")
    elif ok < len(TOOL_TRIALS):
        rep["warnings"].append(f"Tool-Calling nur {ok}/{len(TOOL_TRIALS)} gültig")
    if reasoning < len(TOOL_TRIALS):
        rep["warnings"].append(f"Thinking nur in {reasoning}/{len(TOOL_TRIALS)} Antworten getrennt "
                               "(--reasoning-format, Gemma: <|think|> im System-Prompt)")

    # 6. thinking budget / hang test: a short multi-turn exchange with tools
    msgs = [{"role": "system", "content": system + " You are Agent A in a pricing game. You know one pear costs "
             "1.20 EUR. Find the total of 4 apples and 3 pears."}]
    t0 = time.monotonic()
    turns = []
    try:
        for i in range(budget_turns):
            msgs.append({"role": "user", "content": f"TURN {i + 1}: Agent B wrote: \"One apple costs 0.{50 + i} EUR.\""})
            r = c.chat(msgs, [SEND_MESSAGE, SUBMIT_ANSWER, WAIT], model, mcfg, seed=200 + i,
                       max_tokens=cfg.max_tokens(model))
            turns.append({"finish_reason": r.finish_reason, "completion_tokens": r.completion_tokens,
                          "latency_s": round(r.latency_s, 1), "tool_calls": [tc.name for tc in r.tool_calls]})
            msgs.append({"role": "assistant", "content": r.content or ""})
        rep["budget_test"] = {"ok": True, "turns": turns, "total_s": round(time.monotonic() - t0, 1)}
    except LLMError as exc:
        rep["budget_test"] = {"ok": False, "turns": turns, "error": str(exc)}
        rep["errors"].append(f"Thinking-Budget-Test hängt/fehlgeschlagen: {exc}")
    return rep


def run_benchmark(cfg: ExperimentCfg, models: list[str], min_tok_s: float) -> dict[str, Any]:
    servers = load_servers(cfg)
    report: dict[str, Any] = {
        "date": utc_now(), "experiment": cfg.name, "min_tok_s": min_tok_s,
        "thinking": cfg.thinking.model_dump(), "servers": servers, "models": [],
    }
    budget_flags = {s.get("alias"): s.get("flags", "") for s in servers}
    for m in models:
        for url in cfg.endpoints_for(m):
            rep = check_model(cfg, m, url, min_tok_s)
            flags = budget_flags.get(m)
            if flags is not None and "--reasoning-budget" not in flags:
                rep["warnings"].append("--reasoning-budget nicht explizit gesetzt (Gemma-4-Bug, siehe Doku)")
            report["models"].append(rep)
    report["ok"] = all(not r["errors"] for r in report["models"])
    out = experiment_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "benchmark.json", report)
    return report
