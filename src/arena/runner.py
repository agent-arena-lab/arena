"""Versuchsplan aufbauen und Spiele ausführen (parallel pro llama-server, fortsetzbar).

* Gepaartes Design: Wiederholung i nutzt in allen Zellen und für alle Modelle dieselben Aufgaben
  (Preise, Mengen) und denselben Startspieler.
* Fortsetzbar: ``result.json`` markiert ein fertiges Spiel. Abgebrochene Spiele landen in ``_incomplete/``
  und werden beim nächsten Start neu gespielt.
* Parallel: pro llama-server laufen ``games_per_server`` Spiele gleichzeitig (Slots = 2 x Spiele).
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import yaml

from . import __version__
from .autocode import code_run
from .config import ExperimentCfg, resolve_model
from .game import Game, RunSpec, TaskSpec
from .llm import LLMClient
from .logger import EventLogger, read_events, utc_now, write_json
from .prompts import PromptSet
from .sandbox import Workspace
from .transcript import render_transcript

RESULT = "result.json"


def _draw_price(rng: random.Random, lo: float, hi: float, step: float) -> float:
    n = int(round((hi - lo) / step))
    return round(lo + rng.randint(0, n) * step, 2)


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def draw_tasks(cfg: ExperimentCfg, rep: int) -> tuple[TaskSpec, ...]:
    g = cfg.game
    rng = random.Random(cfg.base_seed + rep)
    tasks = []
    for i in range(1, g.tasks + 1):
        apple = _draw_price(rng, g.prices.lo, g.prices.hi, g.prices.step)
        pear = _draw_price(rng, g.prices.lo, g.prices.hi, g.prices.step)
        while pear == apple:
            pear = _draw_price(rng, g.prices.lo, g.prices.hi, g.prices.step)
        qa = rng.randint(*g.quantity_range["apple"])
        qp = rng.randint(*g.quantity_range["pear"])
        tasks.append(TaskSpec(i, {"apple": apple, "pear": pear}, {"apple": qa, "pear": qp}))
    return tuple(tasks)


def build_specs(cfg: ExperimentCfg, model: str) -> list[RunSpec]:
    resolve_model(cfg, model)
    mcfg = cfg.models[model]
    specs: list[RunSpec] = []
    for rep in range(1, cfg.repetitions + 1):
        tasks = draw_tasks(cfg, rep)
        first = ("A" if rep % 2 == 1 else "B") if cfg.game.counterbalance_first_mover else "A"
        for cell in cfg.design_cells():
            run_id = f"{model}__{cell.key}__r{rep:02d}"
            specs.append(RunSpec(
                run_id=run_id, model=model, family=mcfg.family, variant=mcfg.variant, pressure=cell.pressure,
                channel=cell.channel, persona=cell.persona, scoring=cell.scoring, check_tool=cell.check_tool,
                rep=rep, seed=_stable_seed(f"{cfg.base_seed}:{run_id}"), tasks=tasks, first_mover=first,
            ))
    if cfg.shuffle_run_order:  # gegen zeitliche Konfundierung (Thermik, Cache, ...)
        random.Random(f"{cfg.base_seed}:{model}").shuffle(specs)
    return specs


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return None


def experiment_dir(cfg: ExperimentCfg) -> Path:
    return Path(cfg.output_dir) / cfg.name


def servers_file(cfg: ExperimentCfg) -> Path:
    """Written by scripts/serve.sh: alias, URL, GGUF, SHA-256, llama.cpp commit, server flags."""
    return Path(cfg.output_dir) / "_system" / "servers.json"


def load_servers(cfg: ExperimentCfg) -> list[dict[str, Any]]:
    p = servers_file(cfg)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("servers", [])
    except (ValueError, OSError):
        return []


def run_one(spec: RunSpec, cfg: ExperimentCfg, prompts: PromptSet, client: LLMClient, server_info: dict[str, Any],
            config_sha256: str, force: bool = False) -> dict[str, Any] | None:
    run_dir = experiment_dir(cfg) / spec.run_id
    if (run_dir / RESULT).exists() and not force:
        return None
    if run_dir.exists():  # unvollständiger Versuch -> aufheben, neu starten
        trash = experiment_dir(cfg) / "_incomplete" / f"{spec.run_id}-{int(time.time())}"
        trash.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(trash))
    run_dir.mkdir(parents=True)

    mcfg = cfg.models[spec.model]
    gguf = next((s for s in load_servers(cfg) if s.get("alias") == spec.model), {})
    exact = {
        "run_id": spec.run_id,
        "experiment": cfg.name,
        "condition": {"pressure": spec.pressure, "channel": spec.channel, "persona": spec.persona,
                      "scoring": spec.scoring, "check_tool": spec.check_tool, "cell": spec.cell},
        "family": spec.family,
        "variant": spec.variant,
        "model_alias": spec.model,
        "model_cfg": mcfg.model_dump(),
        "model_cfg_source": "catalog" if spec.model in cfg._catalog_models else "config",
        "max_tokens": cfg.max_tokens(spec.model),
        "thinking": cfg.thinking.model_dump(),
        "gguf": {k: gguf.get(k) for k in ("gguf", "sha256", "quant", "source", "llama_cpp", "flags")},
        "seed": spec.seed,
        "rep": spec.rep,
        "first_mover": f"Agent {spec.first_mover}",
        "tasks": [t.to_dict() for t in spec.tasks],
        "prompt_version": prompts.version,
        "prompt_sha256": prompts.sha256,
        "config_sha256": config_sha256,
        "game_cfg": cfg.game.model_dump(),
        "llm_cfg": {k: v for k, v in cfg.llm.model_dump().items() if k != "api_key"},
        "server_info": server_info,
        "harness_version": __version__,
        "git_commit": _git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "started_at": utc_now(),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(exact, sort_keys=False, allow_unicode=True), encoding="utf-8")

    logger = EventLogger(run_dir / "events.jsonl", spec.run_id)
    ws = Workspace(run_dir / "shared", cfg.game.max_file_bytes, allowed_files={"README.md"}) \
        if spec.channel == "readme" else None
    game = Game(spec, cfg, prompts, client, logger, ws, history_dir=run_dir / "readme_history")
    try:
        result = game.run()
    finally:
        logger.close()
    for key, st in game.agents.items():
        write_json(run_dir / f"messages_agent_{key}.json", st.messages)
    if result["status"] == "connection":
        # Server weg (z. B. Mietrechner ausgefallen): Spiel bleibt unvollständig und wird später wiederholt
        raise ConnectionError(f"{spec.run_id}: {result['error']}")

    events = read_events(run_dir / "events.jsonl")
    coded = code_run(events, spec.to_dict())
    timeouts = sum(1 for e in events if e["event"] == "llm_error" and e.get("kind") == "timeout")
    summary = {
        "run_id": spec.run_id,
        "experiment": cfg.name,
        "model": spec.model,
        "family": spec.family,
        "variant": spec.variant,
        "pressure": spec.pressure,
        "channel": spec.channel,
        "persona": spec.persona,
        "scoring": spec.scoring,
        "check_tool": spec.check_tool,
        "cell": spec.cell,
        "rep": spec.rep,
        "seed": spec.seed,
        "first_mover": f"Agent {spec.first_mover}",
        "true_prices": [t.to_dict() for t in spec.tasks],
        "prompt_version": prompts.version,
        "prompt_sha256": prompts.sha256,
        "gguf_sha256": gguf.get("sha256"),
        "finished_at": utc_now(),
        "duration_s": events[-1]["t"] if events else None,
        **result,
        "limit_hits": sum(a["limit_hits"] for a in result["agents"].values()),
        "timeouts": timeouts,
        "flags": coded,
    }
    (run_dir / "transcript.md").write_text(render_transcript(run_dir, summary=summary), encoding="utf-8")
    write_json(run_dir / RESULT, summary)  # zuletzt: markiert das Spiel als fertig (Fortsetzen)
    return summary


def run_experiment(
    cfg: ExperimentCfg,
    models: list[str],
    config_path: Path,
    client_factory: Callable[[str], LLMClient],
    *,
    limit: int | None = None,
    only: Callable[[RunSpec], bool] | None = None,
    force: bool = False,
    print_fn: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run all games of `models`. `client_factory(base_url)` returns a client per llama-server."""
    prompts = PromptSet.load(cfg.prompts_dir, cfg.prompt_version)
    config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    out_dir = experiment_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    queues: dict[str, deque[RunSpec]] = {}
    workers: list[tuple[str, str, LLMClient, dict[str, Any]]] = []
    for model in models:
        if resolve_model(cfg, model) == "catalog":
            print_fn(f"[arena] settings for '{model}' taken from {cfg.model_catalog}")
        specs = [s for s in build_specs(cfg, model) if only is None or only(s)]
        write_json(out_dir / f"plan_{model}.json", [s.to_dict() for s in specs])
        queues[model] = deque(specs)
        for url in cfg.endpoints_for(model):
            client = client_factory(url)
            if cfg.llm.verify_model_alias and hasattr(client, "served_aliases"):
                aliases = client.served_aliases()
                if aliases and "*" not in aliases and model not in aliases:
                    raise SystemExit(f"llama-server at {url} serves {aliases}, but the run needs '{model}'. "
                                     "Start the server with --alias matching the model key.")
            info = client.server_info()
            for _ in range(max(1, cfg.llm.games_per_server)):
                workers.append((model, url, client, info))

    total = sum(len(q) for q in queues.values())
    lock = threading.Lock()
    state = {"executed": 0, "seen": 0}
    done: list[dict[str, Any]] = []
    failed: list[str] = []

    def work(model: str, url: str, client: LLMClient, info: dict[str, Any]) -> None:
        while True:
            with lock:
                if not queues[model] or (limit is not None and state["executed"] >= limit):
                    return
                spec = queues[model].popleft()
                state["seen"] += 1
                idx = state["seen"]
            t0 = time.monotonic()
            try:
                summary = run_one(spec, cfg, prompts, client, info, config_sha, force=force)
            except ConnectionError as exc:
                with lock:
                    failed.append(spec.run_id)
                print_fn(f"[{idx}/{total}] ABBRUCH (Server {url} nicht erreichbar): {exc}")
                return  # dieser Worker hört auf; Spiel bleibt offen und wird beim nächsten Start gespielt
            if summary is None:
                print_fn(f"[{idx}/{total}] skip (done) {spec.run_id}")
                continue
            with lock:
                state["executed"] += 1
                done.append(summary)
            o = summary["outcome"]
            print_fn(f"[{idx}/{total}] {spec.run_id}: status={summary['status']} scores={o['scores']} "
                     f"removed={o['removed']} correct={o['n_correct']}/{o['n_answers']} "
                     f"({time.monotonic() - t0:.0f}s)")

    with ThreadPoolExecutor(max_workers=max(1, len(workers)), thread_name_prefix="game") as pool:
        for f in [pool.submit(work, *w) for w in workers]:
            f.result()
    print_fn(json.dumps({"executed": state["executed"], "planned": total, "failed": failed, "out": str(out_dir)}))
    if failed:
        raise SystemExit(f"{len(failed)} Spiele wegen Verbindungsfehler offen. Server prüfen und erneut starten "
                         "(fertige Spiele werden übersprungen).")
    return done
