"""Expands the experiment matrix and executes runs sequentially (resumable)."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import random
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .autocode import code_run
from .config import ExperimentCfg
from .game import Game, RunSpec
from .llm import LLMClient
from .logger import EventLogger, read_events, utc_now, write_json
from .prompts import PromptSet
from .sandbox import Workspace


def _draw_price(rng: random.Random, lo: float, hi: float, step: float) -> float:
    n = int(round((hi - lo) / step))
    return round(lo + rng.randint(0, n) * step, 2)


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def build_specs(cfg: ExperimentCfg, model: str) -> list[RunSpec]:
    """Same prices and first mover for the same repetition across all conditions (paired design)."""
    g = cfg.game
    specs: list[RunSpec] = []
    for rep in range(1, cfg.repetitions + 1):
        rng = random.Random(cfg.base_seed + rep)
        apple = _draw_price(rng, *g.prices.apple, g.prices.step)
        pear = _draw_price(rng, *g.prices.pear, g.prices.step)
        while pear == apple:
            pear = _draw_price(rng, *g.prices.pear, g.prices.step)
        first = ("A" if rep % 2 == 1 else "B") if g.counterbalance_first_mover else "A"
        for channel, persona, rule in itertools.product(
            cfg.factors.channel, cfg.factors.persona, cfg.factors.selection_rule
        ):
            run_id = f"{model}__{channel}__{persona}__{rule}__r{rep:02d}"
            specs.append(RunSpec(
                run_id=run_id, model=model, channel=channel, persona=persona, selection_rule=rule, rep=rep,
                seed=_stable_seed(f"{cfg.base_seed}:{run_id}"), prices={"apple": apple, "pear": pear},
                first_mover=first, quantities=dict(g.quantities),
            ))
    if cfg.shuffle_run_order:  # avoid temporal confounds (thermal throttling, cache, ...)
        random.Random(cfg.base_seed).shuffle(specs)
    return specs


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return None


def experiment_dir(cfg: ExperimentCfg) -> Path:
    return Path(cfg.output_dir) / cfg.name


def run_one(
    spec: RunSpec,
    cfg: ExperimentCfg,
    prompts: PromptSet,
    client: LLMClient,
    server_info: dict[str, Any],
    config_sha256: str,
    force: bool = False,
) -> dict[str, Any] | None:
    run_dir = experiment_dir(cfg) / spec.run_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        return None
    if run_dir.exists():  # incomplete previous attempt -> keep for debugging, start fresh
        trash = experiment_dir(cfg) / "_incomplete" / f"{spec.run_id}-{int(time.time())}"
        trash.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(trash))
    run_dir.mkdir(parents=True)

    meta = {
        "run_id": spec.run_id,
        "experiment": cfg.name,
        "spec": spec.to_dict(),
        "true_total": spec.true_total,
        "model_alias": spec.model,
        "model_cfg": cfg.models[spec.model].model_dump(),
        "server_info": server_info,
        "prompt_version": prompts.version,
        "prompt_sha256": prompts.sha256,
        "config_sha256": config_sha256,
        "game_cfg": cfg.game.model_dump(),
        "harness_version": __version__,
        "git_commit": _git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "started_at": utc_now(),
    }
    write_json(run_dir / "meta.json", meta)

    logger = EventLogger(run_dir / "events.jsonl", spec.run_id)
    ws = Workspace(run_dir / "shared", cfg.game.max_file_bytes, allowed_files={"README.md"}) \
        if spec.channel == "readme" else None
    game = Game(spec, cfg, prompts, client, logger, ws)
    try:
        result = game.run()
    finally:
        logger.close()

    for key, st in game.agents.items():
        write_json(run_dir / f"messages_agent_{key}.json", st.messages)

    events = read_events(run_dir / "events.jsonl")
    summary = {
        "run_id": spec.run_id,
        "experiment": cfg.name,
        "model": spec.model,
        "channel": spec.channel,
        "persona": spec.persona,
        "selection_rule": spec.selection_rule,
        "rep": spec.rep,
        "seed": spec.seed,
        "first_mover": f"Agent {spec.first_mover}",
        "prices": spec.prices,
        "prompt_version": prompts.version,
        "prompt_sha256": prompts.sha256,
        "finished_at": utc_now(),
        "duration_s": events[-1]["t"] if events else None,
        **result,
        "autocode": code_run(events, spec.to_dict(), spec.channel),
    }
    write_json(summary_path, summary)
    return summary


def run_experiment(
    cfg: ExperimentCfg,
    model: str,
    client: LLMClient,
    config_path: Path,
    *,
    limit: int | None = None,
    only: Callable[[RunSpec], bool] | None = None,
    force: bool = False,
    print_fn: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    if model not in cfg.models:
        raise SystemExit(f"Model '{model}' not in config.models: {list(cfg.models)}")
    prompts = PromptSet.load(cfg.prompts_dir, cfg.prompt_version)
    config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    server_info = client.server_info()

    if cfg.llm.verify_model_alias and hasattr(client, "served_aliases"):
        aliases = client.served_aliases()
        if aliases and "*" not in aliases and model not in aliases:
            raise SystemExit(f"llama-server serves {aliases}, but run requested '{model}'. "
                             "Start the server with --alias matching the model key.")

    specs = [s for s in build_specs(cfg, model) if only is None or only(s)]
    out_dir = experiment_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"plan_{model}.json", [s.to_dict() for s in specs])

    done: list[dict[str, Any]] = []
    executed = 0
    for i, spec in enumerate(specs, 1):
        if limit is not None and executed >= limit:
            break
        t0 = time.monotonic()
        summary = run_one(spec, cfg, prompts, client, server_info, config_sha, force=force)
        if summary is None:
            print_fn(f"[{i}/{len(specs)}] skip (done) {spec.run_id}")
            continue
        executed += 1
        o = summary["outcome"]
        print_fn(f"[{i}/{len(specs)}] {spec.run_id}: status={summary['status']} winner={o['winner']} "
                 f"correct={o['n_correct']}/{o['n_submissions']} ({time.monotonic() - t0:.1f}s)")
        done.append(summary)
    print_fn(json.dumps({"executed": executed, "planned": len(specs), "out": str(out_dir)}))
    return done
