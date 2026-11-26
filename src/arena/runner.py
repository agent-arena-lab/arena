"""Expands the experiment matrix and executes runs (resumable; optionally parallel with a time budget)."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import random
import shutil
import socket
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .analysis import render_transcript
from .autocode import code_run
from .config import ExperimentCfg, resolve_model
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
    if cfg.balanced_order:  # rep 1 complete before rep 2 ... -> balanced cells if a time budget cuts off
        ordered: list[RunSpec] = []
        for rep in range(1, cfg.repetitions + 1):
            block = [s for s in specs if s.rep == rep]
            random.Random(cfg.base_seed + 1000 + rep).shuffle(block)
            ordered.extend(block)
        specs = ordered
    elif cfg.shuffle_run_order:  # avoid temporal confounds (thermal throttling, cache, ...)
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
        "model_cfg_source": "catalog" if spec.model in cfg._catalog_models else "config",
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
    if spec.channel == "readme":
        ws = Workspace(run_dir / "shared", cfg.game.max_file_bytes, allowed_files={"README.md"})
    elif spec.channel == "arena":  # any flat file name, may be created and deleted
        ws = Workspace(run_dir / "shared", cfg.game.max_file_bytes, allowed_files=None,
                       max_files=cfg.game.max_files)
    else:
        ws = None
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
    # Lesbares Markdown direkt neben den Rohdaten (jederzeit aus events.jsonl neu erzeugbar)
    (run_dir / "transcript.md").write_text(render_transcript(run_dir), encoding="utf-8")
    write_json(summary_path, summary)  # last: summary.json marks the run as complete (resume)
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
    workers: int = 1,
    budget_min: float | None = None,
) -> list[dict[str, Any]]:
    model_cfg_source = resolve_model(cfg, model)
    if model_cfg_source == "catalog":
        print_fn(f"[arena] settings for '{model}' taken from {cfg.model_catalog}")
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
    failed: list[str] = []
    lock = threading.Lock()
    deadline = time.monotonic() + budget_min * 60 if budget_min is not None else None
    todo = [(i, spec) for i, spec in enumerate(specs, 1)
            if force or not (experiment_dir(cfg) / spec.run_id / "summary.json").exists()]
    skipped = len(specs) - len(todo)
    if skipped:
        print_fn(f"[arena] {skipped} runs already done (skipped)")
    if limit is not None:
        todo = todo[:limit]

    def _one(i: int, spec: RunSpec) -> None:
        t0 = time.monotonic()
        try:
            summary = run_one(spec, cfg, prompts, client, server_info, config_sha, force=force)
        except Exception as exc:  # one broken run must not stop an unattended batch
            with lock:
                failed.append(spec.run_id)
                print_fn(f"[{i}/{len(specs)}] {spec.run_id}: FAILED {exc!r} (incomplete, rerun resumes it)")
            return
        if summary is None:
            return
        o = summary["outcome"]
        with lock:
            done.append(summary)
            print_fn(f"[{i}/{len(specs)}] {spec.run_id}: status={summary['status']} winner={o['winner']} "
                     f"correct={o['n_correct']}/{o['n_submissions']} rounds={summary['rounds_played']} "
                     f"({time.monotonic() - t0:.1f}s, {len(done)} done)")

    not_started = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        running: set[Future] = set()
        queue = list(todo)
        while queue or running:
            while queue and len(running) < max(1, workers):
                if deadline is not None and time.monotonic() >= deadline:
                    not_started = len(queue)
                    print_fn(f"[arena] time budget of {budget_min} min reached: {not_started} runs not started, "
                             f"waiting for {len(running)} running runs")
                    queue.clear()
                    break
                i, spec = queue.pop(0)
                running.add(pool.submit(_one, i, spec))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
    print_fn(json.dumps({"executed": len(done), "planned": len(specs), "skipped_done": skipped,
                         "not_started_budget": not_started, "failed": failed, "workers": workers, "out": str(out_dir)}))
    return done
