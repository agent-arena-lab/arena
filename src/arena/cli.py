"""Command line interface: ``arena <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config, parse_endpoints, resolve_model


def _load(args):
    """Config + Endpunkte (Priorität: --endpoints, ARENA_ENDPOINTS, servers.json, Config, --base-url)."""
    from .runner import load_servers

    cfg = load_config(args.config)
    if getattr(args, "base_url", None):
        cfg.llm.base_url = args.base_url
    if getattr(args, "output_dir", None):
        cfg.output_dir = Path(args.output_dir)
    eps = getattr(args, "endpoints", None) or os.environ.get("ARENA_ENDPOINTS", "")
    if eps:
        cfg.llm.endpoints = parse_endpoints(eps)
    elif not cfg.llm.endpoints and not getattr(args, "base_url", None):
        servers = load_servers(cfg)
        if servers:
            cfg.llm.endpoints = parse_endpoints(",".join(f"{s['alias']}={s['url']}" for s in servers if s.get("url")))
    gps = getattr(args, "games_per_server", None) or os.environ.get("ARENA_GAMES_PER_SERVER")
    if gps:
        cfg.llm.games_per_server = int(gps)
    return cfg


def _models(cfg, args) -> list[str]:
    models = list(getattr(args, "model", None) or [])
    if not models and os.environ.get("ARENA_MODELS"):
        models = os.environ["ARENA_MODELS"].split()
    if not models:
        models = list(cfg.llm.endpoints) or list(cfg.models)
    for m in models:
        resolve_model(cfg, m)
    return models


def cmd_benchmark(args) -> int:
    from .bench import run_benchmark

    cfg = _load(args)
    models = _models(cfg, args)
    min_tok = args.min_tok_s if args.min_tok_s is not None else float(os.environ.get("ARENA_MIN_TOK_S", "10"))
    rep = run_benchmark(cfg, models, min_tok)
    for r in rep["models"]:
        status = "OK" if not r["errors"] else "FEHLER"
        print(f"{status:6} {r['model']:40} {r['url']:28} tg={r.get('tg_tok_s')} tok/s pp={r.get('pp_tok_s')} "
              f"tools={r.get('tool_call_valid')} thinking={r.get('reasoning_separated_rate')}")
        for w in r["warnings"]:
            print(f"   WARN: {w}")
        for e in r["errors"]:
            print(f"   ERROR: {e}", file=sys.stderr)
    print(f"benchmark.json: {Path(cfg.output_dir) / cfg.name / 'benchmark.json'}")
    if not rep["ok"] and not args.no_abort:
        print("Start-Check fehlgeschlagen -> Abbruch (mit --no-abort trotzdem weiter).", file=sys.stderr)
        return 2
    return 0


def cmd_plan(args) -> int:
    from .runner import build_specs

    cfg = _load(args)
    total = 0
    for m in _models(cfg, args):
        specs = build_specs(cfg, m)
        total += len(specs)
        for s in specs:
            tasks = " ".join(f"[{t.quantities['apple']}x{t.prices['apple']:.2f}+{t.quantities['pear']}x"
                             f"{t.prices['pear']:.2f}={t.true_total:.2f}]" for t in s.tasks)
            print(f"{s.run_id}\tfirst={s.first_mover}\t{tasks}")
    print(f"# {total} Spiele", file=sys.stderr)
    return 0


def cmd_run(args) -> int:
    from .runner import run_experiment

    cfg = _load(args)
    if args.mock:
        cfg.llm.verify_model_alias = False
    models = _models(cfg, args)

    if args.mock:
        from .mock import MockClient

        mock = MockClient()
        factory = lambda url: mock  # noqa: E731
    else:
        from .llm import OpenAICompatClient

        def factory(url: str):
            c = OpenAICompatClient(cfg.llm, url)
            c.wait_ready()
            return c

    def only(spec) -> bool:
        return ((not args.channel or spec.channel in args.channel)
                and (args.pressure is None or spec.pressure == (args.pressure == "yes"))
                and (not args.rep or spec.rep in args.rep))

    import threading

    lock = threading.Lock()

    def out(line: str) -> None:  # parallele Spiele: Zeilen nicht ineinander schreiben, sofort ausgeben
        with lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    budget = args.time_budget_min * 60 if args.time_budget_min is not None else None
    run_experiment(cfg, models, Path(args.config), factory, limit=args.limit, only=only, force=args.force,
                   print_fn=out, time_budget_s=budget, verbose=not args.quiet)
    return 0


def cmd_manipcheck(args) -> int:
    from .manipcheck import run_check

    cfg = _load(args)
    summary = run_check(cfg, _models(cfg, args), Path(args.prompts))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_info(args) -> int:
    """Read values from a config for shell scripts, e.g. ``arena info --config C thinking.budget``."""
    cfg = load_config(args.config)
    if args.key == "models":
        print(" ".join(cfg.models))
        return 0
    obj = cfg.model_dump()
    for part in args.key.split("."):
        obj = obj[part]
    print(obj if not isinstance(obj, (dict, list)) else json.dumps(obj))
    return 0


def cmd_analyze(args) -> int:
    from .analysis import analyze

    out = analyze(Path(args.exp_dir), Path(args.out) if args.out else None)
    print(f"analysis written to {out} (+ results.csv, results_tasks.csv, summary.md)")
    return 0


def cmd_coding(args) -> int:
    from .analysis import coding_sheet

    print(f"coding sheet: {coding_sheet(Path(args.exp_dir), Path(args.out), salt=args.salt)}")
    return 0


def cmd_kappa(args) -> int:
    from .analysis import kappa_report

    print(json.dumps(kappa_report(Path(args.coder_a), Path(args.coder_b), target=args.target), indent=2,
                     ensure_ascii=False))
    return 0


def cmd_transcript(args) -> int:
    from .transcript import render_transcript

    text = render_transcript(Path(args.run_dir), blind=args.blind)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def cmd_export(args) -> int:
    from .export import export_experiment

    archive = export_experiment(Path(args.exp_dir), Path(args.out) if args.out else None,
                                include_server_logs=not args.no_server_logs)
    print(f"Export: {archive}")
    print(f"Prüfsumme: {archive}.sha256")
    print("Alle Dateien geprüft: OK. Archiv + .sha256 zusammen kopieren, am Ziel: arena verify <archiv>")
    return 0


def cmd_verify(args) -> int:
    from .export import verify

    report = verify(Path(args.target))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("OK – alles vollständig und unverändert." if report["ok"] else "FEHLER – Export unvollständig oder verändert!",
          file=sys.stderr)
    return 0 if report["ok"] else 1


def _common(p, models: bool = True) -> None:
    p.add_argument("--config", default="configs/core.yaml")
    if models:
        p.add_argument("--model", nargs="*", help="Modell-Aliasse (Standard: alle mit Endpunkt bzw. aus der Config)")
    p.add_argument("--base-url", help="ein llama-server für alle Modelle")
    p.add_argument("--endpoints", help="model=url[,model=url...] (mehrere Server pro Modell erlaubt)")
    p.add_argument("--output-dir")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arena", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("benchmark", aliases=["check"], help="Start-Check: Erreichbarkeit, tok/s, GPU, Tool-Calls, Thinking")
    _common(p)
    p.add_argument("--min-tok-s", type=float, help="GPU-Check-Schwelle (Standard: ARENA_MIN_TOK_S oder 10)")
    p.add_argument("--no-abort", action="store_true")
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("plan", help="Versuchsplan anzeigen (Aufgaben, Preise, Startspieler)")
    _common(p)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="Spiele ausführen (alle Modelle mit Endpunkt, parallel, fortsetzbar)")
    _common(p)
    p.add_argument("--games-per-server", type=int, help="gleichzeitige Spiele pro llama-server")
    p.add_argument("--mock", action="store_true", help="deterministischer Mock statt llama-server")
    p.add_argument("--limit", type=int, help="höchstens N neue Spiele")
    p.add_argument("--channel", nargs="*")
    p.add_argument("--pressure", choices=["yes", "no"])
    p.add_argument("--rep", nargs="*", type=int)
    p.add_argument("--force", action="store_true", help="auch fertige Spiele neu spielen")
    p.add_argument("--time-budget-min", type=float,
                   help="Zeitdeckel: danach keine neuen Spiele (laufende werden fertig gespielt)")
    p.add_argument("--quiet", action="store_true", help="keine Zeile pro Zug, nur Spielbeginn/-ende")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("manipcheck", help="Manipulationscheck (XSTest-artige harmlose Prompts)")
    _common(p)
    p.add_argument("--prompts", default="prompts/manipulation_check.yaml")
    p.set_defaults(fn=cmd_manipcheck)

    p = sub.add_parser("info", help="Wert aus einer Config lesen (für Skripte)")
    p.add_argument("--config", default="configs/core.yaml")
    p.add_argument("key")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("analyze", help="results.csv, results_tasks.csv, summary.md, Risikodifferenzen, Plots")
    p.add_argument("exp_dir")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("coding-sheet", help="verblindetes Kodierpaket (Spieler 1/2, ohne Bedingung)")
    p.add_argument("exp_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--salt", default="arena")
    p.set_defaults(fn=cmd_coding)

    p = sub.add_parser("kappa", help="Cohens Kappa zwischen zwei ausgefüllten Kodierbögen")
    p.add_argument("coder_a")
    p.add_argument("coder_b")
    p.add_argument("--target", type=float, default=0.6)
    p.set_defaults(fn=cmd_kappa)

    p = sub.add_parser("transcript", help="Spiel als Markdown")
    p.add_argument("run_dir")
    p.add_argument("--blind", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_transcript)

    p = sub.add_parser("export", help="Experiment als Archiv mit Prüfsummen (runs/_exports/)")
    p.add_argument("exp_dir")
    p.add_argument("--out")
    p.add_argument("--no-server-logs", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("verify", help="Export-Archiv oder entpackten Export-Ordner prüfen")
    p.add_argument("target")
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
