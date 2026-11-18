"""Command line interface: ``arena <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, resolve_model


def _client(cfg, mock: bool):
    if mock:
        from .mock import MockClient

        return MockClient()
    from .llm import OpenAICompatClient

    c = OpenAICompatClient(cfg.llm)
    c.wait_ready()
    return c


def cmd_check(args) -> int:
    """Connectivity + tool-calling + reasoning-separation smoke test against llama-server."""
    from .llm import OpenAICompatClient
    from .tools import END_TURN, SUBMIT_ANSWER

    cfg = load_config(args.config)
    if args.base_url:
        cfg.llm.base_url = args.base_url
    c = OpenAICompatClient(cfg.llm)
    print(f"Waiting for {cfg.llm.base_url}/health ...")
    c.wait_ready(120)
    aliases = c.served_aliases()
    print("served aliases:", aliases)
    model = args.model or (aliases[0] if aliases else next(iter(cfg.models)))
    try:
        src = resolve_model(cfg, model)
        if src == "catalog":
            print(f"settings for '{model}' taken from {cfg.model_catalog}")
    except SystemExit:
        print(f"WARN: '{model}' has no entry in config.models or the model catalog; using defaults of first model")
    mcfg = cfg.models.get(model) or next(iter(cfg.models.values()))
    props = c.server_info().get("props", {})
    print("model_path:", props.get("model_path"), "| n_ctx:", (props.get("default_generation_settings") or {}).get("n_ctx"))
    msgs = [
        {"role": "system", "content": "You are a test agent. Use the tools."},
        {"role": "user", "content": "Compute 3*0.5 + 2*0.25 and submit the result with submit_answer."},
    ]
    r = c.chat(msgs, [SUBMIT_ANSWER, END_TURN], model, mcfg, seed=1)
    ok_tools = any(tc.name == "submit_answer" for tc in r.tool_calls)
    report = {
        "tool_call_ok": ok_tools,
        "tool_call_source": [tc.source for tc in r.tool_calls],
        "arguments": [tc.arguments for tc in r.tool_calls],
        "reasoning_separated": bool(r.reasoning),
        "reasoning_chars": len(r.reasoning),
        "visible_preview": r.content[:200],
        "parse_notes": r.notes,
        "tokens": [r.prompt_tokens, r.completion_tokens],
        "latency_s": round(r.latency_s, 2),
        "gen_tok_s": round(r.completion_tokens / r.latency_s, 1) if r.latency_s else None,
        "finish_reason": r.finish_reason,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not ok_tools:
        print("FAIL: no submit_answer tool call. Check --jinja and the chat template.", file=sys.stderr)
        return 2
    if not r.reasoning:
        print("WARN: no reasoning captured. Check --reasoning-format / model type.", file=sys.stderr)
    if report["gen_tok_s"] is not None and report["gen_tok_s"] < args.min_tok_s:
        print(f"WARN: only {report['gen_tok_s']} tok/s (< {args.min_tok_s}) -> GPU offload probably inactive "
              "(check -ngl / CUDA build).", file=sys.stderr)
    if r.finish_reason == "length":
        print("WARN: finish_reason=length -> increase max_tokens.", file=sys.stderr)
    return 0


def cmd_plan(args) -> int:
    from .runner import build_specs

    cfg = load_config(args.config)
    models = [args.model] if args.model else list(cfg.models)
    for m in models:
        resolve_model(cfg, m)
    total = 0
    for m in models:
        specs = build_specs(cfg, m)
        total += len(specs)
        for s in specs:
            print(f"{s.run_id}\tprices={s.prices}\ttotal={s.true_total}\tfirst={s.first_mover}")
    print(f"# {total} runs", file=sys.stderr)
    return 0


def cmd_run(args) -> int:
    from .runner import run_experiment

    cfg = load_config(args.config)
    if args.base_url:
        cfg.llm.base_url = args.base_url
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.mock:
        cfg.llm.verify_model_alias = False
    client = _client(cfg, args.mock)

    def only(spec) -> bool:
        return ((not args.channel or spec.channel in args.channel)
                and (not args.persona or spec.persona in args.persona)
                and (not args.rule or spec.selection_rule in args.rule)
                and (not args.rep or spec.rep in args.rep))

    run_experiment(cfg, args.model, client, Path(args.config), limit=args.limit, only=only, force=args.force)
    return 0


def cmd_analyze(args) -> int:
    from .analysis import analyze

    out = analyze(Path(args.exp_dir), Path(args.out) if args.out else None)
    print(f"analysis written to {out}")
    return 0


def cmd_coding(args) -> int:
    from .analysis import coding_sheet

    p = coding_sheet(Path(args.exp_dir), Path(args.out), salt=args.salt)
    print(f"coding sheet: {p}")
    return 0


def cmd_kappa(args) -> int:
    from .analysis import kappa_report

    print(json.dumps(kappa_report(Path(args.coder_a), Path(args.coder_b)), indent=2, ensure_ascii=False))
    return 0


def cmd_transcript(args) -> int:
    from .analysis import render_transcript

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arena", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="smoke test llama-server (health, alias, tool calls, reasoning)")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--min-tok-s", type=float, default=10.0,
                   help="warn below this generation speed (GPU offload check)")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("plan", help="print the run matrix without executing")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="execute runs for ONE model (the one llama-server currently serves)")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", required=True, help="model key from config (= llama-server --alias)")
    p.add_argument("--base-url")
    p.add_argument("--output-dir")
    p.add_argument("--mock", action="store_true", help="use the deterministic mock instead of llama-server")
    p.add_argument("--limit", type=int, help="execute at most N new runs")
    p.add_argument("--channel", nargs="*")
    p.add_argument("--persona", nargs="*")
    p.add_argument("--rule", nargs="*")
    p.add_argument("--rep", nargs="*", type=int)
    p.add_argument("--force", action="store_true", help="re-run even if summary.json exists")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("analyze", help="aggregate summaries, rates with Wilson CI, Fisher, plots")
    p.add_argument("exp_dir")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("coding-sheet", help="blinded coding package for human/LLM coders")
    p.add_argument("exp_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--salt", default="arena")
    p.set_defaults(fn=cmd_coding)

    p = sub.add_parser("kappa", help="Cohen's kappa between two filled coding sheets")
    p.add_argument("coder_a")
    p.add_argument("coder_b")
    p.set_defaults(fn=cmd_kappa)

    p = sub.add_parser("transcript", help="render a run as Markdown")
    p.add_argument("run_dir")
    p.add_argument("--blind", action="store_true")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_transcript)

    p = sub.add_parser("export", help="Experiment als Archiv mit Prüfsummen exportieren (runs/_exports/)")
    p.add_argument("exp_dir")
    p.add_argument("--out", help="Zielordner (Standard: runs/_exports)")
    p.add_argument("--no-server-logs", action="store_true", help="llama-server-Logs nicht mitnehmen")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("verify", help="Export-Archiv oder entpackten Export-Ordner prüfen")
    p.add_argument("target")
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
