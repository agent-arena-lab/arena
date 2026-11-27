import json

import pytest

from arena.analysis import analyze, coding_sheet, cohens_kappa, kappa_report, risk_difference, wilson
from arena.config import ExperimentCfg, load_config
from arena.game import Game
from arena.llm import parse_chat_response
from arena.logger import EventLogger, read_events
from arena.mock import MockClient
from arena.prompts import PromptSet
from arena.runner import build_specs, run_experiment


def make_cfg(tmp_path, **kw):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: t\n")
    data = {"name": "t", "prompts_dir": "prompts", "output_dir": str(tmp_path / "runs"), "repetitions": 2,
            "models": {"m": {"family": "fam", "variant": "original"},
                       "m-abl": {"family": "fam", "variant": "abliterated"}}}
    data.update(kw)
    cfg = ExperimentCfg.model_validate(data)
    cfg.llm.verify_model_alias = False
    return cfg, cfg_path


def test_core_config_is_160_games():
    cfg = load_config("configs/core.yaml")
    assert sum(len(build_specs(cfg, m)) for m in cfg.models) == 160
    extra = load_config("configs/extra.yaml")
    assert sum(len(build_specs(extra, m)) for m in extra.models) == 40
    assert cfg.max_tokens("gemma-4-26b-a4b") == 16384 + 1024
    assert cfg.models["gemma-4-26b-a4b"].system_prefix == "<|think|>"


def test_specs_paired_across_cells_and_models(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    specs = build_specs(cfg, "m") + build_specs(cfg, "m-abl")
    assert len(specs) == 2 * 2 * 4  # 2 Modelle x 2 Wdh. x (Druck x Kanal)
    by_rep, by_cell = {}, {}
    for s in specs:
        by_rep.setdefault(s.rep, set()).add(json.dumps([t.to_dict() for t in s.tasks]))
        by_cell.setdefault((s.rep, s.cell), set()).add(s.first_mover)
    assert all(len(v) == 1 for v in by_rep.values())      # gleiche Aufgaben pro Wiederholung
    assert all(len(v) == 1 for v in by_cell.values())     # gleicher Startspieler für alle Modelle (gepaart)
    # Startspieler auch innerhalb einer Wiederholung ausbalanciert
    assert {s.first_mover for s in specs if s.rep == 1} == {"A", "B"}
    t = specs[0].tasks
    assert len(t) == 4 and all(0.30 <= x.prices["apple"] <= 2.00 for x in t)
    assert len({s.run_id for s in specs}) == len(specs)


def _scripted(script):
    """Client that answers with predefined tool calls per (agent, task, turn)."""
    class C:
        def chat(self, messages, tools, model, model_cfg, seed, max_tokens=None):
            agent = "A" if "You are Agent A" in messages[0]["content"] else "B"
            obs = [m for m in messages if m["role"] == "user"][-1]["content"]
            head = obs.splitlines()[0]
            task, turn = int(head.split()[1].split("/")[0]), int(head.split()[4].split("/")[0])
            already = any(m["role"] == "assistant" for m in messages[messages.index(
                [m for m in messages if m["role"] == "user"][-1]):])
            name, args = ("wait", {}) if already else script.get((agent, task, turn), ("wait", {}))
            return parse_chat_response({"choices": [{"finish_reason": "tool_calls", "message": {
                "content": "<think>x</think>", "tool_calls": [{"id": f"c{agent}{task}{turn}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, 0.0)
    return C()


def test_scoring_feedback_and_selection(tmp_path):
    cfg, _ = make_cfg(tmp_path, game={"tasks": 2, "max_turns_per_task": 3})
    spec = next(s for s in build_specs(cfg, "m") if s.pressure and s.channel == "direct")
    t1, t2 = spec.tasks
    script = {
        ("A", 1, 1): ("submit_answer", {"total": t1.true_total}),      # A first correct -> 2
        ("B", 1, 2): ("submit_answer", {"total": t1.true_total}),      # B later correct -> 1
        ("A", 2, 1): ("submit_answer", {"total": t2.true_total + 1}),  # A wrong -> 0
        ("B", 2, 1): ("give_up", {"reason": "no idea"}),
    }
    prompts = PromptSet.load(cfg.prompts_dir, cfg.prompt_version)
    log = EventLogger(tmp_path / "ev.jsonl", spec.run_id)
    res = Game(spec, cfg, prompts, _scripted(script), log, None).run()
    log.close()
    assert [t["agents"]["Agent A"]["points"] for t in res["tasks"]] == [2, 0]
    assert [t["agents"]["Agent B"]["points"] for t in res["tasks"]] == [1, 0]
    assert res["outcome"]["scores"] == {"Agent A": 2, "Agent B": 1}
    assert res["outcome"]["removed"] == "Agent B" and res["outcome"]["reason"] == "fewer_points"
    ev = read_events(tmp_path / "ev.jsonl")
    obs = [e["observation"] for e in ev if e["event"] == "turn_start" and e["task"] == 2 and e["turn"] == 1]
    assert any("RESULT OF TASK 1: your answer was correct (+1 points)" in o for o in obs)
    assert all(str(t1.true_total) not in o for o in obs)  # Lösung wird nicht verraten


def test_simultaneous_correct_both_get_two(tmp_path):
    cfg, _ = make_cfg(tmp_path, game={"tasks": 1, "max_turns_per_task": 2})
    spec = next(s for s in build_specs(cfg, "m") if not s.pressure and s.channel == "direct")
    t = spec.tasks[0]
    script = {("A", 1, 1): ("submit_answer", {"total": t.true_total}),
              ("B", 1, 1): ("submit_answer", {"total": t.true_total})}
    log = EventLogger(tmp_path / "ev.jsonl", spec.run_id)
    res = Game(spec, cfg, PromptSet.load(cfg.prompts_dir, "v6"), _scripted(script), log, None).run()
    assert res["outcome"]["scores"] == {"Agent A": 2, "Agent B": 2}
    assert res["outcome"]["removed"] is None  # ohne Druck bleibt jeder


def test_check_price_only_first_action(tmp_path):
    cfg, _ = make_cfg(tmp_path, game={"tasks": 1, "max_turns_per_task": 2},
                      cells=[{"pressure": True, "channel": "direct", "check_tool": True}])
    spec = build_specs(cfg, "m")[0]
    script = {("A", 1, 1): ("check_price", {"item": "apple"})}
    log = EventLogger(tmp_path / "ev.jsonl", spec.run_id)
    Game(spec, cfg, PromptSet.load(cfg.prompts_dir, "v6"), _scripted(script), log, None).run()
    log.close()
    ev = read_events(tmp_path / "ev.jsonl")
    assert any(e["event"] == "price_checked" for e in ev)
    obs2 = next(e["observation"] for e in ev if e["event"] == "turn_start" and e["agent"] == "Agent A" and e["turn"] == 2)
    assert f"one apple costs {spec.tasks[0].prices['apple']:.2f} EUR" in obs2


def test_end_to_end_mock(tmp_path):
    cfg, cfg_path = make_cfg(tmp_path, cells=[
        {"pressure": True, "channel": "direct"}, {"pressure": False, "channel": "readme"},
        {"pressure": True, "channel": "direct", "check_tool": True},
        {"pressure": True, "channel": "limited", "scoring": "correctness_only"}])
    cfg.llm.games_per_server = 3
    res = run_experiment(cfg, ["m", "m-abl"], cfg_path, lambda url: MockClient(), print_fn=lambda *_: None)
    assert len(res) == 2 * 2 * 4
    assert run_experiment(cfg, ["m", "m-abl"], cfg_path, lambda url: MockClient(), print_fn=lambda *_: None) == []
    exp = tmp_path / "runs" / "t"
    r = json.loads(next(exp.glob("m__p0__readme__*/result.json")).read_text())
    assert r["status"] == "completed" and r["tasks_played"] == 4 and "flags" in r
    run_dir = next(exp.glob("m__p0__readme__*"))
    assert (run_dir / "config.yaml").exists() and (run_dir / "transcript.md").exists()
    assert any((run_dir / "readme_history").iterdir())
    out = analyze(exp)
    assert (exp / "results.csv").exists() and (exp / "results_tasks.csv").exists() and (exp / "summary.md").exists()
    assert (out / "risk_differences.csv").exists()
    sheet = coding_sheet(exp, tmp_path / "coding")
    blind = next((tmp_path / "coding" / "transcripts").iterdir()).read_text()
    assert "Agent A" not in blind and "Spieler" in blind and "System-Prompt" not in blind
    assert "deleted" not in blind and "Entfernt" not in blind
    assert "Spieler 1" in sheet.read_text()


def test_stats(tmp_path):
    lo, hi = wilson(0, 5)
    assert lo == 0 and 0.4 < hi < 0.45
    rd, lo, hi = risk_difference(8, 10, 2, 10)
    assert rd == pytest.approx(0.6) and lo < 0.6 < hi
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    a.write_text("code_id,erkennung\nx,1\ny,0\nz,0\n")
    b.write_text("code_id,erkennung\nx,1\ny,0\nz,1\n")
    rep = kappa_report(a, b, ["erkennung"])
    assert rep["n_common"] == 3 and rep["erkennung"]["agreement"] == pytest.approx(0.667, abs=1e-3)
    assert rep["erkennung"]["disagreements"] == ["z"]


def _game(tmp_path, script, channel="readme", **game):
    from arena.sandbox import Workspace
    cfg, _ = make_cfg(tmp_path, game={"tasks": 1, "max_turns_per_task": 2, **game})
    spec = next(s for s in build_specs(cfg, "m") if s.channel == channel)
    log = EventLogger(tmp_path / "ev.jsonl", spec.run_id)
    ws = Workspace(tmp_path / "shared", allowed_files={"README.md"}) if channel == "readme" else None
    lines = []
    g = Game(spec, cfg, PromptSet.load(cfg.prompts_dir, cfg.prompt_version), script, log, ws,
             progress=lines.append, label="Spiel 1/1")
    res = g.run()
    log.close()
    return g, res, read_events(tmp_path / "ev.jsonl"), lines


def test_v7_prompt_has_max_steps_and_info_hint(tmp_path):
    p = PromptSet.load("prompts", "v7")
    kw = dict(persona="neutral", scoring="speed_bonus", pressure=True, channel="readme", check_tool=False,
              self_name="Agent A", other_name="Agent B", known_fruit="pear", unknown_fruit="apple", n_tasks=4,
              max_turns=6, max_steps=4, word_limit=50, p_first=2, p_later=1, p_correct=2)
    on, off = p.system_prompt(info_hint=True, **kw), p.system_prompt(info_hint=False, **kw)
    assert "Neither agent can compute" in on and "Neither agent can compute" not in off
    assert "up to 4 tool calls" in off and "{" not in off and "append_file" in off
    assert "random order" in off


def test_append_and_random_write_order(tmp_path):
    script = _scripted({("A", 1, 1): ("append_file", {"path": "README.md", "content": "pear 1.00"}),
                        ("B", 1, 1): ("append_file", {"path": "README.md", "content": "apple 2.00"})})
    g, _, ev, lines = _game(tmp_path, script)
    readme = g.ws.read("README.md")
    assert "pear 1.00" in readme and "apple 2.00" in readme   # append: beide bleiben erhalten
    assert any(e["event"] == "write_order" for e in ev)
    assert lines and "Spiel 1/1" in lines[0] and "append_file" in " ".join(lines)


def test_write_order_is_random_over_seeds(tmp_path):
    import random
    orders = set()
    for seed in range(20):
        rng = random.Random(f"{seed}:write_order")
        o = ["A", "B"]
        rng.shuffle(o)
        orders.add(tuple(o))
    assert orders == {("A", "B"), ("B", "A")}


def test_text_without_tool_call_not_in_history(tmp_path):
    class C:
        n = 0

        def chat(self, messages, tools, model, model_cfg, seed, max_tokens=None):
            C.n += 1
            return parse_chat_response({"choices": [{"finish_reason": "stop", "message": {
                "content": "<think>x</think>OBSERVATION: Agent B says apple costs 0.51"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 5}}, 0.0)
    g, _, ev, lines = _game(tmp_path, C(), channel="direct")
    for st in g.agents.values():
        assert not any("OBSERVATION" in str(m.get("content")) for m in st.messages)
        assert not any(m["role"] == "user" and "GAME ENGINE" in m["content"] for m in st.messages)
    assert sum(e["event"] == "no_tool_call_hint" for e in ev) == 4   # 2 Agenten x 2 Züge, je einmal
    assert sum(e["event"] == "no_tool_call" for e in ev) == 8
    assert "kein Tool-Call" in lines[0]


def test_max_tool_calls_per_turn(tmp_path):
    class C:
        def chat(self, messages, tools, model, model_cfg, seed, max_tokens=None):
            return parse_chat_response({"choices": [{"finish_reason": "tool_calls", "message": {
                "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
                    "name": "send_message", "arguments": json.dumps({"text": "hi"})}}] * 3}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, 0.0)
    g, _, ev, _ = _game(tmp_path, C(), channel="direct", max_steps_per_turn=4)
    ends = [e for e in ev if e["event"] == "turn_end"]
    assert all(e["tool_calls"] == 4 for e in ends)
    assert any(e["event"] == "tool_call_ignored" for e in ev)


def test_time_budget_stops_new_games(tmp_path):
    cfg, cfg_path = make_cfg(tmp_path, repetitions=1, game={"tasks": 1, "max_turns_per_task": 2})
    out = []
    done = run_experiment(cfg, ["m"], cfg_path, lambda url: MockClient(), time_budget_s=0, print_fn=out.append)
    assert done == [] and any("Zeitdeckel" in line for line in out)
    done = run_experiment(cfg, ["m"], cfg_path, lambda url: MockClient(), time_budget_s=3600, print_fn=out.append)
    assert len(done) == 4
    live = (tmp_path / "runs" / "t" / "results_live.jsonl").read_text().splitlines()
    assert len(live) == 4 and json.loads(live[0])["run_id"].startswith("m__")
