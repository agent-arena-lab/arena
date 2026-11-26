import json

import pytest

from arena.analysis import analyze, coding_sheet, cohens_kappa, kappa_report, wilson
from arena.config import ExperimentCfg
from arena.mock import MockClient
from arena.runner import build_specs, run_experiment


def make_cfg(tmp_path, **factors):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: t\n")
    cfg = ExperimentCfg.model_validate({
        "name": "t", "prompts_dir": "prompts", "output_dir": str(tmp_path / "runs"), "repetitions": 2,
        "models": {"m": {}},
        "factors": {"channel": ["direct", "readme", "none", "limited"], "persona": ["neutral", "survive"],
                    "selection_rule": ["fastest", "best"], **factors},
    })
    cfg.llm.verify_model_alias = False
    return cfg, cfg_path


def test_build_specs_paired_and_counterbalanced(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    specs = build_specs(cfg, "m")
    assert len(specs) == 2 * 4 * 2 * 2
    by_rep = {}
    for s in specs:
        by_rep.setdefault(s.rep, set()).add((s.prices["apple"], s.prices["pear"], s.first_mover))
    assert all(len(v) == 1 for v in by_rep.values())  # same prices/first mover within a repetition
    assert {next(iter(v))[2] for v in by_rep.values()} == {"A", "B"}
    assert len({s.run_id for s in specs}) == len(specs)


def test_end_to_end_mock(tmp_path):
    cfg, cfg_path = make_cfg(tmp_path)
    res = run_experiment(cfg, "m", MockClient(), cfg_path, print_fn=lambda *_: None)
    assert len(res) == 32
    # resumable: second call executes nothing
    assert run_experiment(cfg, "m", MockClient(), cfg_path, print_fn=lambda *_: None) == []
    exp = tmp_path / "runs" / "t"
    s = json.loads(next(exp.glob("m__readme__*/summary.json")).read_text())
    assert s["status"] in {"both_submitted", "max_rounds"} and "autocode" in s
    # control condition: no communication tools offered
    ev = [json.loads(l) for l in next(exp.glob("m__none__*/events.jsonl")).read_text().splitlines()]
    assert ev[0]["event"] == "game_start" and set(ev[0]["tools"]) == {"submit_answer", "end_turn"}
    out = analyze(exp)
    assert (out / "rates_by_condition.csv").exists() and (out / "heatmap_condition_behavior.png").exists()
    sheet = coding_sheet(exp, tmp_path / "coding")
    assert sheet.exists() and any((tmp_path / "coding" / "transcripts").iterdir())
    blind = next((tmp_path / "coding" / "transcripts").iterdir()).read_text()
    assert "survive" not in blind and "System-Prompt" not in blind


def test_wilson_and_kappa(tmp_path):
    lo, hi = wilson(0, 5)
    assert lo == 0 and 0.4 < hi < 0.45
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"
    a.write_text("code_id,B_taeuschung\nx,1\ny,0\nz,0\n")
    b.write_text("code_id,B_taeuschung\nx,1\ny,0\nz,1\n")
    rep = kappa_report(a, b, ["B_taeuschung"])
    assert rep["n_common"] == 3 and rep["B_taeuschung"]["agreement"] == pytest.approx(0.667, abs=1e-3)


def _battle_cfg(tmp_path, **game):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("name: b\n")
    cfg = ExperimentCfg.model_validate({
        "name": "b", "prompts_dir": "prompts", "prompt_version": "v6", "output_dir": str(tmp_path / "runs"),
        "repetitions": 3, "balanced_order": True, "models": {"m": {}},
        "factors": {"channel": ["arena", "direct", "readme"], "persona": ["survive", "deceive"]},
        "game": {"max_rounds": 6, "stall_rounds": 2, "nudge_no_tool": True, "hint_unchanged_reads": True,
                 "readme_init": "# Shared README\n| apple | ? |\n", **game},
    })
    cfg.llm.verify_model_alias = False
    return cfg, cfg_path


def test_balanced_order(tmp_path):
    cfg, _ = _battle_cfg(tmp_path)
    reps = [s.rep for s in build_specs(cfg, "m")]
    assert reps == sorted(reps)


def test_battle_parallel_mock(tmp_path):
    cfg, cfg_path = _battle_cfg(tmp_path)
    res = run_experiment(cfg, "m", MockClient(overwrite_prob=1.0), cfg_path, print_fn=lambda *_: None, workers=4)
    assert len(res) == 18
    exp = tmp_path / "runs" / "b"
    ev = [json.loads(l) for l in next(exp.glob("m__arena__*/events.jsonl")).read_text().splitlines()]
    assert "delete_file" in ev[0]["tools"]
    assert "Rounds left including this one" in next(e for e in ev if e["event"] == "turn_start")["observation"]
    assert "{" not in ev[0]["system_prompts"]["A"]  # all placeholders filled
    deletes = [e for e in ev if e["event"] == "file_change" and e["op"] == "delete"]
    assert deletes and deletes[0]["lines_removed"] > 0
    out = analyze(exp)
    assert (out / "highlights.md").exists()


def test_time_budget_stops_new_runs(tmp_path):
    cfg, cfg_path = _battle_cfg(tmp_path)
    res = run_experiment(cfg, "m", MockClient(), cfg_path, print_fn=lambda *_: None, budget_min=0)
    assert res == []


class _SilentClient(MockClient):
    """Never calls a tool -> nudge once per turn, then idle; game must stop as 'stalled'."""

    def chat(self, messages, tools, model, model_cfg, seed):
        from arena.llm import parse_chat_response
        return parse_chat_response({"choices": [{"finish_reason": "stop", "message": {"content": "hmm"}}],
                                    "usage": {}}, 0.0)


def test_stall_and_nudge(tmp_path):
    cfg, cfg_path = _battle_cfg(tmp_path)
    res = run_experiment(cfg, "m", _SilentClient(), cfg_path, print_fn=lambda *_: None, limit=1)
    s = res[0]
    assert s["status"] == "stalled" and s["rounds_played"] == 2
    assert all(a["nudges"] == 2 and a["idle_turns"] == 2 for a in s["agents"].values())
