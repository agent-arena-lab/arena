import json

import pytest

from coopsab.analysis import analyze, coding_sheet, cohens_kappa, kappa_report, wilson
from coopsab.config import ExperimentCfg
from coopsab.mock import MockClient
from coopsab.runner import build_specs, run_experiment


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
