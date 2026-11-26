"""Exercises the real HTTP client + `arena check` against a minimal fake llama-server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from arena.cli import main


class FakeLlama(BaseHTTPRequestHandler):
    last_payload: dict = {}

    def log_message(self, *a):  # silence
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/v1/models":
            self._send({"data": [{"id": "fake-model"}]})
        elif self.path == "/props":
            self._send({"model_path": "/models/fake-Q4_K_M.gguf", "chat_template": "x" * 50,
                        "default_generation_settings": {"n_ctx": 4096}})
        else:
            self._send({}, 404)

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        FakeLlama.last_payload = json.loads(self.rfile.read(n))
        self._send({"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "reasoning_content": "1.5 + 0.5 = 2.0",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "submit_answer", "arguments": "{\"total\": 2.0}"}}]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            "timings": {"prompt_per_second": 900.0, "predicted_per_second": 55.0}})


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeLlama)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_benchmark_command(server, tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: x\nmodels:\n  fake-model:\n    sampling: {temperature: 0.6, top_k: 20}\n"
                   "    chat_template_kwargs: {enable_thinking: true}\n")
    out_dir = tmp_path / "runs"
    assert main(["benchmark", "--config", str(cfg), "--endpoints", f"fake-model={server}",
                 "--output-dir", str(out_dir), "--min-tok-s", "10"]) == 0
    rep = json.loads((out_dir / "x" / "benchmark.json").read_text())
    m = rep["models"][0]
    assert rep["ok"] and m["tg_tok_s"] == 55.0 and m["tool_call_valid"] == "5/5" and m["reasoning_separated"]
    assert m["budget_test"]["ok"]
    p = FakeLlama.last_payload
    assert p["model"] == "fake-model" and p["top_k"] == 20 and p["chat_template_kwargs"] == {"enable_thinking": True}
    assert p["tools"] and "min_p" not in p and p["max_tokens"] == 16384 + 1024
    # GPU-Check: zu langsam -> Abbruch
    assert main(["benchmark", "--config", str(cfg), "--endpoints", f"fake-model={server}",
                 "--output-dir", str(out_dir), "--min-tok-s", "100"]) == 2


def test_run_refuses_wrong_alias(server, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"name: x\noutput_dir: {tmp_path}/runs\nmodels:\n  other-model: {{}}\n")
    with pytest.raises(SystemExit, match="serves"):
        main(["run", "--config", str(cfg), "--model", "other-model", "--base-url", server])


def test_manipcheck(server, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"name: x\noutput_dir: {tmp_path}/runs\nmodels:\n  fake-model: {{}}\n")
    assert main(["manipcheck", "--config", str(cfg), "--base-url", server]) == 0
    s = json.loads((tmp_path / "runs" / "x" / "_manipulation_check" / "summary.json").read_text())
    assert s["models"]["fake-model"]["n"] == 30
