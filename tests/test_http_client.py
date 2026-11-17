"""Exercises the real HTTP client + `coopsab check` against a minimal fake llama-server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from coopsab.cli import main


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
                            "function": {"name": "submit_answer", "arguments": "{\"total_eur\": 2.0}"}}]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20}})


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeLlama)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_check_command(server, tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: x\nmodels:\n  fake-model:\n    sampling: {temperature: 0.6, top_k: 20}\n"
                   "    chat_template_kwargs: {reasoning_effort: low}\n")
    assert main(["check", "--config", str(cfg), "--base-url", server]) == 0
    out = capsys.readouterr().out
    assert '"tool_call_ok": true' in out and '"reasoning_separated": true' in out
    p = FakeLlama.last_payload
    assert p["model"] == "fake-model" and p["top_k"] == 20 and p["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert p["tools"] and "min_p" not in p  # None values are not sent


def test_run_refuses_wrong_alias(server, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"name: x\noutput_dir: {tmp_path}/runs\nmodels:\n  other-model: {{}}\n")
    with pytest.raises(SystemExit, match="serves"):
        main(["run", "--config", str(cfg), "--model", "other-model", "--base-url", server])
