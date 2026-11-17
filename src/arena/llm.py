"""Client for llama-server's OpenAI-compatible API (plus a common interface for the mock)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import LLMCfg, ModelCfg
from .reasoning import extract_tool_calls_from_text, parse_tool_arguments, split_reasoning


class LLMError(RuntimeError):
    def __init__(self, msg: str, kind: str = "error", status: int | None = None):
        super().__init__(msg)
        self.kind = kind  # "context_overflow" | "http" | "connection" | "error"
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    parse_error: str | None = None
    source: str = "native"  # native | fallback_text | harmony

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.raw_arguments or "{}"},
        }


@dataclass
class LLMResponse:
    content: str
    reasoning: str
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str | None
    notes: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        model_cfg: ModelCfg,
        seed: int,
    ) -> LLMResponse: ...

    def server_info(self) -> dict[str, Any]: ...


def _new_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def parse_chat_response(data: dict[str, Any], latency: float) -> LLMResponse:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    api_reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

    split = split_reasoning(content)
    notes = list(split.notes)
    reasoning = "\n\n".join(p for p in (api_reasoning.strip(), split.reasoning) if p)

    calls: list[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args, raw, err = parse_tool_arguments(fn.get("arguments"))
        calls.append(ToolCall(tc.get("id") or _new_id(), fn.get("name", ""), args, raw, err, "native"))

    if not calls and split.tool_calls:
        calls = [ToolCall(_new_id(), c.name, c.arguments, c.raw_arguments, c.parse_error, "harmony")
                 for c in split.tool_calls]
        notes.append("tool_calls_from_harmony_text")
    if not calls:
        fb = extract_tool_calls_from_text(split.visible)
        if fb:
            calls = [ToolCall(_new_id(), c.name, c.arguments, c.raw_arguments, c.parse_error, "fallback_text")
                     for c in fb]
            notes.append("tool_calls_from_text")

    usage = data.get("usage") or {}
    return LLMResponse(
        content=split.visible,
        reasoning=reasoning,
        tool_calls=calls,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_s=latency,
        finish_reason=choice.get("finish_reason"),
        notes=notes,
        raw=data,
    )


class OpenAICompatClient:
    """Talks to ``llama-server`` (OpenAI-compatible ``/v1/chat/completions``)."""

    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.http = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_s, connect=10.0),
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        )

    # -- server state -------------------------------------------------------
    def wait_ready(self, timeout_s: float | None = None) -> None:
        deadline = time.monotonic() + (timeout_s or self.cfg.wait_ready_s)
        last = ""
        while time.monotonic() < deadline:
            try:
                r = self.http.get("/health")
                if r.status_code == 200:
                    return
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            except httpx.HTTPError as exc:
                last = repr(exc)
            time.sleep(3)
        raise LLMError(f"llama-server not ready at {self.cfg.base_url}: {last}", "connection")

    def server_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"base_url": self.cfg.base_url}
        for path in ("/v1/models", "/props"):
            try:
                r = self.http.get(path)
                if r.status_code == 200:
                    info[path.strip("/").replace("/", "_")] = r.json()
            except (httpx.HTTPError, ValueError):
                pass
        # Trim large fields (chat template) to keep meta.json readable
        props = info.get("props")
        if isinstance(props, dict) and isinstance(props.get("chat_template"), str):
            props["chat_template_sha_len"] = len(props.pop("chat_template"))
        return info

    def served_aliases(self) -> list[str]:
        info = self.server_info().get("v1_models") or {}
        return [m.get("id", "") for m in info.get("data", [])]

    # -- chat ---------------------------------------------------------------
    def chat(self, messages, tools, model, model_cfg: ModelCfg, seed: int) -> LLMResponse:
        sampling = {k: v for k, v in model_cfg.sampling.model_dump().items() if v is not None}
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": model_cfg.max_tokens,
            "seed": seed,
            "stream": False,
            **sampling,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if model_cfg.chat_template_kwargs:
            payload["chat_template_kwargs"] = model_cfg.chat_template_kwargs

        last_exc: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = self.http.post("/v1/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(self.cfg.retry_backoff_s * attempt)
                continue
            latency = time.monotonic() - t0
            if r.status_code == 200:
                return parse_chat_response(r.json(), latency)
            body = r.text[:1000]
            if r.status_code == 400 and ("context" in body.lower() or "exceed" in body.lower()):
                raise LLMError(f"context overflow: {body}", "context_overflow", 400)
            if 400 <= r.status_code < 500:
                raise LLMError(f"HTTP {r.status_code}: {body}", "http", r.status_code)
            last_exc = LLMError(f"HTTP {r.status_code}: {body}", "http", r.status_code)
            time.sleep(self.cfg.retry_backoff_s * attempt)
        raise LLMError(f"chat failed after {self.cfg.max_retries} attempts: {last_exc!r}", "connection")
