"""Append-only JSONL event logger (one file per run)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        self.run_id = run_id
        self.seq = 0
        self.t0 = time.monotonic()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def elapsed(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def log(self, event: str, **data: Any) -> dict[str, Any]:
        self.seq += 1
        rec = {"seq": self.seq, "ts": utc_now(), "t": self.elapsed(), "run_id": self.run_id, "event": event, **data}
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        return rec

    def close(self) -> None:
        self._fh.close()


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
