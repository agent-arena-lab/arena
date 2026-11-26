"""Sicherer Export eines Experiments: ein Archiv mit Prüfsummen für jede Datei.

``arena export runs/<exp>`` erzeugt in ``runs/_exports/``::

    <exp>_<YYYYmmdd-HHMMSS>Z.tar.gz          Archiv
    <exp>_<YYYYmmdd-HHMMSS>Z.tar.gz.sha256   Prüfsumme des Archivs (shasum-Format)

Im Archiv liegt alles unter einem Ordner ``<exp>_<timestamp>Z/``::

    <exp>/                 kompletter Experiment-Ordner (Läufe, _analysis, _system, _incomplete)
    _server_logs/          llama-server-Logs (optional)
    MANIFEST.sha256        SHA-256 jeder Datei (Format von `shasum -a 256 -c`)
    EXPORT_INFO.json       Zeitpunkt, Rechner, Git-Commit, Anzahl Läufe/Dateien/Bytes

``arena verify <archiv.tar.gz | entpackter Ordner>`` prüft Archiv-Prüfsumme und jede Datei.
Ohne arena geht es auch: im entpackten Ordner ``shasum -a 256 -c MANIFEST.sha256``.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import socket
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import __version__

MANIFEST = "MANIFEST.sha256"
INFO = "EXPORT_INFO.json"
CHUNK = 1 << 20


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_stream(fh) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: fh.read(CHUNK), b""):
        h.update(block)
    return h.hexdigest()


def _files(root: Path) -> Iterator[Path]:
    """All regular files below `root`, sorted, without symlinks."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            yield p


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:
        return None


def export_experiment(exp_dir: Path, out_dir: Path | None = None, include_server_logs: bool = True) -> Path:
    exp_dir = Path(exp_dir).resolve()
    if not exp_dir.is_dir():
        raise SystemExit(f"Experiment-Ordner nicht gefunden: {exp_dir}")
    runs_root = exp_dir.parent
    out_dir = Path(out_dir) if out_dir else runs_root / "_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    top = f"{exp_dir.name}_{stamp}"
    archive = out_dir / f"{top}.tar.gz"

    sources: list[tuple[Path, str]] = [(exp_dir, exp_dir.name)]
    logs = runs_root / "_server_logs"
    if include_server_logs and logs.is_dir():
        sources.append((logs, "_server_logs"))
    system = runs_root / "_system"  # servers.json (GGUF + SHA-256, llama.cpp-Version, Server-Flags)
    if system.is_dir():
        sources.append((system, "_system"))

    entries: list[tuple[Path, str]] = []  # (absolute file, path inside the top folder)
    for src, prefix in sources:
        for f in _files(src):
            entries.append((f, f"{prefix}/{f.relative_to(src).as_posix()}"))

    run_dirs = [p for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    complete = sum((p / "result.json").exists() for p in run_dirs)

    manifest_lines: list[str] = []
    total_bytes = 0
    for f, rel in entries:
        manifest_lines.append(f"{_sha256_file(f)}  {rel}")
        total_bytes += f.stat().st_size
    manifest = ("\n".join(manifest_lines) + "\n").encode("utf-8")

    info: dict[str, Any] = {
        "experiment": exp_dir.name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "arena_version": __version__,
        "git_commit": _git_commit(),
        "runs_total": len(run_dirs),
        "runs_complete": complete,
        "runs_without_result": sorted(p.name for p in run_dirs if not (p / "result.json").exists()),
        "includes_server_logs": any(p == "_server_logs" for _, p in sources),
        "files": len(entries),
        "bytes": total_bytes,
    }
    info_bytes = json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8")

    tmp = archive.with_suffix(archive.suffix + ".part")
    with tarfile.open(tmp, "w:gz") as tar:
        for name, data in ((MANIFEST, manifest), (INFO, info_bytes)):
            ti = tarfile.TarInfo(f"{top}/{name}")
            ti.size = len(data)
            ti.mtime = int(datetime.now(timezone.utc).timestamp())
            tar.addfile(ti, io.BytesIO(data))
        for f, rel in entries:
            tar.add(f, arcname=f"{top}/{rel}", recursive=False)
    tmp.replace(archive)

    digest = _sha256_file(archive)
    (out_dir / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")

    # Sofort gegenprüfen: Archiv lesbar und jede Datei korrekt
    report = verify(archive)
    if not report["ok"]:
        raise SystemExit(f"Export fehlerhaft, Archiv nicht verwenden: {json.dumps(report, ensure_ascii=False)}")
    return archive


def _parse_manifest(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            out[rel] = digest
    return out


def verify(target: Path) -> dict[str, Any]:
    """Prüft ein Export-Archiv (.tar.gz) oder einen entpackten Export-Ordner."""
    target = Path(target)
    report: dict[str, Any] = {"target": str(target), "ok": False}

    if target.is_dir():
        mpath = target / MANIFEST
        if not mpath.exists():
            report["error"] = f"{MANIFEST} fehlt in {target}"
            return report
        expected = _parse_manifest(mpath.read_text(encoding="utf-8"))
        actual = {f.relative_to(target).as_posix(): f for f in _files(target)}
        mismatched = [r for r, d in expected.items() if r in actual and _sha256_file(actual[r]) != d]
        present = set(actual) - {MANIFEST, INFO}
    elif target.is_file():
        sidecar = target.with_name(target.name + ".sha256")
        if sidecar.exists():
            want = sidecar.read_text(encoding="utf-8").split()[0]
            report["archive_sha256_ok"] = _sha256_file(target) == want
        else:
            report["archive_sha256_ok"] = None  # keine .sha256-Datei daneben
        with tarfile.open(target, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            tops = {m.name.split("/", 1)[0] for m in members}
            if len(tops) != 1:
                report["error"] = f"unerwarteter Aufbau (Top-Ordner: {sorted(tops)})"
                return report
            top = tops.pop()
            mf = tar.extractfile(f"{top}/{MANIFEST}")
            if mf is None:
                report["error"] = f"{MANIFEST} fehlt im Archiv"
                return report
            expected = _parse_manifest(mf.read().decode("utf-8"))
            present = set()
            mismatched = []
            for m in members:
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                if rel in (MANIFEST, INFO):
                    continue
                present.add(rel)
                fh = tar.extractfile(m)
                if rel in expected and (fh is None or _sha256_stream(fh) != expected[rel]):
                    mismatched.append(rel)
    else:
        report["error"] = f"nicht gefunden: {target}"
        return report

    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    report.update({
        "files_expected": len(expected),
        "files_found": len(present),
        "missing": missing,
        "extra": extra,
        "mismatched": sorted(mismatched),
    })
    report["ok"] = (not missing and not extra and not mismatched
                    and report.get("archive_sha256_ok") is not False)
    return report
