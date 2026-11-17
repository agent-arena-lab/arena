"""Shared workspace for the README channel.

* All paths are confined to the run's ``shared/`` directory (no traversal, no
  symlinks, no subdirectories outside it).
* Every modification is recorded as a :class:`FileChange` incl. unified diff and a
  per-line "blame" so that overwriting/deleting the other agent's content can be
  measured objectively.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath


class SandboxError(Exception):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class FileChange:
    path: str
    author: str
    op: str  # write | append | create
    before_sha: str
    after_sha: str
    diff: str
    lines_added: int
    lines_removed: int
    removed_by_author: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Workspace:
    def __init__(self, root: Path, max_bytes: int = 32_000, allowed_files: set[str] | None = None):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.allowed_files = allowed_files  # None = any flat file name
        self.blame: dict[str, list[str]] = {}  # path -> author per line
        self.version: dict[str, int] = {}
        self.last_author: dict[str, str] = {}

    # -- path handling ------------------------------------------------------
    def _rel(self, path: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise SandboxError("path must be a non-empty string")
        raw = path.strip()
        if raw.startswith("./"):
            raw = raw[2:]
        p = PurePosixPath(raw)
        if "\\" in raw or p.is_absolute() or ".." in p.parts or len(p.parts) != 1:
            raise SandboxError("only plain file names inside the shared workspace are allowed")
        name = p.parts[0]
        if self.allowed_files is not None and name not in self.allowed_files:
            raise SandboxError(f"unknown file '{name}'. Available: {sorted(self.allowed_files)}")
        return name

    def _abs(self, name: str) -> Path:
        full = (self.root / name)
        if full.is_symlink():
            raise SandboxError("symlinks are not allowed")
        resolved = full.resolve()
        if resolved.parent != self.root:
            raise SandboxError("path escapes workspace")
        return resolved

    # -- operations ---------------------------------------------------------
    def init_file(self, name: str, content: str, author: str = "system") -> None:
        name = self._rel(name)
        self._abs(name).write_text(content, encoding="utf-8")
        self.blame[name] = [author] * len(content.splitlines())
        self.version[name] = 0
        self.last_author[name] = author

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_file() and not p.is_symlink())

    def read(self, path: str) -> str:
        name = self._rel(path)
        full = self._abs(name)
        if not full.exists():
            raise SandboxError(f"file '{name}' does not exist")
        return full.read_text(encoding="utf-8")

    def write(self, path: str, content: str, author: str, append: bool = False) -> FileChange:
        name = self._rel(path)
        if not isinstance(content, str):
            content = str(content)
        full = self._abs(name)
        before = full.read_text(encoding="utf-8") if full.exists() else ""
        if append:
            sep = "" if (not before or before.endswith("\n")) else "\n"
            after = before + sep + content
            op = "append"
        else:
            after = content
            op = "write" if full.exists() else "create"
        if len(after.encode("utf-8")) > self.max_bytes:
            raise SandboxError(f"file would exceed {self.max_bytes} bytes")
        full.write_text(after, encoding="utf-8")

        old_lines, new_lines = before.splitlines(), after.splitlines()
        old_blame = self.blame.get(name, ["unknown"] * len(old_lines))
        if len(old_blame) != len(old_lines):  # defensive: external modification
            old_blame = ["unknown"] * len(old_lines)
        new_blame: list[str] = []
        removed_by: dict[str, int] = {}
        added = removed = 0
        sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                new_blame.extend(old_blame[i1:i2])
                continue
            for k in range(i1, i2):
                removed += 1
                removed_by[old_blame[k]] = removed_by.get(old_blame[k], 0) + 1
            added += j2 - j1
            new_blame.extend([author] * (j2 - j1))
        self.blame[name] = new_blame
        self.version[name] = self.version.get(name, 0) + 1
        self.last_author[name] = author

        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
        return FileChange(name, author, op, _sha(before), _sha(after), diff, added, removed, removed_by)
