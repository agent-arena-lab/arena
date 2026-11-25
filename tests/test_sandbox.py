import pytest

from arena.sandbox import SandboxError, Workspace


@pytest.fixture
def ws(tmp_path):
    w = Workspace(tmp_path / "shared", allowed_files={"README.md"})
    w.init_file("README.md", "# Shared README\n")
    return w


@pytest.mark.parametrize("bad", ["../x", "/etc/passwd", "sub/README.md", "..\\x", "", "other.md"])
def test_rejects_paths(ws, bad):
    with pytest.raises(SandboxError):
        ws.read(bad)


def test_symlink_rejected(ws, tmp_path):
    (tmp_path / "secret").write_text("s")
    (ws.root / "README.md").unlink()
    (ws.root / "README.md").symlink_to(tmp_path / "secret")
    with pytest.raises(SandboxError):
        ws.read("README.md")


def test_blame_tracks_removed_lines_of_other(ws):
    ws.write("README.md", "apple: 0.50\n", "Agent B", append=True)
    ch = ws.write("README.md", "# Shared README\npear: 0.80\n", "Agent A")
    assert ch.removed_by_author == {"Agent B": 1}
    assert "-apple: 0.50" in ch.diff


def test_size_limit(tmp_path):
    w = Workspace(tmp_path / "s", max_bytes=10)
    with pytest.raises(SandboxError):
        w.write("README.md", "x" * 20, "Agent A")


def test_delete_counts_lines_of_other(tmp_path):
    w = Workspace(tmp_path / "s")
    w.write("notes.md", "pear: 0.80\nmore\n", "Agent A", round_no=1)
    ch = w.delete("notes.md", "Agent B", round_no=2)
    assert ch.op == "delete" and ch.removed_by_author == {"Agent A": 2} and not ch.removed_unseen_by_author
    assert w.list_files() == []
    with pytest.raises(SandboxError):
        w.delete("notes.md", "Agent B")


def test_max_files(tmp_path):
    w = Workspace(tmp_path / "s", max_files=1)
    w.write("a.md", "x", "Agent A")
    with pytest.raises(SandboxError):
        w.write("b.md", "y", "Agent A")
