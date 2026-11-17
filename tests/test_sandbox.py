import pytest

from coopsab.sandbox import SandboxError, Workspace


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
