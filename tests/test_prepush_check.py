"""Tests for scripts/prepush_check.py in throwaway git repos (fake keys only)."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepush_check  # noqa: E402

FAKE = "gsk_" + "A1b2C3d4" * 4   # built at runtime so this file itself holds no key-shaped string


def repo(tmp_path):
    def g(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text(".env\n.venv/\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    g("add", "-A")
    g("commit", "-q", "-m", "init")
    return g


def test_clean_repo_passes(tmp_path):
    repo(tmp_path)
    (tmp_path / ".env").write_text(f"GROQ_API_KEY={FAKE}\n")    # ignored, so fine
    assert prepush_check.check(tmp_path) == []


def test_key_in_untracked_file_fails(tmp_path):
    repo(tmp_path)
    (tmp_path / "notes.md").write_text(f"key {FAKE}\n")
    assert any("contains a Groq API key" in p for p in prepush_check.check(tmp_path))


def test_env_value_without_prefix_is_caught(tmp_path):
    repo(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=plainsecretvalue123\n")
    (tmp_path / "b.txt").write_text("plainsecretvalue123")
    assert any("b.txt" in p for p in prepush_check.check(tmp_path))


def test_key_in_history_fails_even_after_removal(tmp_path):
    g = repo(tmp_path)
    (tmp_path / "c.py").write_text(f"K = '{FAKE}'\n")
    g("add", "c.py")
    g("commit", "-q", "-m", "oops")
    (tmp_path / "c.py").write_text("K = None\n")
    g("commit", "-qam", "fix")
    assert any("history" in p for p in prepush_check.check(tmp_path))


def test_env_venv_and_large_files_fail(tmp_path):
    g = repo(tmp_path)
    (tmp_path / ".gitignore").write_text("")
    (tmp_path / ".env").write_text("GROQ_API_KEY=\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x.py").write_text("")
    (tmp_path / "big.bin").write_bytes(b"0" * (prepush_check.MAX_BYTES + 1))
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "raw_runs.jsonl").write_bytes(b"0" * (prepush_check.MAX_BYTES + 1))
    problems = "\n".join(prepush_check.check(tmp_path))
    assert ".env: env file" in problems and ".venv/x.py" in problems and "big.bin" in problems
    assert "raw_runs.jsonl" not in problems   # allowed up to 50 MB
