"""Safety check before pushing to GitHub. Exit code 0 = safe, 1 = problems found.

  python scripts/prepush_check.py

Checks every file that is tracked, staged, or untracked-but-not-ignored (what `git add -A` would
commit):
  - no Groq key (gsk_...) or the actual GROQ_API_KEY value from .env in any file
  - no .env (only .env.example), nothing under .venv/, data/cache/ or results/dev/
  - no file over 5 MB, except results/raw_runs.jsonl (allowed up to 50 MB)
And the whole git history (all refs):
  - no Groq key or the .env key value in any commit
  - .env was never committed
The key itself is never printed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_RE = re.compile(rb"gsk_[A-Za-z0-9]{20,}")
MAX_BYTES = 5 * 1024 * 1024
RAW_RUNS = "results/raw_runs.jsonl"
RAW_RUNS_MAX = 50 * 1024 * 1024
FORBIDDEN_PREFIXES = (".venv/", "data/cache/", "results/dev/")


def git(*args: str, root: Path = ROOT) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def env_key(root: Path) -> bytes | None:
    env = root / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("GROQ_API_KEY="):
            v = line.split("=", 1)[1].strip().strip("'\"")
            return v.encode() if len(v) >= 10 else None
    return None


def candidate_files(root: Path) -> list[str]:
    tracked = git("ls-files", "-z", root=root).split(b"\0")
    staged = git("diff", "--cached", "--name-only", "-z", root=root).split(b"\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z", root=root).split(b"\0")
    return sorted({p.decode() for p in tracked + staged + untracked if p})


def check(root: Path = ROOT) -> list[str]:
    problems = []
    key = env_key(root)

    def has_secret(data: bytes) -> bool:
        return bool(KEY_RE.search(data)) or bool(key and key in data)

    files = candidate_files(root)
    for rel in files:
        name = rel.rsplit("/", 1)[-1]
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            problems.append(f"{rel}: env file would be committed")
        if rel.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"{rel}: under a forbidden directory ({rel.split('/')[0]}...)")
        path = root / rel
        if not path.is_file():
            continue   # staged deletion
        size = path.stat().st_size
        limit = RAW_RUNS_MAX if rel == RAW_RUNS else MAX_BYTES
        if size > limit:
            problems.append(f"{rel}: {size / 1e6:.1f} MB (limit {limit / 1e6:.0f} MB)")
        if has_secret(path.read_bytes()):
            problems.append(f"{rel}: contains a Groq API key")
    # the staged version can differ from the working copy
    if has_secret(git("diff", "--cached", root=root)):
        problems.append("staged diff contains a Groq API key")

    history = git("log", "--all", "-p", "--no-color", "--no-ext-diff", root=root)
    if has_secret(history):
        commits = [c.decode() for c in git("log", "--all", "--format=%h", root=root).split()
                   if has_secret(git("show", "--no-color", c.decode(), root=root))]
        problems.append(f"git history contains a Groq API key (commits: {', '.join(commits) or 'unknown'})")
    names = git("log", "--all", "--name-only", "--format=", root=root).decode().split()
    if any(n == ".env" or n.endswith("/.env") for n in names):
        problems.append(".env appears in git history")
    return problems


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    problems = check()
    n_files = len(candidate_files(ROOT))
    n_commits = len(git("rev-list", "--all").split())
    if problems:
        print("PRE-PUSH CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"Pre-push check passed: {n_files} files and {n_commits} commits scanned; "
          "no Groq key, no .env/.venv/cache files, no oversized files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
