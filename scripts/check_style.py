"""Fail if any tracked text file contains an em dash or AI attribution.

Usage: python scripts/check_style.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EM_DASH = chr(0x2014)
TEXT = {".md", ".py", ".yaml", ".yml", ".cff", ".toml", ".txt", ".json"}
ATTRIBUTION = re.compile(r"co-authored-by|claude-session|generated with \[?claude", re.IGNORECASE)
ALLOWED_ATTRIBUTION = {"CLAUDE.md", "scripts/check_style.py"}


def main() -> int:
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
    problems = []
    for rel in files:
        p = ROOT / rel
        if p.suffix not in TEXT or not p.is_file():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if EM_DASH in line:
                problems.append(f"{rel}:{i}: em dash")
            if ATTRIBUTION.search(line) and rel not in ALLOWED_ATTRIBUTION and not rel.startswith(".claude/"):
                problems.append(f"{rel}:{i}: AI attribution")
    log = subprocess.check_output(["git", "log", "--format=%B"], cwd=ROOT, text=True)
    if ATTRIBUTION.search(log):
        problems.append("git history: AI attribution in a commit message")
    print("\n".join(problems) if problems else "style check passed: no em dashes, no AI attribution")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
