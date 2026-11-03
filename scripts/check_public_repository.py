#!/usr/bin/env python3
"""Fail closed when public-release safety invariants are violated."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_NAMES = {
    ".env",
    "paths.local.toml",
    "golden_references.json",
}
FORBIDDEN_NAME_FRAGMENTS = (
    "raw_response",
    "raw_request",
    "request_registry",
    "batch_result",
    "pool_metadata",
)
FORBIDDEN_CONTENT = {
    "private_absolute_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "restricted_archive_name": re.compile("\u6240\u6709\u7ed3\u679c"),
    "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"gh[oprsu]_[A-Za-z0-9]{20,}"),
}
MAX_FILE_BYTES = 1_000_000


def candidate_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
    )


def main() -> int:
    errors: list[str] = []
    for path in candidate_files():
        relative = path.relative_to(ROOT)
        lower_name = path.name.lower()
        if path.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden filename: {relative}")
        if any(fragment in lower_name for fragment in FORBIDDEN_NAME_FRAGMENTS):
            errors.append(f"restricted filename pattern: {relative}")
        if path.is_symlink():
            errors.append(f"symlink is not allowed: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"expected UTF-8 text: {relative}")
                continue
            for label, pattern in FORBIDDEN_CONTENT.items():
                if pattern.search(text):
                    errors.append(f"{label}: {relative}")
    if errors:
        print("PUBLIC REPOSITORY SAFETY: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"PUBLIC REPOSITORY SAFETY: PASS ({len(candidate_files())} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
