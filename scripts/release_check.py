#!/usr/bin/env python3
"""Pre-flight check script before tagging a release of langchain-okf."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(name: str, cmd: list[str]) -> None:
    print(f"==> Checking: {name}...")
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        print(f"❌ FAILED: {name}")
        sys.exit(res.returncode)
    print(f"✅ PASSED: {name}\n")


def main() -> None:
    print("=" * 60)
    print("langchain-okf: Pre-Flight Release Check")
    print("=" * 60)

    # 1. Extract version from pyproject.toml
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', pyproject)
    if not match:
        print("❌ Error: Could not determine version in pyproject.toml")
        sys.exit(1)
    version = match.group(1)
    print(f"Target Version: v{version}\n")

    # 2. Check CHANGELOG.md contains the version
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## {version}" not in changelog:
        print(f"❌ Error: CHANGELOG.md does not contain section for '## {version}'")
        sys.exit(1)

    # 3. Code quality checks
    check("Ruff Linter", ["uv", "run", "ruff", "check", "langchain_okf/", "tests/"])
    check(
        "Ruff Formatter",
        ["uv", "run", "ruff", "format", "--check", "langchain_okf/", "tests/"],
    )
    check("MyPy Type Checker", ["uv", "run", "mypy", "langchain_okf/"])
    check("Unit Tests", ["uv", "run", "pytest", "tests/"])

    # 4. Build distribution
    check("Clean Distribution Build", ["uv", "build"])

    print(f"🎉 All pre-flight checks passed! Safe to tag and publish v{version}.")


if __name__ == "__main__":
    main()
