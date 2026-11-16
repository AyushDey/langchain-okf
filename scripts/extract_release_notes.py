#!/usr/bin/env python3
"""Extract release notes for a target version/tag from CHANGELOG.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def extract_notes(
    tag: str,
    changelog_path: Path | None = None,
    output_path: Path | None = None,
) -> str:
    """Extract release notes from CHANGELOG.md for a given tag.

    Args:
        tag: Git tag (e.g., 'v0.2.0') or bare version string (e.g., '0.2.0').
        changelog_path: Optional custom path to CHANGELOG.md. Defaults to ROOT / "CHANGELOG.md".
        output_path: Optional path to write the extracted release notes markdown.

    Returns:
        The extracted release notes including the version header.

    Raises:
        ValueError: If the release notes for the target tag are not found or are empty.
    """
    version = tag.lstrip("v")
    if changelog_path is not None:
        source_file = changelog_path
    elif Path("CHANGELOG.md").is_file():
        source_file = Path("CHANGELOG.md")
    else:
        source_file = ROOT / "CHANGELOG.md"

    if not source_file.is_file():
        raise ValueError(f"Changelog file not found at: {source_file}")

    content = source_file.read_text(encoding="utf-8")
    pattern = rf"^##\s+v?{re.escape(version)}\b[^\n]*\n(.*?)(?=^##\s|\Z)"
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

    if not match or not match.group(1).strip():
        raise ValueError(
            f"Release notes for {tag} not found or empty in {source_file.name}"
        )

    release_notes = match.group(0).strip() + "\n"

    if output_path is not None:
        output_path.write_text(release_notes, encoding="utf-8")
        print(
            f"Extracted {len(release_notes)} bytes of release notes for {tag} -> {output_path}"
        )

    return release_notes


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for extracting release notes."""
    args = sys.argv[1:] if argv is None else argv

    if not args:
        print("Usage: extract_release_notes.py <tag> [output_file]", file=sys.stderr)
        sys.exit(1)

    tag = args[0]
    out_file = Path(args[1]) if len(args) > 1 else ROOT / "RELEASE_NOTES.md"

    try:
        extract_notes(tag, output_path=out_file)
    except ValueError as err:
        print(f"❌ Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
