from __future__ import annotations

from pathlib import Path

import pytest

# Target module to test (will be implemented in Step 3)
from scripts.extract_release_notes import extract_notes, main

SAMPLE_CHANGELOG = """# Changelog

All notable changes to this project will be documented in this file.

## v0.2.1 (2026-11-15)

### Fixes
- Guard commit parser tests with pytest importorskip.
- Fix commit parser type matching and configure deterministic patch bump.

## 0.2.0 (2026-11-14)

### Features
- Initial release of langchain-okf.
- Added OKFRetriever with over-fetching.

## v0.1.0 (2026-11-10)

### Chores
- Initial repository setup.
"""

EMPTY_SECTION_CHANGELOG = """# Changelog

## v0.3.0

## v0.2.0
- Previous notes.
"""


def test_extract_notes_with_v_prefix(tmp_path: Path) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(SAMPLE_CHANGELOG, encoding="utf-8")

    notes = extract_notes("v0.2.1", changelog_path=changelog_file)
    assert notes.startswith("## v0.2.1 (2026-11-15)")
    assert "Guard commit parser tests" in notes
    # Must not leak into the next section
    assert "0.2.0" not in notes


def test_extract_notes_query_without_v_on_prefixed_header(tmp_path: Path) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(SAMPLE_CHANGELOG, encoding="utf-8")

    # Tag argument is "0.2.1" without "v"
    notes = extract_notes("0.2.1", changelog_path=changelog_file)
    assert notes.startswith("## v0.2.1 (2026-11-15)")
    assert "Guard commit parser tests" in notes


def test_extract_notes_query_with_v_on_unprefixed_header(tmp_path: Path) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(SAMPLE_CHANGELOG, encoding="utf-8")

    # Header in changelog is "## 0.2.0", query is "v0.2.0"
    notes = extract_notes("v0.2.0", changelog_path=changelog_file)
    assert notes.startswith("## 0.2.0 (2026-11-14)")
    assert "Initial release of langchain-okf" in notes
    assert "0.1.0" not in notes


def test_extract_notes_terminal_section(tmp_path: Path) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(SAMPLE_CHANGELOG, encoding="utf-8")

    # The last section in the sample
    notes = extract_notes("v0.1.0", changelog_path=changelog_file)
    assert notes.startswith("## v0.1.0 (2026-11-10)")
    assert "Initial repository setup" in notes


def test_extract_notes_writes_output_file(tmp_path: Path) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(SAMPLE_CHANGELOG, encoding="utf-8")
    out_file = tmp_path / "RELEASE_NOTES.md"

    notes = extract_notes("v0.2.1", changelog_path=changelog_file, output_path=out_file)
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == notes


def test_extract_notes_missing_tag_raises_value_error(tmp_path: Path) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(SAMPLE_CHANGELOG, encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        extract_notes("v9.9.9", changelog_path=changelog_file)


def test_extract_notes_empty_section_raises_value_error(tmp_path: Path) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(EMPTY_SECTION_CHANGELOG, encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        extract_notes("v0.3.0", changelog_path=changelog_file)


def test_main_cli_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(SAMPLE_CHANGELOG, encoding="utf-8")
    out_file = tmp_path / "OUT.md"

    # Run CLI with custom output path and pointing to changelog_file
    monkeypatch.chdir(tmp_path)
    main(["v0.2.1", str(out_file)])

    assert out_file.exists()
    assert "Guard commit parser tests" in out_file.read_text(encoding="utf-8")


def test_main_cli_missing_tag_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changelog_file = tmp_path / "CHANGELOG.md"
    changelog_file.write_text(SAMPLE_CHANGELOG, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        main(["v9.9.9"])
    assert exc_info.value.code == 1


def test_main_cli_no_args_exits_1() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 1
