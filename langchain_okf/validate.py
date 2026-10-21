"""OKF bundle validation utility — OKF v0.2.

Checks whether a directory conforms to OKF v0.2 spec (§11).

Example:
    .. code-block:: python

        from langchain_okf import validate_bundle

        issues = validate_bundle("./my_knowledge/")
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("Bundle is valid OKF v0.2")
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from langchain_okf._parser import is_reserved, parse_frontmatter

# ISO 8601 datetime pattern (with explicit UTC offset)
_ISO8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)

# Actor convention: <producer>/<version>, human:<id>, process:<id>
_ACTOR_RE = re.compile(r"^([a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-./]+|human:.+|process:.+)$")


def _check_iso8601(value: object, field_name: str) -> str | None:
    """Return an issue string if value is not a valid ISO 8601 datetime."""
    if isinstance(value, (datetime, date)):
        return None
    if not isinstance(value, str):
        return f"  {field_name}: expected ISO 8601 string, got {type(value).__name__}"
    if not _ISO8601_RE.match(value.strip()):
        return (
            f"  {field_name}: not a valid ISO 8601 datetime with UTC offset: {value!r}"
        )
    return None


def _check_actor(value: object, field_name: str) -> str | None:
    """Return an issue string if value doesn't follow the actor convention (§7)."""
    if not isinstance(value, str) or not value.strip():
        return f"  {field_name}: must be a non-empty actor string"
    if not _ACTOR_RE.match(value.strip()):
        return (
            f"  {field_name}: actor {value!r} does not follow convention "
            f"(<producer>/<version>, human:<id>, or process:<id>)"
        )
    return None


def validate_bundle(
    bundle_path: str | Path,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate an OKF bundle against the v0.2 spec.

    Mandatory checks (§11):
    - Every non-reserved ``.md`` file has parseable YAML frontmatter.
    - Every frontmatter block contains a non-empty ``type`` field.
    - Reserved files (``index.md``, ``log.md``) are not concept documents.

    Optional trust/provenance checks (soft guidance, reported as warnings
    unless ``strict=True``):
    - ``generated.by`` follows actor convention (§7).
    - ``generated.at`` is ISO 8601 (§5.2).
    - ``verified`` entries have ``by`` (actor) and ``at`` (ISO 8601) (§5.2).
    - ``stale_after`` is ISO 8601 (§5.5).
    - ``status`` is one of ``draft``, ``stable``, ``deprecated`` (§5.4).
    - Each ``sources`` entry has a ``resource`` field (§5.1).
    - ``Attested Computation`` concepts have ``runtime`` field (§10).

    Args:
        bundle_path: Path to the OKF bundle root directory.
        strict: If True, soft-guidance violations are also returned as issues.

    Returns:
        List of issue strings. Empty list means the bundle is conformant.
    """
    root = Path(bundle_path)
    issues: list[str] = []

    if not root.exists():
        issues.append(f"Bundle directory does not exist: {root}")
        return issues

    if not root.is_dir():
        issues.append(f"Not a directory: {root}")
        return issues

    md_files = sorted(root.rglob("*.md"))
    if not md_files:
        issues.append("No .md files found in bundle")
        return issues

    concept_count = 0
    for md_file in md_files:
        rel = md_file.relative_to(root)

        if is_reserved(md_file):
            continue

        concept_count += 1
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            issues.append(f"{rel}: cannot read file: {e}")
            continue

        try:
            fm, _ = parse_frontmatter(text)
        except Exception as e:
            issues.append(f"{rel}: invalid frontmatter: {e}")
            continue

        if not fm:
            issues.append(f"{rel}: missing YAML frontmatter block")
            continue

        # ── §11 mandatory: type ───────────────────────────────────────────────
        if not fm.get("type"):
            issues.append(f"{rel}: missing required 'type' field in frontmatter")

        if not strict:
            continue

        # ── §5.2 generated ────────────────────────────────────────────────────
        generated = fm.get("generated")
        if generated is not None:
            if not isinstance(generated, dict):
                issues.append(f"{rel}: 'generated' must be a mapping")
            else:
                by_issue = _check_actor(generated.get("by"), f"{rel}: generated.by")
                if by_issue:
                    issues.append(by_issue)
                at = generated.get("at")
                if at is not None:
                    at_issue = _check_iso8601(at, f"{rel}: generated.at")
                    if at_issue:
                        issues.append(at_issue)

        # ── §5.2 verified ─────────────────────────────────────────────────────
        verified = fm.get("verified")
        if verified is not None:
            events = [verified] if isinstance(verified, dict) else verified
            if not isinstance(events, list):
                issues.append(f"{rel}: 'verified' must be a mapping or list")
            else:
                for i, evt in enumerate(events):
                    if not isinstance(evt, dict):
                        issues.append(f"{rel}: verified[{i}] must be a mapping")
                        continue
                    by_issue = _check_actor(evt.get("by"), f"{rel}: verified[{i}].by")
                    if by_issue:
                        issues.append(by_issue)
                    at_issue = _check_iso8601(evt.get("at"), f"{rel}: verified[{i}].at")
                    if at_issue:
                        issues.append(at_issue)

        # ── §5.4 status ───────────────────────────────────────────────────────
        status = fm.get("status")
        if status is not None and status not in ("draft", "stable", "deprecated"):
            issues.append(
                f"{rel}: 'status' must be draft | stable | deprecated, got {status!r}"
            )

        # ── §5.5 stale_after ──────────────────────────────────────────────────
        stale_after = fm.get("stale_after")
        if stale_after is not None:
            sa_issue = _check_iso8601(stale_after, f"{rel}: stale_after")
            if sa_issue:
                issues.append(sa_issue)

        # ── §5.1 sources ──────────────────────────────────────────────────────
        sources = fm.get("sources", [])
        if isinstance(sources, list):
            for i, src in enumerate(sources):
                if not isinstance(src, dict):
                    issues.append(f"{rel}: sources[{i}] must be a mapping")
                    continue
                if not src.get("resource"):
                    issues.append(f"{rel}: sources[{i}].resource is required (§5.1)")

        # ── §10 Attested Computation ──────────────────────────────────────────
        if fm.get("type") == "Attested Computation" and not fm.get("runtime"):
            issues.append(f"{rel}: Attested Computation requires 'runtime' field (§10)")

    if concept_count == 0:
        issues.append("No concept documents found (only reserved files)")

    return issues
