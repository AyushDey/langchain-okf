"""OKF concept document parser — OKF v0.2.

Aligns with OKF v0.2 specification:
https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md

Key v0.2 additions over v0.1:
- ``generated`` / ``verified`` trust families (§5.2)
- ``sources`` provenance list (§5.1)
- ``status`` and ``stale_after`` lifecycle fields (§5.4, §5.5)
- Actor convention for ``generated.by`` / ``verified[].by`` (§7)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_DELIM = "---"
_RESERVED_FILENAMES = {"index.md", "log.md"}
_CROSS_LINK_RE = re.compile(
    r"\[([^\]]*)\]\((/[^)]+\.md|\.\./[^)]+\.md|(?!\w+://)[^)]+\.md)\)"
)


def _to_iso8601_str(val: Any) -> str:
    """Normalize datetime/date or string values to ISO 8601 UTC string."""
    if val is None or val == "":
        return ""
    if isinstance(val, (datetime, date)):
        if isinstance(val, datetime) and val.tzinfo is not None:
            return val.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return val.strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(val)


class OKFParseError(ValueError):
    """Raised when an OKF document cannot be parsed."""


@dataclass
class OKFConcept:
    """A parsed OKF v0.2 concept document.

    Attributes:
        concept_id: Slash-separated path identity (e.g. ``tables/orders``).
        frontmatter: Parsed YAML frontmatter dict.
        body: Markdown body after frontmatter.
        source_path: Absolute path to the source ``.md`` file.
    """

    concept_id: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    source_path: str = ""

    # ── Core fields (§4.1) ────────────────────────────────────────────────────

    @property
    def type(self) -> str:
        return str(self.frontmatter.get("type", ""))

    @property
    def title(self) -> str:
        return str(self.frontmatter.get("title", ""))

    @property
    def description(self) -> str:
        return str(self.frontmatter.get("description", ""))

    @property
    def resource(self) -> str:
        return str(self.frontmatter.get("resource", ""))

    @property
    def tags(self) -> list[str]:
        t = self.frontmatter.get("tags", [])
        return t if isinstance(t, list) else [t]

    # ── v0.1 compat ──────────────────────────────────────────────────────────

    @property
    def timestamp(self) -> str:
        """Legacy v0.1 field. Prefer ``generated_at``."""
        return str(self.frontmatter.get("timestamp", ""))

    # ── Trust: generated (§5.2) ───────────────────────────────────────────────

    @property
    def generated(self) -> dict[str, str]:
        """``generated: { by, at }`` block, or empty dict."""
        g = self.frontmatter.get("generated", {})
        return g if isinstance(g, dict) else {}

    @property
    def generated_by(self) -> str:
        return self.generated.get("by", "")

    @property
    def generated_at(self) -> str:
        """ISO 8601 datetime string of last meaningful content change."""
        return _to_iso8601_str(self.generated.get("at", self.timestamp))

    # ── Trust: verified (§5.2) ────────────────────────────────────────────────

    @property
    def verified(self) -> list[dict[str, str]]:
        """Verification events as a list of ``{ by, at }`` dicts.

        Per spec §5.2, a bare mapping is treated as a one-element list.
        """
        v = self.frontmatter.get("verified")
        if v is None:
            return []
        raw = [v] if isinstance(v, dict) else (v if isinstance(v, list) else [])
        out = []
        for e in raw:
            if isinstance(e, dict):
                item = dict(e)
                if "at" in item:
                    item["at"] = _to_iso8601_str(item["at"])
                out.append(item)
        return out

    @property
    def trust_tier(self) -> str:
        """Derived trust tier per §5.3.

        Returns:
            ``"human-reviewed"`` if any verifier uses ``human:`` prefix.
            ``"machine-confirmed"`` if verified by non-human actors only.
            ``"unverified"`` if ``verified`` is absent or empty.
        """
        events = self.verified
        if not events:
            return "unverified"
        if any(e.get("by", "").startswith("human:") for e in events):
            return "human-reviewed"
        return "machine-confirmed"

    # ── Lifecycle: status (§5.4) ──────────────────────────────────────────────

    @property
    def status(self) -> str:
        """``draft`` | ``stable`` | ``deprecated``. Defaults to ``stable``."""
        return str(self.frontmatter.get("status", "stable"))

    # ── Lifecycle: stale_after (§5.5) ─────────────────────────────────────────

    @property
    def stale_after(self) -> str:
        return _to_iso8601_str(self.frontmatter.get("stale_after", ""))

    @property
    def is_stale(self) -> bool:
        """True when ``now >= stale_after`` (§5.5)."""
        sa = self.stale_after
        if not sa:
            return False
        try:
            dt = datetime.fromisoformat(sa.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) >= dt
        except (ValueError, TypeError):
            return False

    # ── Provenance: sources (§5.1) ────────────────────────────────────────────

    @property
    def sources(self) -> list[dict[str, Any]]:
        """List of source provenance entries per §5.1."""
        s = self.frontmatter.get("sources", [])
        return s if isinstance(s, list) else []

    # ── Attested Computation fields (§10) ─────────────────────────────────────

    @property
    def runtime(self) -> str:
        """Computation runtime (e.g. ``bigquery``, ``python``). §10."""
        return str(self.frontmatter.get("runtime", ""))

    @property
    def parameters(self) -> list[dict[str, Any]]:
        """Declared parameters for an Attested Computation. §10."""
        p = self.frontmatter.get("parameters", [])
        return p if isinstance(p, list) else []

    @property
    def executor(self) -> dict[str, Any]:
        """Executor block ``{ resource, receipt }``. §10."""
        e = self.frontmatter.get("executor", {})
        return e if isinstance(e, dict) else {}

    @property
    def attester(self) -> dict[str, Any]:
        """Attester block ``{ resource }``. §10."""
        a = self.frontmatter.get("attester", {})
        return a if isinstance(a, dict) else {}

    # ── Graph links ───────────────────────────────────────────────────────────

    def cross_links(self) -> list[str]:
        """Extract concept IDs referenced via markdown links in the body."""
        links: list[str] = []
        for _, href in _CROSS_LINK_RE.findall(self.body):
            if href.endswith(".md"):
                href = href[:-3]
            href = href.lstrip("./")
            if href.startswith("/"):
                href = href[1:]
            if href:
                links.append(href)
        return links


# ── Parsing helpers ───────────────────────────────────────────────────────────


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split an OKF markdown file into frontmatter dict and body string."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return {}, text

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_DELIM:
            end_idx = i
            break
    if end_idx is None:
        raise OKFParseError("Unterminated YAML frontmatter block")

    fm_text = "\n".join(lines[1:end_idx])
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        raise OKFParseError(f"Invalid YAML in frontmatter: {e}") from e
    if not isinstance(fm, dict):
        raise OKFParseError("Frontmatter must be a YAML mapping")

    body = "\n".join(lines[end_idx + 1 :])
    if body.startswith("\n"):
        body = body[1:]
    return fm, body


def is_reserved(path: Path) -> bool:
    """Check if a filename is an OKF reserved name (index.md, log.md)."""
    return path.name in _RESERVED_FILENAMES


def path_to_concept_id(bundle_root: Path, path: Path) -> str:
    """Convert a file path to a slash-separated concept ID."""
    rel = path.relative_to(bundle_root).with_suffix("")
    return "/".join(rel.parts)


def parse_bundle(bundle_root: Path) -> list[OKFConcept]:
    """Walk an OKF bundle directory and parse all concept documents.

    Skips reserved files (``index.md``, ``log.md``) per the spec.
    """
    root = Path(bundle_root)
    concepts: list[OKFConcept] = []

    for md_file in sorted(root.rglob("*.md")):
        if is_reserved(md_file):
            continue
        text = md_file.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        concept_id = path_to_concept_id(root, md_file)
        concepts.append(
            OKFConcept(
                concept_id=concept_id,
                frontmatter=fm,
                body=body,
                source_path=str(md_file),
            )
        )
    return concepts
