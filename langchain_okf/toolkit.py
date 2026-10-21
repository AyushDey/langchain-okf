"""OKF agent tools for LangChain — OKF v0.2.

Provides a toolkit of LangChain ``BaseTool`` implementations that let
agents navigate, search, and query OKF knowledge bundles. These tools
give agents structured access to the knowledge graph rather than just
flat document retrieval.

v0.2 additions: trust tier, status, staleness, and sources are surfaced
in tool output so agents can reason about knowledge freshness and
credibility.

Example:
    .. code-block:: python

        from langchain_okf import OKFToolkit
        from langchain_openai import ChatOpenAI

        toolkit = OKFToolkit.from_bundle("./bundles/ga4/")
        tools = toolkit.get_tools()

        agent = create_agent(model="gpt-4o", tools=tools)
        agent.invoke({"messages": [("user", "What tables relate to orders?")]})
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from langchain_okf._parser import OKFConcept, parse_frontmatter
from langchain_okf.graph import OKFKnowledgeGraph


def _trust_badge(concept: OKFConcept) -> str:
    """Short trust label for tool output."""
    tier = concept.trust_tier
    if tier == "human-reviewed":
        return "[✓ human-reviewed]"
    if tier == "machine-confirmed":
        return "[~ machine-confirmed]"
    return "[? unverified]"


def _staleness_warning(concept: OKFConcept) -> str:
    if concept.is_stale:
        return " ⚠️ STALE"
    if concept.stale_after:
        try:
            dt = datetime.fromisoformat(concept.stale_after.replace("Z", "+00:00"))
            remaining = (dt - datetime.now(timezone.utc)).days
            if remaining <= 7:
                return f" ⚠️ expires in {remaining}d"
        except (ValueError, TypeError):
            pass
    return ""


class OKFListConceptsTool(BaseTool):
    """List concepts in an OKF bundle, optionally filtered by type or tag."""

    name: str = "okf_list_concepts"
    description: str = (
        "List concepts in the knowledge base. Optionally filter by "
        "type (e.g. 'type:BigQuery Table'), tag (e.g. 'tag:sales'), "
        "trust tier ('trust:human-reviewed'), status ('status:stable'), or 'all'. "
        "Concepts marked STALE are past their freshness date."
    )
    graph: OKFKnowledgeGraph

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str = "all") -> str:
        query = query.strip()
        if query.startswith("type:"):
            type_name = query[5:].strip()
            ids = self.graph.concepts_by_type(type_name)
            label = f"type '{type_name}'"
        elif query.startswith("tag:"):
            tag = query[4:].strip()
            ids = self.graph.concepts_by_tag(tag)
            label = f"tag '{tag}'"
        elif query.startswith("trust:"):
            tier = query[6:].strip()
            ids = [
                cid
                for cid in self.graph.concept_ids
                if (c := self.graph.get_concept(cid)) and c.trust_tier == tier
            ]
            label = f"trust tier '{tier}'"
        elif query.startswith("status:"):
            status = query[7:].strip()
            ids = [
                cid
                for cid in self.graph.concept_ids
                if (c := self.graph.get_concept(cid)) and c.status == status
            ]
            label = f"status '{status}'"
        else:
            ids = self.graph.concept_ids
            label = "all types"

        if not ids:
            return f"No concepts found for {label}."

        lines = [f"Found {len(ids)} concepts ({label}):"]
        for cid in ids:
            concept = self.graph.get_concept(cid)
            if concept:
                if (
                    concept.title
                    and concept.description
                    and concept.title != concept.description
                ):
                    desc = f"{concept.title} - {concept.description}"
                else:
                    desc = concept.description or concept.title or ""
                badge = _trust_badge(concept)
                stale = _staleness_warning(concept)
                source = _format_source_short(concept)
                source_suffix = f" {source}" if source else ""
                lines.append(
                    f"  - {cid} [{concept.type}] {badge}{stale}: {desc}{source_suffix}"
                )
            else:
                lines.append(f"  - {cid}")
        return "\n".join(lines)

    async def _arun(self, query: str = "all") -> str:
        return self._run(query)


class OKFReadConceptTool(BaseTool):
    """Read the full content of a specific OKF concept by its ID."""

    name: str = "okf_read_concept"
    description: str = (
        "Read the full content of a knowledge base concept. "
        "Input: the concept ID (e.g. 'tables/orders', 'metrics/wau'). "
        "Output includes trust tier, status, staleness, and provenance."
    )
    graph: OKFKnowledgeGraph

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, concept_id: str) -> str:
        concept_id = concept_id.strip().strip("/")
        concept = self.graph.get_concept(concept_id)
        if concept is None:
            available = self.graph.concept_ids
            suggestions = [c for c in available if concept_id.split("/")[-1] in c]
            msg = f"Concept '{concept_id}' not found."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions[:5])}?"
            return msg

        parts = []
        if concept.title:
            parts.append(f"# {concept.title}")
        parts.append(f"Type: {concept.type}")
        if concept.description:
            parts.append(f"Description: {concept.description}")
        if concept.resource:
            parts.append(f"Resource: {concept.resource}")
        if concept.tags:
            parts.append(f"Tags: {', '.join(concept.tags)}")

        # v0.2 trust + lifecycle
        parts.append(f"Trust: {_trust_badge(concept)}")
        parts.append(f"Status: {concept.status}{_staleness_warning(concept)}")
        if concept.generated_at:
            parts.append(f"Generated at: {concept.generated_at}")
        if concept.generated_by:
            parts.append(f"Generated by: {concept.generated_by}")
        for evt in concept.verified:
            parts.append(f"Verified: by={evt.get('by', '')} at={evt.get('at', '')}")

        # Provenance sources
        if concept.sources:
            parts.append("\nSources:")
            for src in concept.sources:
                src_line = f"  - {src.get('resource', '?')}"
                if src.get("title"):
                    src_line += f" ({src['title']})"
                if src.get("author"):
                    src_line += f" [author: {src['author']}]"
                parts.append(src_line)

        # Attested Computation fields
        if concept.runtime:
            parts.append(f"\nRuntime: {concept.runtime}")
        if concept.parameters:
            parts.append("Parameters:")
            for p in concept.parameters:
                parts.append(
                    f"  - {p.get('name')} ({p.get('type')}) "
                    f"{'required' if p.get('required') else 'optional'}"
                )

        # Legacy source citation
        fm = concept.frontmatter
        citation_parts: list[str] = []
        if fm.get("source_file"):
            cite = f"Source: `{fm['source_file']}`"
            if fm.get("source_page"):
                cite += f", page {fm['source_page']}"
            citation_parts.append(cite)
        if fm.get("image_source"):
            img_cite = f"Image: `{fm['image_source']}`"
            if fm.get("image_type"):
                img_cite += f" ({fm['image_type']})"
            citation_parts.append(img_cite)
        if citation_parts:
            parts.append(f"Citation: {' | '.join(citation_parts)}")

        parts.append("")
        parts.append(concept.body)
        return "\n".join(parts)

    async def _arun(self, concept_id: str) -> str:
        return self._run(concept_id)


class OKFRelatedConceptsTool(BaseTool):
    """Find concepts related to a given concept via cross-links."""

    name: str = "okf_related_concepts"
    description: str = (
        "Find concepts related to a given concept via cross-links. "
        "Shows both outgoing links (what it references) and incoming "
        "links (what references it). Input: concept ID."
    )
    graph: OKFKnowledgeGraph

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, concept_id: str) -> str:
        concept_id = concept_id.strip().strip("/")
        if concept_id not in self.graph.concept_ids:
            return f"Concept '{concept_id}' not found."

        outgoing = self.graph.neighbors(concept_id)
        incoming = self.graph.backlinks(concept_id)

        lines = [f"Relationships for '{concept_id}':"]

        if outgoing:
            lines.append(f"\nLinks to ({len(outgoing)}):")
            for cid in outgoing:
                c = self.graph.get_concept(cid)
                label = f" [{c.type}]" if c else ""
                stale = _staleness_warning(c) if c else ""
                lines.append(f"  -> {cid}{label}{stale}")
        else:
            lines.append("\nLinks to: (none)")

        if incoming:
            lines.append(f"\nLinked from ({len(incoming)}):")
            for cid in incoming:
                c = self.graph.get_concept(cid)
                label = f" [{c.type}]" if c else ""
                lines.append(f"  <- {cid}{label}")
        else:
            lines.append("\nLinked from: (none)")

        return "\n".join(lines)

    async def _arun(self, concept_id: str) -> str:
        return self._run(concept_id)


class OKFSearchTool(BaseTool):
    """Search OKF concepts by keyword matching on title, description, and body."""

    name: str = "okf_search"
    description: str = (
        "Search the knowledge base by keyword. Matches against concept "
        "titles, descriptions, tags, and body content. Input: search query. "
        "Results show trust tier and staleness indicators."
    )
    graph: OKFKnowledgeGraph
    max_results: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, query: str) -> str:
        query_lower = query.strip().lower()
        scored: list[tuple[int, str]] = []

        for cid in self.graph.concept_ids:
            concept = self.graph.get_concept(cid)
            if concept is None:
                continue
            score = 0
            if query_lower in concept.title.lower():
                score += 3
            if query_lower in concept.description.lower():
                score += 2
            if any(query_lower in t.lower() for t in concept.tags):
                score += 2
            if query_lower in cid.lower():
                score += 2
            if query_lower in concept.body.lower():
                score += 1
            if score > 0:
                scored.append((score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = scored[: self.max_results]

        if not results:
            return f"No concepts matching '{query}'."

        lines = [f"Found {len(results)} concepts matching '{query}':"]
        for _score, cid in results:
            concept = self.graph.get_concept(cid)
            if concept:
                desc = concept.description or concept.title or ""
                badge = _trust_badge(concept)
                stale = _staleness_warning(concept)
                source = _format_source_short(concept)
                source_suffix = f" {source}" if source else ""
                lines.append(
                    f"  - {cid} [{concept.type}] {badge}{stale}: {desc}{source_suffix}"
                )
        return "\n".join(lines)

    async def _arun(self, query: str) -> str:
        return self._run(query)


def _format_source_short(concept: OKFConcept) -> str:
    fm = concept.frontmatter
    parts: list[str] = []
    if fm.get("source_file"):
        s = fm["source_file"]
        if fm.get("source_page"):
            s += f":p{fm['source_page']}"
        parts.append(s)
    if fm.get("image_source"):
        parts.append(f"img:{fm['image_source']}")
    if parts:
        return f"(from {', '.join(parts)})"
    return ""


class OKFWriteConceptTool(BaseTool):
    """Write or update a concept document in the OKF bundle.

    Sub-second local operation (<5ms): immediately persists to markdown
    and registers in the live knowledge graph.
    """

    name: str = "okf_write_concept"
    description: str = (
        "Write or update a concept in the OKF knowledge bundle. "
        "Inputs: concept_id (e.g. 'rules/retry_policy' or 'preferences/python'), "
        "type (e.g. 'Policy', 'Metric', 'Finding', 'Playbook'), "
        "body (markdown text), and optional title, description, tags (comma-separated)."
    )
    bundle_path: Path
    graph: OKFKnowledgeGraph
    actor: str = "agent/langchain"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(
        self,
        concept_id: str,
        type: str,
        body: str,
        title: str = "",
        description: str = "",
        tags: str = "",
    ) -> str:
        cid = concept_id.strip().strip("/")
        if not cid:
            return "Error: concept_id is required."
        if not type:
            return "Error: concept type is required per OKF §4.1."

        file_path = self.bundle_path / f"{cid}.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Load existing frontmatter if updating
        existing_fm: dict[str, Any] = {}
        if file_path.exists():
            try:
                existing_fm, _ = parse_frontmatter(
                    file_path.read_text(encoding="utf-8")
                )
            except Exception:
                existing_fm = {}

        tag_list = (
            [t.strip() for t in tags.split(",") if t.strip()]
            if tags
            else existing_fm.get("tags", [])
        )

        frontmatter: dict[str, Any] = {
            "type": type.strip(),
            "title": title.strip() or existing_fm.get("title") or cid.split("/")[-1],
            "description": description.strip() or existing_fm.get("description", ""),
        }
        if tag_list:
            frontmatter["tags"] = tag_list

        frontmatter["status"] = existing_fm.get("status", "stable")
        frontmatter["generated"] = {
            "by": self.actor,
            "at": now,
        }
        if "verified" in existing_fm:
            frontmatter["verified"] = existing_fm["verified"]
        if "stale_after" in existing_fm:
            frontmatter["stale_after"] = existing_fm["stale_after"]
        if "sources" in existing_fm:
            frontmatter["sources"] = existing_fm["sources"]

        fm_text = yaml.safe_dump(
            frontmatter, sort_keys=False, allow_unicode=True
        ).rstrip()
        file_content = f"---\n{fm_text}\n---\n\n{body.strip()}\n"
        file_path.write_text(file_content, encoding="utf-8")

        # Update knowledge graph in-memory immediately (< 1ms)
        concept = OKFConcept(
            concept_id=cid,
            frontmatter=frontmatter,
            body=body.strip(),
            source_path=str(file_path),
        )
        self.graph._concepts[cid] = concept
        self.graph._concept_id_set.add(cid)
        if cid not in self.graph._types[type.strip()]:
            self.graph._types[type.strip()].append(cid)
        for t in tag_list:
            if cid not in self.graph._tags[t]:
                self.graph._tags[t].append(cid)
        self.graph._forward[cid] = [
            target for target in concept.cross_links() if target in self.graph._concepts
        ]

        # Log change to log.md
        log_path = self.bundle_path / "log.md"
        date_str = now[:10]
        log_entry = (
            f"\n## {date_str}\n* **Update**: Saved concept `{cid}` "
            f"[{type}] by `{self.actor}`.\n"
        )
        if log_path.exists():
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        else:
            log_path.write_text("# Bundle Update Log\n" + log_entry, encoding="utf-8")

        return f"Successfully saved concept '{cid}' [{type}]."

    async def _arun(
        self,
        concept_id: str,
        type: str,
        body: str,
        title: str = "",
        description: str = "",
        tags: str = "",
    ) -> str:
        return self._run(concept_id, type, body, title, description, tags)


class OKFVerifyConceptTool(BaseTool):
    """Verify an existing concept in the OKF bundle (§5.2, §5.3).

    Sub-second local operation: records verification event and upgrades trust tier.
    """

    name: str = "okf_verify_concept"
    description: str = (
        "Verify an existing concept in the knowledge base, upgrading its trust tier. "
        "Input: concept_id (e.g. 'rules/retry_policy') and optional actor (e.g. 'human:alice' or 'agent/auditor')."
    )
    bundle_path: Path
    graph: OKFKnowledgeGraph
    default_actor: str = "agent/verifier"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, concept_id: str, actor: str = "") -> str:
        cid = concept_id.strip().strip("/")
        file_path = self.bundle_path / f"{cid}.md"
        if not file_path.exists():
            return f"Error: Concept '{cid}' not found at {file_path}."

        verifier = actor.strip() or self.default_actor
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            fm, body = parse_frontmatter(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            return f"Error reading concept '{cid}': {e}"

        verified = fm.get("verified", [])
        if isinstance(verified, dict):
            verified = [verified]
        elif not isinstance(verified, list):
            verified = []

        verified.append({"by": verifier, "at": now})
        fm["verified"] = verified

        fm_text = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip()
        file_content = f"---\n{fm_text}\n---\n\n{body}\n"
        file_path.write_text(file_content, encoding="utf-8")

        # Reload concept in graph
        concept = OKFConcept(
            concept_id=cid,
            frontmatter=fm,
            body=body,
            source_path=str(file_path),
        )
        self.graph._concepts[cid] = concept

        return (
            f"Concept '{cid}' verified by '{verifier}'. "
            f"New trust tier: '{concept.trust_tier}'."
        )

    async def _arun(self, concept_id: str, actor: str = "") -> str:
        return self._run(concept_id, actor)


class OKFToolkit:
    """Toolkit providing all OKF agent tools for a bundle.

    Convenience wrapper that creates an ``OKFKnowledgeGraph`` and
    OKF tools in one call.

    Args:
        graph: The OKF knowledge graph backing the tools.
        bundle_path: Path to the OKF bundle root directory.
        actor: Identity string for write operations (defaults to ``agent/langchain``).

    Example:
        .. code-block:: python

            from langchain_okf import OKFToolkit

            # Read-only tools
            toolkit = OKFToolkit.from_bundle("./bundles/ga4/")
            tools = toolkit.get_tools()

            # Active memory tools (including write and verify)
            agent_tools = toolkit.get_tools(include_write=True)
    """

    def __init__(
        self,
        graph: OKFKnowledgeGraph,
        bundle_path: Path | None = None,
        actor: str = "agent/langchain",
    ) -> None:
        self.graph = graph
        self.bundle_path = bundle_path
        self.actor = actor

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        actor: str = "agent/langchain",
    ) -> OKFToolkit:
        """Create a toolkit from an OKF bundle directory.

        Args:
            bundle_path: Path to the OKF bundle root.
            actor: Actor identity for writes and verifications.

        Returns:
            An ``OKFToolkit`` with all tools configured.
        """
        b_path = Path(bundle_path)
        graph = OKFKnowledgeGraph.from_bundle(b_path)
        return cls(graph=graph, bundle_path=b_path, actor=actor)

    def get_tools(self, *, include_write: bool = False) -> list[BaseTool]:
        """Return OKF tools for use with a LangChain agent.

        Args:
            include_write: If True, also includes ``OKFWriteConceptTool``
                and ``OKFVerifyConceptTool`` for agent memory writing.

        Returns:
            List of BaseTool instances.
        """
        tools: list[BaseTool] = [
            OKFListConceptsTool(graph=self.graph),
            OKFReadConceptTool(graph=self.graph),
            OKFRelatedConceptsTool(graph=self.graph),
            OKFSearchTool(graph=self.graph),
        ]
        if include_write and self.bundle_path is not None:
            tools.extend(
                [
                    OKFWriteConceptTool(
                        graph=self.graph,
                        bundle_path=self.bundle_path,
                        actor=self.actor,
                    ),
                    OKFVerifyConceptTool(
                        graph=self.graph,
                        bundle_path=self.bundle_path,
                        default_actor=self.actor,
                    ),
                ]
            )
        return tools
