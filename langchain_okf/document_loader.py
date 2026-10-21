"""OKF Bundle Document Loader for LangChain — OKF v0.2.

Loads an OKF v0.2 knowledge bundle into LangChain ``Document`` objects,
mapping frontmatter fields to metadata and preserving concept identity.

v0.2 additions: ``generated``, ``verified``, ``sources``, ``status``,
``stale_after``, and ``trust_tier`` are mapped to Document metadata so
retrievers can filter on them.

See the `OKF spec <https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md>`_
for format details.

Example:
    .. code-block:: python

        from langchain_okf import OKFBundleLoader

        loader = OKFBundleLoader("path/to/bundle/")
        docs = loader.load()

        # Each doc has OKF v0.2 frontmatter as metadata
        docs[0].metadata["type"]         # "BigQuery Table"
        docs[0].metadata["concept_id"]   # "tables/orders"
        docs[0].metadata["tags"]         # ["sales", "orders"]
        docs[0].metadata["trust_tier"]   # "human-reviewed"
        docs[0].metadata["status"]       # "stable"
        docs[0].metadata["generated_at"] # "2026-06-20T22:53:05Z"
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_okf._parser import OKFConcept, parse_bundle


class OKFBundleLoader(BaseLoader):
    """Load an OKF v0.2 knowledge bundle as LangChain Documents.

    Each concept document in the bundle becomes one ``Document``, with
    YAML frontmatter fields mapped to ``Document.metadata``. The
    concept's markdown body becomes ``page_content``.

    v0.2 metadata fields included in every document:
    - ``trust_tier``: ``"human-reviewed"`` | ``"machine-confirmed"`` | ``"unverified"``
    - ``status``: ``"draft"`` | ``"stable"`` | ``"deprecated"``
    - ``stale_after``: ISO 8601 datetime (if set)
    - ``generated_at`` / ``generated_by``: authorship info
    - ``sources``: list of provenance entries

    Args:
        bundle_path: Path to the OKF bundle root directory.
        include_types: If set, only load concepts matching these types.
        exclude_types: If set, skip concepts matching these types.
        include_tags: If set, only load concepts having at least one
            of these tags.
        include_status: If set, only load concepts with this status.
            Common values: ``"stable"``, ``"draft"``, ``"deprecated"``.
        exclude_stale: If True (default), exclude concepts whose
            ``stale_after`` has passed.

    Example:
        .. code-block:: python

            from langchain_okf import OKFBundleLoader

            loader = OKFBundleLoader(
                "./bundles/ga4/",
                include_status="stable",
                exclude_stale=True,
            )
            docs = loader.load()
    """

    def __init__(
        self,
        bundle_path: str | Path,
        *,
        include_types: list[str] | None = None,
        exclude_types: list[str] | None = None,
        include_tags: list[str] | None = None,
        include_status: str | None = None,
        exclude_stale: bool = True,
    ) -> None:
        self.bundle_path = Path(bundle_path)
        self.include_types = set(include_types) if include_types else None
        self.exclude_types = set(exclude_types) if exclude_types else None
        self.include_tags = set(include_tags) if include_tags else None
        self.include_status = include_status
        self.exclude_stale = exclude_stale

    def _should_include(self, concept: OKFConcept) -> bool:
        if self.include_types and concept.type not in self.include_types:
            return False
        if self.exclude_types and concept.type in self.exclude_types:
            return False
        if self.include_tags and not (set(concept.tags) & self.include_tags):
            return False
        if self.include_status and concept.status != self.include_status:
            return False
        return not (self.exclude_stale and concept.is_stale)

    def _concept_to_document(self, concept: OKFConcept) -> Document:
        metadata: dict = {
            "concept_id": concept.concept_id,
            "source": concept.source_path,
        }

        # ── Core fields ───────────────────────────────────────────────────────
        if concept.type:
            metadata["type"] = concept.type
        if concept.title:
            metadata["title"] = concept.title
        if concept.description:
            metadata["description"] = concept.description
        if concept.resource:
            metadata["resource"] = concept.resource
        if concept.tags:
            metadata["tags"] = concept.tags

        # ── v0.2 trust ────────────────────────────────────────────────────────
        metadata["trust_tier"] = concept.trust_tier
        metadata["status"] = concept.status
        if concept.stale_after:
            metadata["stale_after"] = concept.stale_after
        if concept.generated_at:
            metadata["generated_at"] = concept.generated_at
        if concept.generated_by:
            metadata["generated_by"] = concept.generated_by

        # ── v0.2 provenance ───────────────────────────────────────────────────
        if concept.sources:
            # Store as JSON string so vector stores (e.g. Chroma) accept it
            metadata["sources"] = json.dumps(concept.sources, default=str)
            first_resource = concept.sources[0].get("resource", "")
            if first_resource:
                metadata["primary_source"] = first_resource

        # ── v0.1 compat ───────────────────────────────────────────────────────
        if concept.timestamp:
            metadata["timestamp"] = concept.timestamp

        # ── Cross-links ───────────────────────────────────────────────────────
        cross_links = concept.cross_links()
        if cross_links:
            metadata["cross_links"] = cross_links

        # ── Attested Computation ──────────────────────────────────────────────
        if concept.runtime:
            metadata["runtime"] = concept.runtime

        # ── Remaining frontmatter (extension fields) ──────────────────────────
        skip = {
            "type",
            "title",
            "description",
            "resource",
            "tags",
            "generated",
            "verified",
            "sources",
            "status",
            "stale_after",
            "timestamp",
            "runtime",
            "parameters",
            "executor",
            "attester",
        }
        for key, value in concept.frontmatter.items():
            if key not in skip and key not in metadata:
                metadata[key] = value

        return Document(page_content=concept.body, metadata=metadata)

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load OKF concepts as Documents.

        Yields:
            One ``Document`` per concept, with frontmatter as metadata.
        """
        for concept in parse_bundle(self.bundle_path):
            if self._should_include(concept):
                yield self._concept_to_document(concept)
