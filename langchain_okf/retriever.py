"""Graph-aware OKF retriever for LangChain — OKF v0.2.

Combines vector similarity search with OKF's cross-link graph to
retrieve not just the most relevant concepts, but also their linked
neighbors. Supports trust-tier filtering, staleness exclusion, and
**persistent** vector indexes for sub-second cold loads.

Example — one-time index build (writes to disk):
    .. code-block:: python

        from langchain_okf import OKFRetriever
        from langchain_ollama import OllamaEmbeddings

        retriever = OKFRetriever.from_bundle(
            "./bundles/ga4/",
            embeddings=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory="./bundles/ga4/.chroma",
        )
        docs = retriever.invoke("How are weekly active users computed?")

Example — subsequent calls load in milliseconds from the persisted index:
    .. code-block:: python

        retriever = OKFRetriever.from_bundle(
            "./bundles/ga4/",
            embeddings=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory="./bundles/ga4/.chroma",   # reloads, no re-embed
        )
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict

from langchain_okf._parser import OKFConcept
from langchain_okf.document_loader import OKFBundleLoader
from langchain_okf.graph import OKFKnowledgeGraph


def _bundle_content_hash(bundle_path: Path) -> str:
    """SHA-256 fingerprint of all concept .md files in a bundle.

    Used to detect whether the bundle has changed since the last index
    build so we can skip re-embedding on unchanged bundles (sub-second
    cold load).
    """
    h = hashlib.sha256()
    for md_file in sorted(bundle_path.rglob("*.md")):
        if md_file.name in ("index.md", "log.md"):
            continue
        stat = md_file.stat()
        h.update(f"{md_file}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _concept_content_hash(doc: Document) -> str:
    """SHA-256 fingerprint of a concept document's content and metadata."""
    meta_items = sorted((str(k), str(v)) for k, v in doc.metadata.items())
    payload = f"{doc.page_content}:{meta_items}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_stale(metadata: dict, now: datetime) -> bool:
    """Return True if the concept's stale_after has passed."""
    sa = metadata.get("stale_after", "")
    if not sa:
        return False
    try:
        dt = datetime.fromisoformat(str(sa).replace("Z", "+00:00"))
        return now >= dt
    except (ValueError, TypeError):
        return False


class OKFRetriever(BaseRetriever):
    """Graph-aware retriever over an OKF v0.2 knowledge bundle.

    Retrieves concepts by vector similarity, then optionally expands
    results by following cross-links in the OKF graph. Supports
    filtering by concept type, tags, trust tier, and staleness.

    Args:
        vectorstore: A LangChain ``VectorStore`` containing indexed
            OKF documents.
        graph: The ``OKFKnowledgeGraph`` for cross-link traversal.
        k: Number of documents to retrieve via similarity search.
        expand_depth: Number of cross-link hops to follow from each
            retrieved concept. Set to 0 to disable graph expansion.
        filter_types: If set, only return concepts of these types.
        filter_tags: If set, only return concepts with at least one
            of these tags.
        filter_status: If set, only return concepts with this status
            (``draft`` | ``stable`` | ``deprecated``). Default ``stable``.
        min_trust_tier: Minimum trust tier to return. One of
            ``"unverified"`` (all), ``"machine-confirmed"``,
            ``"human-reviewed"``. Default ``"unverified"`` (all).
        exclude_stale: If True, exclude concepts past their
            ``stale_after`` date. Default True.
        score_threshold: Minimum similarity score (0-1) for results.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: VectorStore
    graph: OKFKnowledgeGraph
    k: int = 5
    expand_depth: int = 1
    filter_types: list[str] | None = None
    filter_tags: list[str] | None = None
    filter_status: str | None = "stable"
    min_trust_tier: str = "unverified"
    exclude_stale: bool = True
    score_threshold: float | None = None

    _TIER_ORDER = {"unverified": 0, "machine-confirmed": 1, "human-reviewed": 2}

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore] | Any | None = None,
        vectorstore_kwargs: dict[str, Any] | None = None,
        persist_directory: str | Path | None = None,
        k: int = 5,
        expand_depth: int = 1,
        filter_types: list[str] | None = None,
        filter_tags: list[str] | None = None,
        filter_status: str | None = "stable",
        min_trust_tier: str = "unverified",
        exclude_stale: bool = True,
        score_threshold: float | None = None,
    ) -> OKFRetriever:
        """Build a retriever from an OKF bundle in one call.

        When ``persist_directory`` is provided, the vector index is
        written to disk after the first build and reloaded on subsequent
        calls without re-embedding (sub-second cold load).  A
        ``.okf_index_hash`` file is stored alongside the Chroma data to
        detect bundle changes and trigger a rebuild automatically.

        Args:
            bundle_path: Path to the OKF bundle root directory.
            embeddings: Embedding model for vector indexing.
            vectorstore_cls: Vector store class to use. Defaults to
                ``langchain_chroma.Chroma`` if installed.
            vectorstore_kwargs: Extra kwargs passed to the vector store
                constructor.
            persist_directory: Directory to persist the Chroma index.
                Subsequent calls with the same path skip re-embedding.
            k: Number of results from similarity search.
            expand_depth: Cross-link expansion depth.
            filter_types: Restrict results to these concept types.
            filter_tags: Restrict results to concepts with these tags.
            filter_status: Only return concepts with this status.
            min_trust_tier: Minimum trust tier to include.
            exclude_stale: Exclude concepts past ``stale_after``.
            score_threshold: Minimum similarity score.

        Returns:
            A configured ``OKFRetriever``.
        """
        bundle_path = Path(bundle_path)
        graph = OKFKnowledgeGraph.from_bundle(bundle_path)

        if vectorstore_cls is None:
            try:
                from langchain_chroma import Chroma

                vectorstore_cls = Chroma
            except ImportError as err:
                raise ImportError(
                    "No vector store class provided and langchain-chroma is "
                    "not installed. Install it with: pip install langchain-chroma"
                ) from err

        vs_kwargs = dict(vectorstore_kwargs or {})
        vs_cls: Any = vectorstore_cls

        # ── Persistent index: load or build ───────────────────────────────────
        vectorstore: VectorStore
        if persist_directory is not None:
            persist_dir = Path(persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            hash_file = persist_dir / ".okf_index_hash"
            manifest_file = persist_dir / ".okf_manifest.json"

            current_hash = _bundle_content_hash(bundle_path)
            stored_hash = hash_file.read_text().strip() if hash_file.exists() else ""

            has_existing = (
                hash_file.exists()
                and stored_hash == current_hash
                and any(
                    p.name not in (".okf_index_hash", ".okf_manifest.json")
                    for p in persist_dir.iterdir()
                )
            )

            if has_existing:
                # Fast path: load existing index — sub-second
                try:
                    vectorstore = vs_cls(
                        embedding_function=embeddings,
                        persist_directory=str(persist_dir),
                        **vs_kwargs,
                    )
                except TypeError:
                    vectorstore = vs_cls(
                        persist_directory=str(persist_dir),
                        **vs_kwargs,
                    )
            else:
                # Check for incremental update opportunity
                stored_manifest: dict[str, str] = {}
                if manifest_file.exists():
                    try:
                        stored_manifest = json.loads(
                            manifest_file.read_text(encoding="utf-8")
                        )
                    except Exception:
                        stored_manifest = {}

                loader = OKFBundleLoader(bundle_path)
                docs = loader.load()
                current_manifest = {
                    d.metadata.get("concept_id", ""): _concept_content_hash(d)
                    for d in docs
                    if d.metadata.get("concept_id")
                }

                can_incremental = bool(stored_manifest) and any(
                    p.name not in (".okf_index_hash", ".okf_manifest.json")
                    for p in persist_dir.iterdir()
                )

                synced_incrementally = False
                if can_incremental:
                    try:
                        try:
                            vectorstore = vs_cls(
                                embedding_function=embeddings,
                                persist_directory=str(persist_dir),
                                **vs_kwargs,
                            )
                        except TypeError:
                            vectorstore = vs_cls(
                                persist_directory=str(persist_dir),
                                **vs_kwargs,
                            )

                        if hasattr(vectorstore, "add_documents"):
                            # Determine delta
                            new_or_modified_docs = [
                                d
                                for d in docs
                                if current_manifest.get(
                                    str(d.metadata.get("concept_id", ""))
                                )
                                != stored_manifest.get(
                                    str(d.metadata.get("concept_id", ""))
                                )
                            ]
                            deleted_ids = [
                                cid
                                for cid in stored_manifest
                                if cid not in current_manifest
                            ]
                            to_delete = deleted_ids + [
                                d.metadata["concept_id"]
                                for d in new_or_modified_docs
                                if d.metadata.get("concept_id") in stored_manifest
                            ]

                            if to_delete and hasattr(vectorstore, "delete"):
                                with contextlib.suppress(Exception):
                                    vectorstore.delete(ids=to_delete)

                            if new_or_modified_docs:
                                ids = [
                                    d.metadata.get("concept_id", "")
                                    for d in new_or_modified_docs
                                ]
                                vectorstore.add_documents(new_or_modified_docs, ids=ids)

                            manifest_file.write_text(
                                json.dumps(current_manifest, indent=2), encoding="utf-8"
                            )
                            hash_file.write_text(current_hash)
                            synced_incrementally = True
                    except Exception:
                        synced_incrementally = False

                if not synced_incrementally:
                    # Full cold build and persist
                    vectorstore = vs_cls.from_documents(
                        docs,
                        embedding=embeddings,
                        persist_directory=str(persist_dir),
                        **vs_kwargs,
                    )
                    manifest_file.write_text(
                        json.dumps(current_manifest, indent=2), encoding="utf-8"
                    )
                    hash_file.write_text(current_hash)
        else:
            # In-memory (no persistence)
            loader = OKFBundleLoader(bundle_path)
            docs = loader.load()
            vectorstore = vs_cls.from_documents(docs, embedding=embeddings, **vs_kwargs)

        return cls(
            vectorstore=vectorstore,
            graph=graph,
            k=k,
            expand_depth=expand_depth,
            filter_types=filter_types,
            filter_tags=filter_tags,
            filter_status=filter_status,
            min_trust_tier=min_trust_tier,
            exclude_stale=exclude_stale,
            score_threshold=score_threshold,
        )

    def _build_vs_filter(self) -> dict | None:
        if self.filter_types and len(self.filter_types) == 1:
            return {"type": self.filter_types[0]}
        return None

    def _passes_filters(self, doc: Document) -> bool:
        meta = doc.metadata
        now = datetime.now(timezone.utc)

        if self.filter_types and meta.get("type") not in self.filter_types:
            return False
        if self.filter_tags:
            doc_tags = set(meta.get("tags", []))
            if not doc_tags & set(self.filter_tags):
                return False
        if self.filter_status and meta.get("status", "stable") != self.filter_status:
            return False
        if self.exclude_stale and _is_stale(meta, now):
            return False
        if self.min_trust_tier != "unverified":
            tier = meta.get("trust_tier", "unverified")
            if self._TIER_ORDER.get(tier, 0) < self._TIER_ORDER.get(
                self.min_trust_tier, 0
            ):
                return False
        return True

    def _get_retrieve_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"k": self.k}
        vs_filter = self._build_vs_filter()
        if vs_filter:
            params["filter"] = vs_filter
        return params

    def _expand_with_graph(self, docs: list[Document]) -> list[Document]:
        if self.expand_depth <= 0:
            return docs

        seen_ids: set[str] = set()
        result: list[Document] = []

        for doc in docs:
            cid = doc.metadata.get("concept_id", "")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            result.append(doc)

        expansion_ids: set[str] = set()
        for doc in docs:
            cid = doc.metadata.get("concept_id", "")
            if cid:
                expansion_ids |= self.graph.subgraph(cid, depth=self.expand_depth)

        for cid in expansion_ids - seen_ids:
            concept = self.graph.get_concept(cid)
            if concept is None:
                continue
            expanded_doc = self._concept_to_document(concept)
            if self._passes_filters(expanded_doc):
                result.append(expanded_doc)
                seen_ids.add(cid)

        return result

    @staticmethod
    def _concept_to_document(concept: OKFConcept) -> Document:
        metadata: dict[str, Any] = {
            "concept_id": concept.concept_id,
            "source": concept.source_path,
            "expanded_via_graph": True,
        }
        if concept.type:
            metadata["type"] = concept.type
        if concept.title:
            metadata["title"] = concept.title
        if concept.description:
            metadata["description"] = concept.description
        if concept.tags:
            metadata["tags"] = concept.tags
        # v0.2 trust + lifecycle
        metadata["status"] = concept.status
        metadata["trust_tier"] = concept.trust_tier
        if concept.stale_after:
            metadata["stale_after"] = concept.stale_after
        if concept.generated_at:
            metadata["generated_at"] = concept.generated_at
        return Document(page_content=concept.body, metadata=metadata)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve documents relevant to the query.

        1. Vector similarity search over indexed OKF concepts.
        2. Filter by type / tags / status / trust / staleness.
        3. Expand results via cross-link graph traversal.
        """
        params = self._get_retrieve_params()

        if self.score_threshold is not None:
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                query, **params
            )
            docs = [doc for doc, score in scored if score >= self.score_threshold]
        else:
            docs = self.vectorstore.similarity_search(query, **params)

        docs = [d for d in docs if self._passes_filters(d)]
        docs = self._expand_with_graph(docs)
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Async variant of _get_relevant_documents."""
        params = self._get_retrieve_params()

        if self.score_threshold is not None:
            scored = await self.vectorstore.asimilarity_search_with_relevance_scores(
                query, **params
            )
            docs = [doc for doc, score in scored if score >= self.score_threshold]
        else:
            docs = await self.vectorstore.asimilarity_search(query, **params)

        docs = [d for d in docs if self._passes_filters(d)]
        docs = self._expand_with_graph(docs)
        return docs
