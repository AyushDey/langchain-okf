"""Tests for OKF v0.2 features: trust, lifecycle, provenance, and persistent index."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from langchain_okf._parser import OKFConcept, parse_frontmatter
from langchain_okf.document_loader import OKFBundleLoader
from langchain_okf.generator import OKFBundleGenerator, _llm_actor_name
from langchain_okf.retriever import OKFRetriever, _bundle_content_hash, _is_stale
from langchain_okf.toolkit import OKFToolkit
from langchain_okf.validate import validate_bundle

# ── Fixtures ──────────────────────────────────────────────────────────────────

V02_CONCEPT = """\
---
type: Metric
title: Revenue
description: Recognized revenue for a fiscal year.
tags: [finance, revenue]
status: stable
stale_after: 2099-01-01T00:00:00Z
generated:
  by: okf-generator/gpt-4o
  at: 2026-06-20T22:53:05Z
verified:
  - by: human:alice
    at: 2026-06-25T09:00:00Z
sources:
  - id: rev-policy
    resource: https://wiki.acme/finance/revenue-recognition
    title: Revenue recognition policy
    author: team:finance-fpa
    last_modified: 2026-04-02T00:00:00Z
---

# Definition

Recognized revenue sums `amount` over rows booked to the fiscal year.
"""

STALE_CONCEPT = """\
---
type: Reference
title: Old Pricing
description: Outdated pricing sheet.
status: stable
stale_after: 2020-01-01T00:00:00Z
generated:
  by: process:importer
  at: 2019-06-01T00:00:00Z
---

Some old content.
"""

DRAFT_CONCEPT = """\
---
type: Playbook
title: Draft Playbook
description: Work in progress.
status: draft
generated:
  by: human:bob
  at: 2026-10-01T00:00:00Z
---

Draft content.
"""

ATTESTED_CONCEPT = """\
---
type: Attested Computation
title: Revenue Computation
description: Sanctioned SQL for revenue.
status: stable
runtime: bigquery
parameters:
  - name: year
    type: integer
    required: true
executor:
  resource: references/skills/run-on-bq.md
  receipt: [job_id, executed_sql, result]
attester:
  resource: references/attesters/revenue.py
generated:
  by: reference_agent/gemini-2.5-pro
  at: 2026-06-20T22:53:05Z
verified:
  by: human:ahormati
  at: 2026-06-25T09:00:00Z
sources:
  - id: rev-policy
    resource: https://wiki.acme/finance/revenue-recognition
    title: Revenue recognition policy
---

# Computation

    SELECT SUM(amount) AS revenue
    FROM finance.recognized_revenue
    WHERE fiscal_year = @year
"""


@pytest.fixture
def v02_bundle(tmp_path: Path) -> Path:
    """Create a minimal v0.2-compliant bundle."""
    metrics = tmp_path / "metrics"
    metrics.mkdir()

    (metrics / "revenue.md").write_text(V02_CONCEPT, encoding="utf-8")
    (metrics / "old_pricing.md").write_text(STALE_CONCEPT, encoding="utf-8")
    (metrics / "draft_playbook.md").write_text(DRAFT_CONCEPT, encoding="utf-8")

    computations = tmp_path / "computations"
    computations.mkdir()
    (computations / "revenue_sql.md").write_text(ATTESTED_CONCEPT, encoding="utf-8")

    # root index.md with okf_version
    (tmp_path / "index.md").write_text(
        '---\nokf_version: "0.2"\n---\n\n# Test Bundle\n', encoding="utf-8"
    )
    return tmp_path


# ── Parser v0.2 property tests ────────────────────────────────────────────────


class TestV02Parser:
    def test_generated_fields(self):
        fm, _ = parse_frontmatter(V02_CONCEPT)
        c = OKFConcept("metrics/revenue", frontmatter=fm, body="")
        assert c.generated_by == "okf-generator/gpt-4o"
        assert c.generated_at == "2026-06-20T22:53:05Z"

    def test_verified_list(self):
        fm, _ = parse_frontmatter(V02_CONCEPT)
        c = OKFConcept("metrics/revenue", frontmatter=fm, body="")
        assert len(c.verified) == 1
        assert c.verified[0]["by"] == "human:alice"

    def test_verified_bare_mapping(self):
        """§5.2: a bare mapping must be treated as a one-element list."""
        fm, _ = parse_frontmatter(ATTESTED_CONCEPT)
        c = OKFConcept("computations/revenue_sql", frontmatter=fm, body="")
        events = c.verified
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["by"] == "human:ahormati"

    def test_trust_tier_human_reviewed(self):
        fm, _ = parse_frontmatter(V02_CONCEPT)
        c = OKFConcept("metrics/revenue", frontmatter=fm, body="")
        assert c.trust_tier == "human-reviewed"

    def test_trust_tier_machine_confirmed(self):
        text = V02_CONCEPT.replace("human:alice", "process:nightly")
        fm, _ = parse_frontmatter(text)
        c = OKFConcept("metrics/revenue", frontmatter=fm, body="")
        assert c.trust_tier == "machine-confirmed"

    def test_trust_tier_unverified(self):
        fm, _ = parse_frontmatter(DRAFT_CONCEPT)
        c = OKFConcept("metrics/draft", frontmatter=fm, body="")
        assert c.trust_tier == "unverified"

    def test_status(self):
        fm, _ = parse_frontmatter(V02_CONCEPT)
        c = OKFConcept("metrics/revenue", frontmatter=fm, body="")
        assert c.status == "stable"

    def test_status_default(self):
        c = OKFConcept("metrics/x", frontmatter={"type": "Concept"})
        assert c.status == "stable"

    def test_stale_after_fresh(self):
        fm, _ = parse_frontmatter(V02_CONCEPT)
        c = OKFConcept("metrics/revenue", frontmatter=fm, body="")
        assert c.stale_after == "2099-01-01T00:00:00Z"
        assert not c.is_stale

    def test_stale_after_expired(self):
        fm, _ = parse_frontmatter(STALE_CONCEPT)
        c = OKFConcept("metrics/old", frontmatter=fm, body="")
        assert c.is_stale

    def test_sources(self):
        fm, _ = parse_frontmatter(V02_CONCEPT)
        c = OKFConcept("metrics/revenue", frontmatter=fm, body="")
        assert len(c.sources) == 1
        assert (
            c.sources[0]["resource"] == "https://wiki.acme/finance/revenue-recognition"
        )
        assert c.sources[0]["author"] == "team:finance-fpa"

    def test_attested_computation_fields(self):
        fm, _ = parse_frontmatter(ATTESTED_CONCEPT)
        c = OKFConcept("computations/revenue_sql", frontmatter=fm, body="")
        assert c.runtime == "bigquery"
        assert len(c.parameters) == 1
        assert c.parameters[0]["name"] == "year"
        assert c.executor["resource"] == "references/skills/run-on-bq.md"
        assert c.attester["resource"] == "references/attesters/revenue.py"


# ── Document Loader v0.2 metadata tests ───────────────────────────────────────


class TestV02Loader:
    def test_trust_tier_in_metadata(self, v02_bundle: Path):
        loader = OKFBundleLoader(v02_bundle, exclude_stale=False, include_status=None)
        docs = loader.load()
        revenue = next(
            d for d in docs if d.metadata.get("concept_id") == "metrics/revenue"
        )
        assert revenue.metadata["trust_tier"] == "human-reviewed"
        assert revenue.metadata["status"] == "stable"
        assert revenue.metadata["generated_by"] == "okf-generator/gpt-4o"

    def test_sources_in_metadata(self, v02_bundle: Path):
        loader = OKFBundleLoader(v02_bundle, exclude_stale=False, include_status=None)
        docs = loader.load()
        revenue = next(
            d for d in docs if d.metadata.get("concept_id") == "metrics/revenue"
        )
        assert "sources" in revenue.metadata
        assert revenue.metadata["primary_source"].startswith("https://")

    def test_exclude_stale_default(self, v02_bundle: Path):
        loader = OKFBundleLoader(v02_bundle, include_status=None)
        docs = loader.load()
        ids = [d.metadata["concept_id"] for d in docs]
        assert "metrics/old_pricing" not in ids  # stale
        assert "metrics/revenue" in ids  # fresh

    def test_include_status_filter(self, v02_bundle: Path):
        loader = OKFBundleLoader(
            v02_bundle, include_status="draft", exclude_stale=False
        )
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata["concept_id"] == "metrics/draft_playbook"


# ── Validator v0.2 tests ──────────────────────────────────────────────────────


class TestV02Validator:
    def test_valid_bundle(self, v02_bundle: Path):
        issues = validate_bundle(v02_bundle)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_strict_valid(self, v02_bundle: Path):
        issues = validate_bundle(v02_bundle, strict=True)
        assert issues == [], f"Strict issues: {issues}"

    def test_missing_type(self, tmp_path: Path):
        (tmp_path / "bad.md").write_text("---\ntitle: No type\n---\n", encoding="utf-8")
        issues = validate_bundle(tmp_path)
        assert any("type" in i for i in issues)

    def test_strict_invalid_actor(self, tmp_path: Path):
        (tmp_path / "bad.md").write_text(
            "---\ntype: Metric\ngenerated:\n  by: not-an-actor\n  at: 2026-01-01T00:00:00Z\n---\n",
            encoding="utf-8",
        )
        issues = validate_bundle(tmp_path, strict=True)
        assert any("actor" in i for i in issues)

    def test_strict_invalid_iso8601(self, tmp_path: Path):
        (tmp_path / "bad.md").write_text(
            "---\ntype: Metric\nstale_after: next-week\n---\n",
            encoding="utf-8",
        )
        issues = validate_bundle(tmp_path, strict=True)
        assert any("stale_after" in i for i in issues)

    def test_strict_missing_sources_resource(self, tmp_path: Path):
        (tmp_path / "bad.md").write_text(
            "---\ntype: Metric\nsources:\n  - title: No resource here\n---\n",
            encoding="utf-8",
        )
        issues = validate_bundle(tmp_path, strict=True)
        assert any("sources" in i for i in issues)

    def test_strict_attested_computation_missing_runtime(self, tmp_path: Path):
        (tmp_path / "bad.md").write_text(
            "---\ntype: Attested Computation\ntitle: Broken\n---\n",
            encoding="utf-8",
        )
        issues = validate_bundle(tmp_path, strict=True)
        assert any("runtime" in i for i in issues)


# ── Retriever staleness / trust filtering ─────────────────────────────────────


class TestV02RetrieverFilters:
    def test_is_stale_expired(self):
        meta = {"stale_after": "2020-01-01T00:00:00Z"}
        assert _is_stale(meta, datetime.now(timezone.utc))

    def test_is_stale_future(self):
        meta = {"stale_after": "2099-01-01T00:00:00Z"}
        assert not _is_stale(meta, datetime.now(timezone.utc))

    def test_is_stale_no_field(self):
        assert not _is_stale({}, datetime.now(timezone.utc))

    def test_bundle_content_hash_stable(self, v02_bundle: Path):
        h1 = _bundle_content_hash(v02_bundle)
        h2 = _bundle_content_hash(v02_bundle)
        assert h1 == h2

    def test_bundle_content_hash_changes_on_write(self, v02_bundle: Path):
        h1 = _bundle_content_hash(v02_bundle)
        (v02_bundle / "metrics" / "new.md").write_text(
            "---\ntype: Metric\ntitle: New\n---\n", encoding="utf-8"
        )
        h2 = _bundle_content_hash(v02_bundle)
        assert h1 != h2


# ── Toolkit v0.2 trust output ─────────────────────────────────────────────────


class TestV02Toolkit:
    def test_list_by_trust(self, v02_bundle: Path):
        toolkit = OKFToolkit.from_bundle(v02_bundle)
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["okf_list_concepts"]._run("trust:human-reviewed")
        assert "revenue" in result.lower()
        assert "human-reviewed" in result

    def test_read_concept_shows_trust(self, v02_bundle: Path):
        toolkit = OKFToolkit.from_bundle(v02_bundle)
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["okf_read_concept"]._run("metrics/revenue")
        assert "human-reviewed" in result
        assert "Sources:" in result
        assert "okf-generator/gpt-4o" in result

    def test_read_concept_shows_stale_warning(self, v02_bundle: Path):
        toolkit = OKFToolkit.from_bundle(v02_bundle)
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["okf_read_concept"]._run("metrics/old_pricing")
        assert "STALE" in result

    def test_search_shows_trust_badge(self, v02_bundle: Path):
        toolkit = OKFToolkit.from_bundle(v02_bundle)
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["okf_search"]._run("revenue")
        assert (
            "human-reviewed" in result
            or "unverified" in result
            or "machine-confirmed" in result
        )

    def test_list_by_status_draft(self, v02_bundle: Path):
        toolkit = OKFToolkit.from_bundle(v02_bundle)
        tools = {t.name: t for t in toolkit.get_tools()}
        result = tools["okf_list_concepts"]._run("status:draft")
        assert "Draft Playbook" in result


# ── Generator v0.2 output tests ───────────────────────────────────────────────


class TestV02Generator:
    def _mock_llm(self, concepts: list[dict]) -> MagicMock:
        import json

        mock = MagicMock()
        response = MagicMock()
        response.content = json.dumps(concepts)
        response.response_metadata = {}
        mock.invoke.return_value = response
        mock.model_name = "test-model"
        return mock

    def test_writes_generated_field(self, tmp_path: Path):
        concepts = [
            {
                "type": "Metric",
                "title": "Revenue",
                "description": "Revenue metric.",
                "tags": ["finance"],
                "category": "metrics",
                "filename": "revenue",
                "body": "Revenue body.",
            }
        ]
        llm = self._mock_llm(concepts)
        gen = OKFBundleGenerator(llm=llm, output_dir=tmp_path / "bundle")
        gen.add_source(__file__)  # use this test file as a source
        gen.generate()

        md_files = list((tmp_path / "bundle").rglob("revenue.md"))
        assert md_files, "revenue.md not found"
        fm, _ = parse_frontmatter(md_files[0].read_text())
        assert "generated" in fm
        assert "by" in fm["generated"]
        assert "at" in fm["generated"]
        assert fm["generated"]["by"].startswith("okf-generator/")
        assert "timestamp" not in fm, "v0.1 timestamp field must not be present"

    def test_writes_status_and_no_timestamp(self, tmp_path: Path):
        concepts = [
            {
                "type": "Playbook",
                "title": "Incident",
                "description": "Incident playbook.",
                "tags": [],
                "category": "playbooks",
                "filename": "incident",
                "body": "Steps.",
            }
        ]
        llm = self._mock_llm(concepts)
        gen = OKFBundleGenerator(llm=llm, output_dir=tmp_path / "bundle")
        gen.add_source(__file__)
        gen.generate()

        md_files = list((tmp_path / "bundle").rglob("incident.md"))
        assert md_files
        fm, _ = parse_frontmatter(md_files[0].read_text())
        assert fm.get("status") == "stable"
        assert "timestamp" not in fm

    def test_writes_stale_after_when_configured(self, tmp_path: Path):
        concepts = [
            {
                "type": "Metric",
                "title": "Revenue",
                "description": "Revenue.",
                "tags": [],
                "category": "metrics",
                "filename": "revenue",
                "body": ".",
            }
        ]
        llm = self._mock_llm(concepts)
        gen = OKFBundleGenerator(
            llm=llm, output_dir=tmp_path / "bundle", stale_after_days=30
        )
        gen.add_source(__file__)
        gen.generate()

        md_files = list((tmp_path / "bundle").rglob("revenue.md"))
        assert md_files
        fm, _ = parse_frontmatter(md_files[0].read_text())
        assert "stale_after" in fm

    def test_writes_sources_from_source_file(self, tmp_path: Path):
        concepts = [
            {
                "type": "Metric",
                "title": "Revenue",
                "description": "Revenue.",
                "tags": [],
                "category": "metrics",
                "filename": "revenue",
                "body": ".",
                "source_file": "report.pdf",
                "source_page": 3,
            }
        ]
        llm = self._mock_llm(concepts)
        gen = OKFBundleGenerator(llm=llm, output_dir=tmp_path / "bundle")
        gen.add_source(__file__)
        gen.generate()

        md_files = list((tmp_path / "bundle").rglob("revenue.md"))
        assert md_files
        fm, _ = parse_frontmatter(md_files[0].read_text())
        assert "sources" in fm
        assert fm["sources"][0]["resource"] == "report.pdf"
        assert "source_file" not in fm, (
            "source_file flat key must not be in v0.2 frontmatter"
        )

    def test_writes_log_md(self, tmp_path: Path):
        concepts = [
            {
                "type": "Metric",
                "title": "Revenue",
                "description": "Revenue.",
                "tags": [],
                "category": "metrics",
                "filename": "revenue",
                "body": ".",
            }
        ]
        llm = self._mock_llm(concepts)
        gen = OKFBundleGenerator(llm=llm, output_dir=tmp_path / "bundle")
        gen.add_source(__file__)
        gen.generate()

        log_path = tmp_path / "bundle" / "log.md"
        assert log_path.exists(), "log.md must be written by generator"
        content = log_path.read_text()
        assert "Initial generation" in content
        assert "okf-generator" in content

    def test_writes_okf_version_in_index(self, tmp_path: Path):
        concepts = [
            {
                "type": "Metric",
                "title": "Revenue",
                "description": "Revenue.",
                "tags": [],
                "category": "metrics",
                "filename": "revenue",
                "body": ".",
            }
        ]
        llm = self._mock_llm(concepts)
        gen = OKFBundleGenerator(llm=llm, output_dir=tmp_path / "bundle")
        gen.add_source(__file__)
        gen.generate()

        index_path = tmp_path / "bundle" / "index.md"
        content = index_path.read_text()
        assert 'okf_version: "0.2"' in content

    def test_incremental_appends_to_log(self, tmp_path: Path):
        """Two generate() calls should produce two log entries."""
        concepts = [
            {
                "type": "Metric",
                "title": "Revenue",
                "description": "Revenue.",
                "tags": [],
                "category": "metrics",
                "filename": "revenue",
                "body": ".",
            }
        ]
        llm = self._mock_llm(concepts)
        bundle = tmp_path / "bundle"
        gen = OKFBundleGenerator(llm=llm, output_dir=bundle)
        gen.add_source(__file__)
        gen.generate()

        # Second run — incremental
        gen2 = OKFBundleGenerator(llm=self._mock_llm(concepts), output_dir=bundle)
        gen2.add_source(__file__)
        gen2.generate()

        log_path = bundle / "log.md"
        content = log_path.read_text()
        # Should have at least two ## date entries
        assert content.count("## ") >= 2

    def test_llm_actor_name_extraction(self):
        mock = MagicMock()
        mock.model_name = "gpt-4o"
        assert _llm_actor_name(mock) == "okf-generator/gpt-4o"

    def test_llm_actor_name_fallback(self):
        mock = MagicMock(spec=[])  # no attributes
        assert _llm_actor_name(mock) == "okf-generator/unknown"


# ── Sub-second performance: persistent index round-trip ───────────────────────


class TestPersistentIndex:
    """Verify that the persistent index reloads in well under 1 second."""

    @pytest.fixture
    def fake_vectorstore_cls(self):
        from langchain_core.vectorstores import VectorStore

        class FakeVectorStore(VectorStore):
            def __init__(self, docs=None, persist_directory=None, **kwargs):
                self.docs = docs or []
                self.persist_directory = persist_directory

            @classmethod
            def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
                from langchain_core.documents import Document

                docs = [
                    Document(page_content=t, metadata=m or {})
                    for t, m in zip(texts, metadatas or [{}] * len(texts), strict=False)
                ]
                return cls(docs=docs)

            @classmethod
            def from_documents(
                cls, docs, embedding=None, persist_directory=None, **kwargs
            ):
                if embedding:
                    embedding.embed_documents([d.page_content for d in docs])
                if persist_directory:
                    p = Path(persist_directory)
                    p.mkdir(parents=True, exist_ok=True)
                    (p / "index_data.bin").write_text("persisted")
                return cls(docs=docs, persist_directory=persist_directory)

            def similarity_search(self, query, k=5, **kwargs):
                return self.docs[:k]

            def similarity_search_with_relevance_scores(self, query, k=5, **kwargs):
                return [(d, 0.9) for d in self.docs[:k]]

        return FakeVectorStore

    def test_hash_file_written(
        self, v02_bundle: Path, tmp_path: Path, fake_vectorstore_cls
    ):
        """from_bundle with persist_directory writes a hash file."""
        persist_dir = tmp_path / "chroma"

        from langchain_core.embeddings import Embeddings

        class FakeEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]

            def embed_query(self, text):
                return [0.1, 0.2, 0.3]

        OKFRetriever.from_bundle(
            v02_bundle,
            embeddings=FakeEmbeddings(),
            vectorstore_cls=fake_vectorstore_cls,
            persist_directory=persist_dir,
            filter_status=None,
        )

        hash_file = persist_dir / ".okf_index_hash"
        assert hash_file.exists(), ".okf_index_hash must be written"
        stored = hash_file.read_text().strip()
        expected = _bundle_content_hash(v02_bundle)
        assert stored == expected

    def test_reload_is_sub_second(
        self, v02_bundle: Path, tmp_path: Path, fake_vectorstore_cls
    ):
        """Second from_bundle call with same persist_directory must reload in <1s."""
        persist_dir = tmp_path / "chroma"

        from langchain_core.embeddings import Embeddings

        class FakeEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.1, 0.2, 0.3] for _ in texts]

            def embed_query(self, text):
                return [0.1, 0.2, 0.3]

        # First build
        OKFRetriever.from_bundle(
            v02_bundle,
            embeddings=FakeEmbeddings(),
            vectorstore_cls=fake_vectorstore_cls,
            persist_directory=persist_dir,
            filter_status=None,
        )

        # Second load — must be fast
        t0 = time.perf_counter()
        OKFRetriever.from_bundle(
            v02_bundle,
            embeddings=FakeEmbeddings(),
            vectorstore_cls=fake_vectorstore_cls,
            persist_directory=persist_dir,
            filter_status=None,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Persistent reload took {elapsed:.3f}s — expected <1s"

    def test_stale_index_rebuilt_on_bundle_change(
        self, v02_bundle: Path, tmp_path: Path, fake_vectorstore_cls
    ):
        """When the bundle changes, the hash mismatch triggers a rebuild."""
        persist_dir = tmp_path / "chroma"

        from langchain_core.embeddings import Embeddings

        embed_calls: list[int] = []

        class CountingEmbeddings(Embeddings):
            def embed_documents(self, texts):
                embed_calls.append(len(texts))
                return [[0.1, 0.2, 0.3] for _ in texts]

            def embed_query(self, text):
                return [0.1, 0.2, 0.3]

        # First build
        OKFRetriever.from_bundle(
            v02_bundle,
            embeddings=CountingEmbeddings(),
            vectorstore_cls=fake_vectorstore_cls,
            persist_directory=persist_dir,
            filter_status=None,
        )
        calls_after_first = len(embed_calls)

        # Mutate the bundle
        (v02_bundle / "metrics" / "new_concept.md").write_text(
            "---\ntype: Metric\ntitle: New\ndescription: New metric.\n---\nBody.\n",
            encoding="utf-8",
        )

        # Second load — must rebuild because hash changed
        OKFRetriever.from_bundle(
            v02_bundle,
            embeddings=CountingEmbeddings(),
            vectorstore_cls=fake_vectorstore_cls,
            persist_directory=persist_dir,
            filter_status=None,
        )
        assert len(embed_calls) > calls_after_first, (
            "Expected embed_documents to be called again after bundle change"
        )


# ── Sub-second agent memory operations: write and verify ─────────────────────


class TestAgentMemoryOperations:
    def test_toolkit_write_tools_included_when_requested(self, v02_bundle: Path):
        toolkit = OKFToolkit.from_bundle(v02_bundle)
        read_tools = toolkit.get_tools()
        assert len(read_tools) == 4

        all_tools = toolkit.get_tools(include_write=True)
        assert len(all_tools) == 6
        tool_names = {t.name for t in all_tools}
        assert "okf_write_concept" in tool_names
        assert "okf_verify_concept" in tool_names

    def test_sub_second_write_and_immediate_read(self, v02_bundle: Path):
        """Writing a new concept takes <10ms and is immediately readable."""
        toolkit = OKFToolkit.from_bundle(v02_bundle, actor="agent/test-bot")
        tools = {t.name: t for t in toolkit.get_tools(include_write=True)}

        t0 = time.perf_counter()
        write_result = tools["okf_write_concept"]._run(
            concept_id="rules/agent_memory_rule",
            type="Policy",
            title="Memory Retention Policy",
            description="Policy on remembering user settings.",
            body="Always prioritize verified concepts over unverified drafts.",
            tags="memory,policy",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"Write took {elapsed_ms:.1f}ms — expected <100ms"
        assert "Successfully saved" in write_result

        # Check immediate read
        read_result = tools["okf_read_concept"]._run("rules/agent_memory_rule")
        assert "Memory Retention Policy" in read_result
        assert "agent/test-bot" in read_result
        assert "Always prioritize verified" in read_result

        # Check in list tool
        list_result = tools["okf_list_concepts"]._run("type:Policy")
        assert "agent_memory_rule" in list_result

    def test_sub_second_verify_concept(self, v02_bundle: Path):
        """Verifying an existing concept takes <10ms and upgrades trust tier."""
        toolkit = OKFToolkit.from_bundle(v02_bundle)
        tools = {t.name: t for t in toolkit.get_tools(include_write=True)}

        # Write draft concept
        tools["okf_write_concept"]._run(
            concept_id="playbooks/deploy_guide",
            type="Playbook",
            body="Run deploy steps.",
        )
        # Check initial trust tier
        read_initial = tools["okf_read_concept"]._run("playbooks/deploy_guide")
        assert "[? unverified]" in read_initial

        # Verify with human actor
        t0 = time.perf_counter()
        verify_result = tools["okf_verify_concept"]._run(
            concept_id="playbooks/deploy_guide",
            actor="human:lead_dev",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"Verify took {elapsed_ms:.1f}ms — expected <100ms"
        assert "human-reviewed" in verify_result

        # Verify read reflects new tier
        read_verified = tools["okf_read_concept"]._run("playbooks/deploy_guide")
        assert "[✓ human-reviewed]" in read_verified
