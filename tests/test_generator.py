"""Tests for OKFBundleGenerator using a mock LLM."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from langchain_okf.generator import OKFBundleGenerator, _TextFileLoader

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_BUNDLE = FIXTURES / "sample_bundle"


def _make_mock_llm(concepts: list[dict]) -> MagicMock:
    """Create a mock LLM that returns a fixed JSON array of concepts."""
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=json.dumps(concepts))
    return mock


MOCK_CONCEPTS = [
    {
        "type": "Table",
        "title": "Users",
        "description": "One row per registered user.",
        "tags": ["core", "identity"],
        "category": "tables",
        "filename": "users",
        "body": "# Schema\n\n| Column | Type |\n|--------|------|\n| user_id | STRING |",
    },
    {
        "type": "Metric",
        "title": "Daily Active Users",
        "description": "Count of distinct users active in a 24h window.",
        "tags": ["engagement", "kpi"],
        "category": "metrics",
        "filename": "dau",
        "body": "# Definition\n\nDAU counts distinct user_id values from "
        "the [users table](/tables/users.md) with activity in the last 24 hours.",
    },
]


class TestTextFileLoader:
    def test_loads_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        loader = _TextFileLoader(f)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "Hello world"


class TestOKFBundleGenerator:
    def test_generate_creates_bundle(self, tmp_path):
        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_documents([Document(page_content="Some source text")])

        result = gen.generate()

        assert result == output
        assert (output / "tables" / "users.md").exists()
        assert (output / "metrics" / "dau.md").exists()
        assert (output / "index.md").exists()

    def test_generated_files_have_frontmatter(self, tmp_path):
        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_documents([Document(page_content="Some source text")])
        gen.generate()

        content = (output / "tables" / "users.md").read_text()
        assert content.startswith("---")
        assert "type: Table" in content
        assert "title: Users" in content
        assert "# Schema" in content

    def test_generated_files_have_tags(self, tmp_path):
        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_documents([Document(page_content="Some source text")])
        gen.generate()

        content = (output / "tables" / "users.md").read_text()
        assert "core" in content
        assert "identity" in content

    def test_index_files_generated(self, tmp_path):
        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_documents([Document(page_content="text")])
        gen.generate()

        root_index = (output / "index.md").read_text()
        assert "Users" in root_index
        assert "Daily Active Users" in root_index

        tables_index = (output / "tables" / "index.md").read_text()
        assert "users.md" in tables_index

    def test_add_source_file(self, tmp_path):
        source_file = tmp_path / "notes.txt"
        source_file.write_text("Our users table has columns: user_id, email.")

        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_source(source_file)
        gen.generate()

        assert llm.invoke.called
        assert (output / "tables" / "users.md").exists()

    def test_add_sources_multiple(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("Source A")
        f2.write_text("Source B")

        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_sources([f1, f2])
        gen.generate()

        assert llm.invoke.called

    def test_chaining(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("data")

        output = tmp_path / "bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        result = gen.add_source(f).add_documents([Document(page_content="more data")])
        assert result is gen

    def test_merge_deduplicates(self, tmp_path):
        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)

        gen._merge_concepts(MOCK_CONCEPTS)
        gen._merge_concepts(MOCK_CONCEPTS)  # same again

        assert len(gen._concepts) == 2

    def test_source_not_found(self, tmp_path):
        output = tmp_path / "bundle"
        llm = _make_mock_llm([])
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_source("/nonexistent/file.txt")

        with pytest.raises(FileNotFoundError):
            gen.generate()

    def test_bundle_is_loadable(self, tmp_path):
        """Generated bundle can be loaded by OKFBundleLoader."""
        from langchain_okf import OKFBundleLoader

        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_documents([Document(page_content="text")])
        bundle_path = gen.generate()

        loader = OKFBundleLoader(bundle_path)
        docs = loader.load()
        assert len(docs) == 2
        types = {d.metadata["type"] for d in docs}
        assert "Table" in types
        assert "Metric" in types

    def test_bundle_graph_has_cross_links(self, tmp_path):
        """Generated cross-links are detected by OKFKnowledgeGraph."""
        from langchain_okf import OKFKnowledgeGraph

        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output)
        gen.add_documents([Document(page_content="text")])
        bundle_path = gen.generate()

        graph = OKFKnowledgeGraph.from_bundle(bundle_path)
        assert "tables/users" in graph.concept_ids
        neighbors = graph.neighbors("metrics/dau")
        assert "tables/users" in neighbors

    def test_parse_llm_json_with_markdown_fences(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=output)
        result = gen._parse_llm_json('```json\n[{"type": "X", "title": "Y"}]\n```')
        assert len(result) == 1
        assert result[0]["type"] == "X"

    def test_parse_llm_json_raw(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=output)
        result = gen._parse_llm_json('[{"type": "A", "title": "Alpha"}]')
        assert result[0]["type"] == "A"
        assert result[0]["title"] == "Alpha"

    def test_parse_llm_json_invalid(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=output)
        result = gen._parse_llm_json("not json at all")
        assert result == []

    def test_verbose_false_no_output(self, tmp_path, capsys):
        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output, verbose=False)
        gen.add_documents([Document(page_content="text")])
        gen.generate()

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_custom_logger(self, tmp_path):
        messages: list[str] = []
        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output, logger=messages.append)
        gen.add_documents([Document(page_content="text")])
        gen.generate()

        assert len(messages) > 0
        assert any("Ingesting" in m for m in messages)
        assert any("Backend:" in m for m in messages)
        assert any("Done!" in m for m in messages)

    def test_smart_chunk_preserves_paragraphs(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=output)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = gen._smart_chunk(text, chunk_size=30)
        for chunk in chunks:
            assert not chunk.startswith("\n")

    def test_smart_chunk_single_chunk(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=output)
        text = "Short text."
        chunks = gen._smart_chunk(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_smart_chunk_merges_tiny_tail(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(
            llm=_make_mock_llm([]), output_dir=output, verbose=False
        )
        text = ("A" * 500) + "\n\n" + ("B" * 500) + "\n\n" + "tiny"
        chunks = gen._smart_chunk(text, chunk_size=600)
        assert all(len(c) >= 200 for c in chunks)

    def test_docling_init_graceful_without_docling(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(
            llm=_make_mock_llm([]), output_dir=output, verbose=False
        )
        assert gen._docling_converter is None or gen._docling_converter is not None

    def test_domain_preset(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(
            llm=_make_mock_llm(MOCK_CONCEPTS),
            output_dir=output,
            domain="research",
            verbose=False,
        )
        prefix = gen._domain_prefix()
        assert "RESEARCH" in prefix
        assert "findings" in prefix

    def test_domain_custom(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(
            llm=_make_mock_llm(MOCK_CONCEPTS),
            output_dir=output,
            domain="automotive supply chain",
            verbose=False,
        )
        prefix = gen._domain_prefix()
        assert "automotive supply chain" in prefix

    def test_domain_none(self, tmp_path):
        output = tmp_path / "bundle"
        gen = OKFBundleGenerator(
            llm=_make_mock_llm([]), output_dir=output, verbose=False
        )
        assert gen._domain_prefix() == ""

    def test_enrichment_prompt_caps_at_50(self, tmp_path):
        output = tmp_path / "bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output, verbose=False)
        gen._concepts = [
            {"category": "c", "filename": f"f{i}", "type": "T", "title": f"T{i}"}
            for i in range(100)
        ]
        gen._concept_keys = {f"c/f{i}": i for i in range(100)}
        # Enrichment prompt is only used for incremental updates.
        gen._existing_concept_count = 100
        gen.add_documents([Document(page_content="test")])
        gen.generate()
        call_args = llm.invoke.call_args[0][0][0].content
        assert "showing 50 of 100" in call_args

    def test_generate_with_domain(self, tmp_path):
        output = tmp_path / "bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(
            llm=llm, output_dir=output, domain="data", verbose=False
        )
        gen.add_documents([Document(page_content="text")])
        gen.generate()
        call_args = llm.invoke.call_args[0][0][0].content
        assert "DATA AND ENGINEERING" in call_args

    def test_incremental_auto_detect(self, tmp_path):
        """Second generate() on same dir loads existing concepts."""
        output = tmp_path / "test_bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)

        gen1 = OKFBundleGenerator(llm=llm, output_dir=output, verbose=False)
        gen1.add_documents([Document(page_content="first run")])
        gen1.generate()
        assert (output / "tables" / "users.md").exists()

        new_concepts = [
            {
                "type": "API",
                "title": "Auth API",
                "description": "Authentication endpoint.",
                "tags": ["auth"],
                "category": "apis",
                "filename": "auth_api",
                "body": "# Endpoint\n\nPOST /auth/login",
            }
        ]
        llm2 = _make_mock_llm(new_concepts)
        gen2 = OKFBundleGenerator(llm=llm2, output_dir=output, verbose=False)
        gen2.add_documents([Document(page_content="second run")])
        gen2.generate()

        assert (output / "tables" / "users.md").exists()
        assert (output / "apis" / "auth_api.md").exists()

    def test_incremental_preserves_existing(self, tmp_path):
        """Incremental run preserves concepts from first run."""
        from langchain_okf import OKFBundleLoader

        output = tmp_path / "bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen1 = OKFBundleGenerator(llm=llm, output_dir=output, verbose=False)
        gen1.add_documents([Document(page_content="text")])
        gen1.generate()

        llm2 = _make_mock_llm([])
        gen2 = OKFBundleGenerator(llm=llm2, output_dir=output, verbose=False)
        gen2.add_documents([Document(page_content="more")])
        gen2.generate()

        loader = OKFBundleLoader(output)
        docs = loader.load()
        assert len(docs) == 2

    def test_fresh_detect_on_empty_dir(self, tmp_path):
        output = tmp_path / "bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS)
        gen = OKFBundleGenerator(llm=llm, output_dir=output, verbose=False)
        count = gen._load_existing_bundle()
        assert count == 0

    def test_retry_on_llm_failure(self, tmp_path):
        output = tmp_path / "bundle"
        mock = MagicMock()
        mock.invoke.side_effect = [
            Exception("API error"),
            AIMessage(content=json.dumps(MOCK_CONCEPTS)),
        ]
        gen = OKFBundleGenerator(llm=mock, output_dir=output, verbose=False)
        result = gen._extract_with_retry("some content")
        assert len(result) == 2
        assert mock.invoke.call_count == 2

    def test_retry_exhausted(self, tmp_path):
        output = tmp_path / "bundle"
        mock = MagicMock()
        mock.invoke.side_effect = Exception("always fails")
        gen = OKFBundleGenerator(llm=mock, output_dir=output, verbose=False)
        result = gen._extract_with_retry("content", max_retries=1)
        assert result == []
        assert mock.invoke.call_count == 2

    def test_summarize_error_collapses_multiline(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        exc = ValueError(
            "Response validation failed: 6 validation errors\n"
            "body.choices\n  Field required\nbody.id\n  Field required\n"
            "'code': 502"
        )
        summary = gen._summarize_error(exc)
        assert "\n" not in summary
        assert "ValueError" in summary
        assert "HTTP 502" in summary
        assert len(summary) <= 200

    def test_is_retryable_detects_transient(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        assert gen._is_retryable_error(Exception("Upstream capacity, code 502"))
        assert gen._is_retryable_error(Exception("429 rate limit exceeded"))
        assert gen._is_retryable_error(Exception("Internal Server Error"))
        assert not gen._is_retryable_error(Exception("invalid api key"))

    def test_transient_error_backs_off(self, tmp_path, monkeypatch):
        sleeps = []
        monkeypatch.setattr(
            "langchain_okf.generator.time.sleep", lambda s: sleeps.append(s)
        )
        mock = MagicMock()
        mock.invoke.side_effect = [
            Exception("502 Upstream capacity"),
            AIMessage(content=json.dumps(MOCK_CONCEPTS)),
        ]
        gen = OKFBundleGenerator(llm=mock, output_dir=tmp_path / "b", verbose=False)
        result = gen._extract_with_retry("content")
        assert len(result) == 2
        assert len(sleeps) == 1  # backed off once before the successful retry

    def test_nontransient_error_no_backoff(self, tmp_path, monkeypatch):
        sleeps = []
        monkeypatch.setattr(
            "langchain_okf.generator.time.sleep", lambda s: sleeps.append(s)
        )
        mock = MagicMock()
        mock.invoke.side_effect = [
            Exception("bad request: malformed input"),
            AIMessage(content=json.dumps(MOCK_CONCEPTS)),
        ]
        gen = OKFBundleGenerator(llm=mock, output_dir=tmp_path / "b", verbose=False)
        result = gen._extract_with_retry("content")
        assert len(result) == 2
        assert sleeps == []  # no wait for a non-transient error


class TestModelAwareChunking:
    def test_resolve_chunk_size_default(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        assert gen._resolve_chunk_size(None) == 4000

    def test_resolve_chunk_size_instance(self, tmp_path):
        gen = OKFBundleGenerator(
            llm=_make_mock_llm([]), output_dir=tmp_path / "b", chunk_size=2500
        )
        assert gen._resolve_chunk_size(None) == 2500

    def test_resolve_chunk_size_call_override_wins(self, tmp_path):
        gen = OKFBundleGenerator(
            llm=_make_mock_llm([]), output_dir=tmp_path / "b", chunk_size=2500
        )
        assert gen._resolve_chunk_size(6000) == 6000

    def test_normalize_fills_missing_keys(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        result = gen._parse_llm_json('[{"title": "Only Title"}]')
        assert len(result) == 1
        c = result[0]
        assert c["category"] == "general"
        assert c["filename"] == "only_title"
        assert c["type"] == "Concept"
        assert c["title"] == "Only Title"

    def test_normalize_drops_empty_concepts(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        result = gen._parse_llm_json('[{"tags": ["x"]}, {"title": "Keep"}]')
        assert len(result) == 1
        assert result[0]["title"] == "Keep"

    def test_missing_category_does_not_crash_enrichment(self, tmp_path):
        # A stored concept missing 'category' must not crash the next chunk.
        gen = OKFBundleGenerator(
            llm=_make_mock_llm(MOCK_CONCEPTS),
            output_dir=tmp_path / "b",
            verbose=False,
        )
        gen._concepts.append({"title": "orphan", "body": "x"})
        # Should build the enrichment prompt and return concepts, not raise.
        result = gen._extract_with_retry("more content")
        assert isinstance(result, list)

    def test_salvage_truncated_array(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        # Valid first object, then a truncated second one — both are salvaged!
        text = '[{"type": "A", "title": "First"}, {"type": "B", "title": "Trunca'
        result = gen._parse_llm_json(text)
        assert len(result) == 2
        assert result[0]["title"] == "First"
        assert result[1]["title"] == "Trunca"

    def test_salvage_truncated_object(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        # Response cut off inside an object, e.g. after array field
        text = '[{"type": "Mechanism", "title": "R-SWA", "description": "Desc", "tags": ["attention", "decoding",'
        result = gen._parse_llm_json(text)
        assert len(result) == 1
        assert result[0]["title"] == "R-SWA"
        assert result[0]["type"] == "Mechanism"
        assert result[0]["tags"] == ["attention", "decoding"]

    def test_salvage_newlines_in_strings(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        # Literal newline inside a string value is invalid strict JSON.
        text = '[{"type": "A", "body": "line one\nline two"}]'
        result = gen._parse_llm_json(text)
        assert len(result) == 1
        assert "line" in result[0]["body"]

    def test_content_to_str_from_blocks(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        blocks = [
            {"type": "text", "text": '[{"type": "A", '},
            {"type": "text", "text": '"title": "Split"}]'},
        ]
        result = gen._parse_llm_json(blocks)
        assert len(result) == 1
        assert result[0]["title"] == "Split"

    def test_split_giant_paragraph(self, tmp_path):
        gen = OKFBundleGenerator(llm=_make_mock_llm([]), output_dir=tmp_path / "b")
        huge = "x" * 5000  # one paragraph, no blank lines
        chunks = gen._smart_chunk(huge, chunk_size=1000)
        assert len(chunks) == 5
        assert all(len(c) <= 1000 for c in chunks)

    def test_truncation_triggers_split(self, tmp_path):
        # First call reports truncation; the two half-chunk calls succeed.
        mock = MagicMock()
        truncated = AIMessage(
            content=json.dumps(MOCK_CONCEPTS[:1]),
            response_metadata={"finish_reason": "length"},
        )
        ok = AIMessage(content=json.dumps(MOCK_CONCEPTS[:1]))
        mock.invoke.side_effect = [truncated, ok, ok]
        gen = OKFBundleGenerator(llm=mock, output_dir=tmp_path / "b", verbose=False)
        content = "para one.\n\n" + ("word " * 500)
        result = gen._extract_with_retry(content)
        assert mock.invoke.call_count == 3
        assert len(result) == 2

    def test_no_truncation_no_split(self, tmp_path):
        mock = MagicMock()
        mock.invoke.return_value = AIMessage(content=json.dumps(MOCK_CONCEPTS))
        gen = OKFBundleGenerator(llm=mock, output_dir=tmp_path / "b", verbose=False)
        result = gen._extract_with_retry("small content")
        assert mock.invoke.call_count == 1
        assert len(result) == 2


class TestPathSafety:
    def test_sanitize_removes_slashes(self, tmp_path):
        from langchain_okf.generator import _sanitize_segment

        assert _sanitize_segment("research/r_swa", "x") == "research_r_swa"

    def test_sanitize_strips_md_suffix(self, tmp_path):
        from langchain_okf.generator import _sanitize_segment

        assert _sanitize_segment("r_swa.md", "x") == "r_swa"

    def test_sanitize_lowercases(self, tmp_path):
        from langchain_okf.generator import _sanitize_segment

        assert _sanitize_segment("Research Papers", "x") == "research_papers"

    def test_sanitize_fallback_on_empty(self, tmp_path):
        from langchain_okf.generator import _sanitize_segment

        assert _sanitize_segment("", "general") == "general"
        assert _sanitize_segment(None, "general") == "general"
        assert _sanitize_segment("///", "general") == "general"

    def test_category_with_slash_writes_flat(self, tmp_path):
        concepts = [
            {
                "type": "Finding",
                "title": "R-SWA",
                "category": "research/r_swa",
                "filename": "r_swa.md",
                "body": "# X\n\nbody",
            }
        ]
        gen = OKFBundleGenerator(
            llm=_make_mock_llm(concepts),
            output_dir=tmp_path / "b",
            verbose=False,
        )
        gen.add_documents([Document(page_content="text")])
        gen.generate()
        # Flat category dir, no nested research/r_swa/ tree.
        assert (tmp_path / "b" / "research_r_swa" / "r_swa.md").exists()
        assert not (tmp_path / "b" / "research" / "r_swa").exists()

    def test_unique_path_avoids_dir_collision(self, tmp_path):
        cat_dir = tmp_path / "cat"
        (cat_dir / "r_swa").mkdir(parents=True)  # a directory named r_swa
        used: set = set()
        p = OKFBundleGenerator._unique_path(cat_dir, "r_swa", used)
        # r_swa.md is free (only r_swa/ dir exists), so it should be used.
        assert p == cat_dir / "r_swa.md"

    def test_unique_path_disambiguates_same_run(self, tmp_path):
        cat_dir = tmp_path / "cat"
        cat_dir.mkdir()
        used: set = set()
        p1 = OKFBundleGenerator._unique_path(cat_dir, "dup", used)
        p2 = OKFBundleGenerator._unique_path(cat_dir, "dup", used)
        assert p1 == cat_dir / "dup.md"
        assert p2 == cat_dir / "dup_2.md"


class TestValidateBundle:
    def test_valid_bundle(self):
        from langchain_okf import validate_bundle

        issues = validate_bundle(SAMPLE_BUNDLE)
        assert issues == []

    def test_missing_dir(self, tmp_path):
        from langchain_okf import validate_bundle

        issues = validate_bundle(tmp_path / "nonexistent")
        assert len(issues) == 1
        assert "does not exist" in issues[0]

    def test_empty_dir(self, tmp_path):
        from langchain_okf import validate_bundle

        issues = validate_bundle(tmp_path)
        assert any("No .md files" in i for i in issues)

    def test_missing_type(self, tmp_path):
        from langchain_okf import validate_bundle

        md = tmp_path / "bad.md"
        md.write_text("---\ntitle: No Type\n---\nBody")
        issues = validate_bundle(tmp_path)
        assert any("type" in i for i in issues)

    def test_missing_frontmatter(self, tmp_path):
        from langchain_okf import validate_bundle

        md = tmp_path / "bad.md"
        md.write_text("# Just markdown\nNo frontmatter here.")
        issues = validate_bundle(tmp_path)
        assert any("missing" in i.lower() for i in issues)


class TestLogoAndAssetCleanup:
    def test_is_likely_decorative_or_logo_small(self, tmp_path):
        from langchain_okf.generator import OKFBundleGenerator

        # Create a small file (under 3KB)
        small_img = tmp_path / "logo.png"
        small_img.write_bytes(b"\x00" * 1024)  # 1KB

        gen = OKFBundleGenerator(
            llm=None, output_dir=tmp_path / "bundle", verbose=False
        )
        assert gen._is_likely_decorative_or_logo(small_img) is True

    def test_is_likely_decorative_or_logo_large(self, tmp_path):
        from langchain_okf.generator import OKFBundleGenerator

        # Create a larger file
        large_img = tmp_path / "photo.png"
        large_img.write_bytes(b"\x00" * (10 * 1024))  # 10KB

        gen = OKFBundleGenerator(
            llm=None, output_dir=tmp_path / "bundle", verbose=False
        )
        # Without PIL, defaults to False for files >= 3KB
        assert gen._is_likely_decorative_or_logo(large_img) is False

    def test_write_bundle_cleans_unused_assets(self, tmp_path):
        from langchain_okf.generator import OKFBundleGenerator

        bundle_dir = tmp_path / "bundle"
        gen = OKFBundleGenerator(llm=None, output_dir=bundle_dir, verbose=False)

        # Add a concept referencing one image
        gen._concepts = [
            {
                "type": "Concept",
                "title": "Main",
                "category": "general",
                "filename": "main",
                "body": "Referencing first image.",
                "image_source": str(tmp_path / "used.png"),
            }
        ]

        # Populate assets directory with both a used and an unused asset
        assets_dir = bundle_dir / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "used.png").write_bytes(b"\x00" * 100)
        (assets_dir / "unused.png").write_bytes(b"\x00" * 100)

        gen._write_bundle()

        assert (assets_dir / "used.png").exists()
        assert not (assets_dir / "unused.png").exists()


class TestCachingAndConcurrency:
    def test_caching_skips_llm(self, tmp_path):
        from langchain_okf.generator import OKFBundleGenerator

        output = tmp_path / "bundle"
        llm = MagicMock()
        gen = OKFBundleGenerator(llm=llm, output_dir=output, verbose=False)

        # Populate cache dynamically with the expected chunk text
        from langchain_okf.generator import _sha256_text

        chunk_hash = _sha256_text("[Source: unknown]\nParagraph content.")
        gen._cache = {
            "version": 1,
            "chunks": {
                chunk_hash: [
                    {
                        "type": "Concept",
                        "title": "Cached Title",
                        "category": "general",
                        "filename": "cached",
                        "body": "Cached body",
                    }
                ]
            },
            "images": {},
        }

        gen.add_documents([Document(page_content="Paragraph content.")])
        gen.generate()

        # Verify LLM was never called
        assert llm.invoke.call_count == 0
        assert len(gen._concepts) == 1
        assert gen._concepts[0]["title"] == "Cached Title"
        assert (output / "general" / "cached.md").exists()
        assert (output / ".okf_cache.json").exists()

    def test_concurrency_works(self, tmp_path):
        from langchain_okf.generator import OKFBundleGenerator

        output = tmp_path / "bundle"
        llm = _make_mock_llm(MOCK_CONCEPTS[:1])

        # Use a high concurrency setting
        gen = OKFBundleGenerator(
            llm=llm, output_dir=output, max_concurrency=4, verbose=False
        )

        # Add multiple documents to split into separate chunks
        gen.add_documents(
            [
                Document(page_content="Content one."),
                Document(page_content="Content two."),
                Document(page_content="Content three."),
            ]
        )

        # Force a small chunk size so they split into multiple chunks
        gen.generate(chunk_size=20)

        # Verify it created a cache and extracted concepts
        assert (output / ".okf_cache.json").exists()
        assert len(gen._concepts) > 0
