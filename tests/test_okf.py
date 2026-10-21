"""Tests for OKF parser, document loader, graph, and toolkit."""

from pathlib import Path

import pytest

from langchain_okf._parser import (
    OKFConcept,
    OKFParseError,
    is_reserved,
    parse_bundle,
    parse_frontmatter,
)
from langchain_okf.document_loader import OKFBundleLoader
from langchain_okf.graph import OKFKnowledgeGraph
from langchain_okf.toolkit import (
    OKFListConceptsTool,
    OKFReadConceptTool,
    OKFRelatedConceptsTool,
    OKFSearchTool,
    OKFToolkit,
)

FIXTURES = Path(__file__).parent / "fixtures" / "sample_bundle"


# ── Parser tests ──


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        text = "---\ntype: Table\ntitle: Orders\n---\n\n# Schema\nSome content."
        fm, body = parse_frontmatter(text)
        assert fm["type"] == "Table"
        assert fm["title"] == "Orders"
        assert "# Schema" in body

    def test_no_frontmatter(self):
        text = "# Just a heading\nSome content."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_unterminated_frontmatter(self):
        text = "---\ntype: Table\n# Missing closing delimiter"
        with pytest.raises(OKFParseError, match="Unterminated"):
            parse_frontmatter(text)

    def test_invalid_yaml(self):
        text = "---\n: invalid: yaml: [[\n---\nBody"
        with pytest.raises(OKFParseError, match="Invalid YAML"):
            parse_frontmatter(text)

    def test_tags_as_list(self):
        text = "---\ntype: X\ntags: [a, b, c]\n---\nBody"
        fm, _ = parse_frontmatter(text)
        assert fm["tags"] == ["a", "b", "c"]

    def test_extra_frontmatter_keys_preserved(self):
        text = "---\ntype: X\ncustom_key: custom_value\n---\nBody"
        fm, _ = parse_frontmatter(text)
        assert fm["custom_key"] == "custom_value"


class TestOKFConcept:
    def test_cross_links_absolute(self):
        concept = OKFConcept(
            concept_id="tables/orders",
            body="See [customers](/tables/customers.md) for details.",
        )
        assert concept.cross_links() == ["tables/customers"]

    def test_cross_links_relative(self):
        concept = OKFConcept(
            concept_id="tables/orders",
            body="See [customers](./customers.md) for the join.",
        )
        assert concept.cross_links() == ["customers"]

    def test_no_cross_links(self):
        concept = OKFConcept(concept_id="x", body="No links here.")
        assert concept.cross_links() == []

    def test_properties(self):
        concept = OKFConcept(
            concept_id="test",
            frontmatter={
                "type": "Table",
                "title": "My Table",
                "description": "A table.",
                "tags": ["a", "b"],
                "resource": "https://example.com",
                "timestamp": "2026-01-01T00:00:00Z",
            },
        )
        assert concept.type == "Table"
        assert concept.title == "My Table"
        assert concept.tags == ["a", "b"]
        assert concept.resource == "https://example.com"


class TestIsReserved:
    def test_index_is_reserved(self):
        assert is_reserved(Path("some/dir/index.md"))

    def test_log_is_reserved(self):
        assert is_reserved(Path("log.md"))

    def test_concept_not_reserved(self):
        assert not is_reserved(Path("tables/orders.md"))


class TestParseBundle:
    def test_loads_all_concepts(self):
        concepts = parse_bundle(FIXTURES)
        ids = {c.concept_id for c in concepts}
        assert "datasets/sales" in ids
        assert "tables/orders" in ids
        assert "tables/customers" in ids
        assert "metrics/weekly_active_users" in ids

    def test_skips_reserved_files(self):
        concepts = parse_bundle(FIXTURES)
        ids = {c.concept_id for c in concepts}
        assert "index" not in ids

    def test_frontmatter_parsed(self):
        concepts = parse_bundle(FIXTURES)
        orders = next(c for c in concepts if c.concept_id == "tables/orders")
        assert orders.type == "BigQuery Table"
        assert orders.title == "Orders"
        assert "sales" in orders.tags


# ── Document Loader tests ──


class TestOKFBundleLoader:
    def test_load_all(self):
        loader = OKFBundleLoader(FIXTURES)
        docs = loader.load()
        assert len(docs) == 4
        concept_ids = {d.metadata["concept_id"] for d in docs}
        assert "tables/orders" in concept_ids

    def test_metadata_mapping(self):
        loader = OKFBundleLoader(FIXTURES)
        docs = loader.load()
        orders = next(d for d in docs if d.metadata["concept_id"] == "tables/orders")
        assert orders.metadata["type"] == "BigQuery Table"
        assert orders.metadata["title"] == "Orders"
        assert "sales" in orders.metadata["tags"]
        assert "cross_links" in orders.metadata

    def test_include_types_filter(self):
        loader = OKFBundleLoader(FIXTURES, include_types=["Metric"])
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata["type"] == "Metric"

    def test_exclude_types_filter(self):
        loader = OKFBundleLoader(FIXTURES, exclude_types=["BigQuery Table"])
        docs = loader.load()
        types = {d.metadata["type"] for d in docs}
        assert "BigQuery Table" not in types

    def test_include_tags_filter(self):
        loader = OKFBundleLoader(FIXTURES, include_tags=["kpi"])
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata["concept_id"] == "metrics/weekly_active_users"

    def test_lazy_load(self):
        loader = OKFBundleLoader(FIXTURES)
        docs = list(loader.lazy_load())
        assert len(docs) == 4

    def test_extra_frontmatter_in_metadata(self):
        loader = OKFBundleLoader(FIXTURES)
        docs = loader.load()
        orders = next(d for d in docs if d.metadata["concept_id"] == "tables/orders")
        assert "resource" in orders.metadata


# ── Graph tests ──


class TestOKFKnowledgeGraph:
    @pytest.fixture
    def graph(self):
        return OKFKnowledgeGraph.from_bundle(FIXTURES)

    def test_concept_count(self, graph):
        assert len(graph.concept_ids) == 4

    def test_types(self, graph):
        assert "BigQuery Table" in graph.types
        assert "Metric" in graph.types
        assert "BigQuery Dataset" in graph.types

    def test_neighbors(self, graph):
        neighbors = graph.neighbors("tables/orders")
        assert "tables/customers" in neighbors
        assert "datasets/sales" in neighbors

    def test_backlinks(self, graph):
        backlinks = graph.backlinks("tables/customers")
        assert "tables/orders" in backlinks

    def test_concepts_by_type(self, graph):
        tables = graph.concepts_by_type("BigQuery Table")
        assert "tables/orders" in tables
        assert "tables/customers" in tables
        assert len(tables) == 2

    def test_concepts_by_tag(self, graph):
        sales = graph.concepts_by_tag("sales")
        assert "tables/orders" in sales
        assert "datasets/sales" in sales

    def test_subgraph(self, graph):
        sub = graph.subgraph("metrics/weekly_active_users", depth=2)
        assert "metrics/weekly_active_users" in sub
        assert "tables/orders" in sub
        # depth 2: orders -> customers
        assert "tables/customers" in sub

    def test_edge_count(self, graph):
        assert graph.edge_count() > 0

    def test_summary(self, graph):
        s = graph.summary()
        assert "4 concepts" in s
        assert "cross-links" in s


# ── Toolkit tests ──


class TestOKFToolkit:
    @pytest.fixture
    def toolkit(self):
        return OKFToolkit.from_bundle(FIXTURES)

    def test_get_tools_count(self, toolkit):
        tools = toolkit.get_tools()
        assert len(tools) == 4

    def test_list_all(self, toolkit):
        tool = OKFListConceptsTool(graph=toolkit.graph)
        result = tool._run("all")
        assert "tables/orders" in result
        assert "4 concepts" in result

    def test_list_by_type(self, toolkit):
        tool = OKFListConceptsTool(graph=toolkit.graph)
        result = tool._run("type:Metric")
        assert "weekly_active_users" in result
        assert "1 concepts" in result

    def test_list_by_tag(self, toolkit):
        tool = OKFListConceptsTool(graph=toolkit.graph)
        result = tool._run("tag:sales")
        assert "orders" in result

    def test_read_concept(self, toolkit):
        tool = OKFReadConceptTool(graph=toolkit.graph)
        result = tool._run("tables/orders")
        assert "Orders" in result
        assert "Schema" in result
        assert "order_id" in result

    def test_read_concept_not_found(self, toolkit):
        tool = OKFReadConceptTool(graph=toolkit.graph)
        result = tool._run("nonexistent/thing")
        assert "not found" in result

    def test_related_concepts(self, toolkit):
        tool = OKFRelatedConceptsTool(graph=toolkit.graph)
        result = tool._run("tables/orders")
        assert "tables/customers" in result
        assert "datasets/sales" in result

    def test_search(self, toolkit):
        tool = OKFSearchTool(graph=toolkit.graph)
        result = tool._run("orders")
        assert "tables/orders" in result

    def test_search_no_results(self, toolkit):
        tool = OKFSearchTool(graph=toolkit.graph)
        result = tool._run("zzzznonexistent")
        assert "No concepts matching" in result

    @pytest.mark.asyncio
    async def test_list_async(self, toolkit):
        tool = OKFListConceptsTool(graph=toolkit.graph)
        result = await tool._arun("all")
        assert "4 concepts" in result

    @pytest.mark.asyncio
    async def test_read_async(self, toolkit):
        tool = OKFReadConceptTool(graph=toolkit.graph)
        result = await tool._arun("tables/orders")
        assert "Orders" in result

    @pytest.mark.asyncio
    async def test_related_async(self, toolkit):
        tool = OKFRelatedConceptsTool(graph=toolkit.graph)
        result = await tool._arun("tables/orders")
        assert "tables/customers" in result

    @pytest.mark.asyncio
    async def test_search_async(self, toolkit):
        tool = OKFSearchTool(graph=toolkit.graph)
        result = await tool._arun("orders")
        assert "tables/orders" in result
