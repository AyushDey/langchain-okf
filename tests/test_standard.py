"""Standard LangChain compliance integration tests using langchain-tests."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_tests.integration_tests.retrievers import RetrieversIntegrationTests
from langchain_tests.unit_tests.tools import ToolsUnitTests

from langchain_okf import OKFKnowledgeGraph, OKFRetriever
from langchain_okf.toolkit import (
    OKFListConceptsTool,
    OKFReadConceptTool,
    OKFRelatedConceptsTool,
    OKFSearchTool,
    OKFVerifyConceptTool,
    OKFWriteConceptTool,
)


class FakeEmbeddings:
    """Mock embeddings for standard compliance testing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 10 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 10

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class TestOKFRetrieverStandard(RetrieversIntegrationTests):
    """Standard LangChain compliance test suite for OKFRetriever."""

    @property
    def retriever_constructor(self) -> type[OKFRetriever]:
        return OKFRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        vs = InMemoryVectorStore(embedding=FakeEmbeddings())
        docs = [
            Document(
                page_content="Concept 1: Core OKF architecture.",
                metadata={"title": "Concept 1", "concept_id": "c1"},
            ),
            Document(
                page_content="Concept 2: Knowledge graph retrieval.",
                metadata={"title": "Concept 2", "concept_id": "c2"},
            ),
            Document(
                page_content="Concept 3: Agent memory write locks.",
                metadata={"title": "Concept 3", "concept_id": "c3"},
            ),
            Document(
                page_content="Concept 4: Trust verification tiers.",
                metadata={"title": "Concept 4", "concept_id": "c4"},
            ),
        ]
        vs.add_documents(docs)
        graph = OKFKnowledgeGraph(concepts=[])
        return {
            "vectorstore": vs,
            "graph": graph,
            "expand_depth": 0,
            "min_trust_tier": "unverified",
            "filter_status": None,
            "exclude_stale": False,
        }

    @property
    def retriever_query_example(self) -> str:
        return "Concept"


class TestOKFSearchToolStandard(ToolsUnitTests):
    """Standard test suite for OKFSearchTool."""

    @property
    def tool_constructor(self) -> type[OKFSearchTool]:
        return OKFSearchTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"graph": OKFKnowledgeGraph(concepts=[])}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "authentication"}


class TestOKFReadConceptToolStandard(ToolsUnitTests):
    """Standard test suite for OKFReadConceptTool."""

    @property
    def tool_constructor(self) -> type[OKFReadConceptTool]:
        return OKFReadConceptTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"graph": OKFKnowledgeGraph(concepts=[])}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"concept_id": "auth/tokens"}


class TestOKFRelatedConceptsToolStandard(ToolsUnitTests):
    """Standard test suite for OKFRelatedConceptsTool."""

    @property
    def tool_constructor(self) -> type[OKFRelatedConceptsTool]:
        return OKFRelatedConceptsTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"graph": OKFKnowledgeGraph(concepts=[])}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"concept_id": "auth/tokens"}


class TestOKFListConceptsToolStandard(ToolsUnitTests):
    """Standard test suite for OKFListConceptsTool."""

    @property
    def tool_constructor(self) -> type[OKFListConceptsTool]:
        return OKFListConceptsTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"graph": OKFKnowledgeGraph(concepts=[])}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"type": "policy"}


class TestOKFWriteConceptToolStandard(ToolsUnitTests):
    """Standard test suite for OKFWriteConceptTool."""

    @property
    def tool_constructor(self) -> type[OKFWriteConceptTool]:
        return OKFWriteConceptTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {
            "graph": OKFKnowledgeGraph(concepts=[]),
            "bundle_path": "tests/fixtures/sample_bundle",
            "actor": "agent/test",
        }

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "concept_id": "testing/unit_test_concept",
            "type": "concept",
            "title": "Unit Test Concept",
            "body": "Test body",
        }


class TestOKFVerifyConceptToolStandard(ToolsUnitTests):
    """Standard test suite for OKFVerifyConceptTool."""

    @property
    def tool_constructor(self) -> type[OKFVerifyConceptTool]:
        return OKFVerifyConceptTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {
            "graph": OKFKnowledgeGraph(concepts=[]),
            "bundle_path": "tests/fixtures/sample_bundle",
            "default_actor": "agent/test",
        }

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "concept_id": "concepts/core/sample",
            "trust_tier": "machine-confirmed",
        }
