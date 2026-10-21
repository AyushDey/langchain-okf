"""Integration tests for langchain-okf.

Uses Ollama (local, free) with whatever model is available.
Skipped if Ollama is not running.

To run:
    ollama pull llama3.2       # or any model you have
    uv run pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

_COMMON_MODELS = [
    "gemma4",
    "llama3.2",  # 2.0GB
    "gemma3:1b",  # 815MB
    "phi3:mini",  # 2.0GB
    "qwen2:0.5b",  # 352MB
    "tinyllama",  # 637MB
    "gemma2:2b",  # 1.6GB
    "mistral",  # if user already has it
]

_EMBED_MODELS = [
    "embeddinggemma",
    "nomic-embed-text",  # 274MB
    "all-minilm",  # 45MB
    "mxbai-embed-large",  # 670MB
]


def _get_llm():
    """Try common Ollama models until one works."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        print("SKIP: langchain-ollama not installed")
        return None

    for model in _COMMON_MODELS:
        try:
            probe = ChatOllama(model=model, temperature=0, num_predict=10)
            probe.invoke("say ok")
            print(f"LLM: using {model}")
            return ChatOllama(model=model, temperature=0)
        except Exception as e:
            print(f"LLM: {model} failed: {e}")
            continue
    print("SKIP: no working Ollama LLM model found")
    return None


def _get_embeddings():
    """Try Ollama embedding models until one works."""
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        print("SKIP: langchain-ollama not installed")
        return None

    for model in _EMBED_MODELS + _COMMON_MODELS[:3]:
        try:
            emb = OllamaEmbeddings(model=model)
            emb.embed_query("test")
            print(f"Embeddings: using {model}")
            return emb
        except Exception as e:
            print(f"Embeddings: {model} failed: {e}")
            continue
    print("SKIP: no working Ollama embedding model found")
    return None


LLM = _get_llm()
EMBEDDINGS = _get_embeddings()

SKIP_NO_LLM = LLM is None
SKIP_NO_EMBEDDINGS = EMBEDDINGS is None
SKIP_REASON_LLM = (
    "Ollama not running or no models installed (run: ollama pull llama3.2)"
)
SKIP_REASON_EMB = "No embedding model available (run: ollama pull nomic-embed-text)"


@pytest.mark.skipif(SKIP_NO_LLM, reason=SKIP_REASON_LLM)
class TestGeneratorIntegration:
    def test_generate_from_text(self, tmp_path):
        from langchain_okf import OKFBundleGenerator, OKFBundleLoader, validate_bundle

        gen = OKFBundleGenerator(
            llm=LLM,
            output_dir=tmp_path / "bundle",
            domain="data",
        )
        gen.add_documents(
            [
                Document(
                    page_content=(
                        "The users table has columns: user_id (STRING), "
                        "email (STRING), created_at (TIMESTAMP). "
                        "The orders table has columns: order_id (STRING), "
                        "user_id (STRING, FK to users), total_usd (NUMERIC)."
                    )
                )
            ]
        )
        bundle_path = gen.generate()

        issues = validate_bundle(bundle_path)
        assert issues == [], f"Validation failed: {issues}"

        docs = OKFBundleLoader(bundle_path).load()
        assert len(docs) > 0

    def test_incremental_update(self, tmp_path):
        from langchain_okf import OKFBundleGenerator, OKFBundleLoader

        output = tmp_path / "bundle"

        gen1 = OKFBundleGenerator(llm=LLM, output_dir=output, domain="data")
        gen1.add_documents(
            [Document(page_content="The users table has columns: user_id, email.")]
        )
        gen1.generate()
        first_count = len(OKFBundleLoader(output).load())

        gen2 = OKFBundleGenerator(llm=LLM, output_dir=output, domain="data")
        gen2.add_documents(
            [
                Document(
                    page_content="The products table has columns: product_id, name, price."
                )
            ]
        )
        gen2.generate()
        second_count = len(OKFBundleLoader(output).load())

        assert second_count >= first_count

    def test_generate_from_pdf(self, tmp_path):
        pdf_path = os.environ.get("OKF_TEST_PDF")
        if not pdf_path or not Path(pdf_path).exists():
            pytest.skip("Set OKF_TEST_PDF to a valid PDF path")

        from langchain_okf import OKFBundleGenerator, validate_bundle

        gen = OKFBundleGenerator(llm=LLM, output_dir=tmp_path / "bundle")
        gen.add_source(pdf_path)
        bundle_path = gen.generate()

        issues = validate_bundle(bundle_path)
        assert issues == [], f"Validation failed: {issues}"


@pytest.mark.skipif(
    SKIP_NO_LLM or SKIP_NO_EMBEDDINGS,
    reason=SKIP_REASON_LLM if SKIP_NO_LLM else SKIP_REASON_EMB,
)
class TestRetrieverIntegration:
    def test_retriever_end_to_end(self, tmp_path):
        from langchain_okf import OKFBundleGenerator, OKFRetriever

        gen = OKFBundleGenerator(llm=LLM, output_dir=tmp_path / "bundle")
        gen.add_documents(
            [
                Document(
                    page_content="WAU is weekly active users, counted from the orders table."
                )
            ]
        )
        bundle_path = gen.generate()

        retriever = OKFRetriever.from_bundle(bundle_path, embeddings=EMBEDDINGS)
        docs = retriever.invoke("How is WAU calculated?")
        assert len(docs) > 0

    def test_toolkit_end_to_end(self, tmp_path):
        from langchain_okf import OKFBundleGenerator, OKFToolkit

        gen = OKFBundleGenerator(llm=LLM, output_dir=tmp_path / "bundle")
        gen.add_documents(
            [Document(page_content="The orders table tracks customer purchases.")]
        )
        bundle_path = gen.generate()

        toolkit = OKFToolkit.from_bundle(bundle_path)
        tools = toolkit.get_tools()
        assert len(tools) == 4

        search_result = tools[3].invoke("orders")
        assert "orders" in search_result.lower()
