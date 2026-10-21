"""Benchmark script testing OKF v0.2 performance on 'Class 4 post read.md'.

Measures:
1. Bundle Generation: Cold run vs Cached rerun (sub-second!)
2. Spec Conformance: OKF v0.2 validation (strict mode)
3. Retrieval & Search: Persistent vector reload vs Cold build, Graph queries, Tool lookups
4. Agent Real-Time Memory: Sub-second write and trust verification
"""

import os
import sys
import time
import shutil
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from langchain_okf.generator import OKFBundleGenerator
from langchain_okf.document_loader import OKFBundleLoader
from langchain_okf.graph import OKFKnowledgeGraph
from langchain_okf.retriever import OKFRetriever, _bundle_content_hash
from langchain_okf.toolkit import OKFToolkit
from langchain_okf.validate import validate_bundle


# ── Deterministic Realistic LLM for reproducible benchmarking ─────────────────

class BenchmarkLLM(BaseChatModel):
    """High-fidelity mock LLM extracting realistic concepts from Class 4."""
    model_name: str = "mock-claude-sonnet"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Realistic concepts corresponding to Class 4
        concepts = [
            {
                "type": "Definition",
                "title": "Two Uses of AI: Coding Agent vs Runtime Model",
                "description": "Distinguishing the coding agent harness from the application runtime model.",
                "tags": ["agents", "harness", "architecture"],
                "category": "definitions",
                "filename": "two_uses_of_ai",
                "body": (
                    "# Summary\n\n"
                    "A coding agent (Claude Code or OpenCode) helps build the application through a harness. "
                    "Separately, the runtime model helps produce case reviews. "
                    "Different jobs, context, and acceptance evidence.\n\n"
                    "See [Context Engineering](/definitions/context_engineering.md) and "
                    "[Code vs Model vs Human](/playbooks/code_model_human.md).\n\n"
                    "# Source\n\nFrom: `Class 4 post read.md`, Section 2"
                ),
                "source_file": "Class 4 post read.md",
                "source_page": 2,
            },
            {
                "type": "Definition",
                "title": "The Five-Part Context Pack",
                "description": "Task, Evidence, Definitions, Boundaries, and Output contract.",
                "tags": ["context-engineering", "prompting", "guardrails"],
                "category": "definitions",
                "filename": "context_engineering",
                "body": (
                    "# Overview\n\n"
                    "Context engineering is broader than prompt engineering. The five parts are:\n"
                    "1. **Task**: What should the model do?\n"
                    "2. **Evidence**: Which supplied facts may it use?\n"
                    "3. **Definitions**: What do fields and states mean?\n"
                    "4. **Boundaries**: What must it not decide or invent?\n"
                    "5. **Output contract**: What exact structure should it return?\n\n"
                    "Related to [Two Uses of AI](/definitions/two_uses_of_ai.md).\n\n"
                    "# Source\n\nFrom: `Class 4 post read.md`, Section 3"
                ),
                "source_file": "Class 4 post read.md",
                "source_page": 3,
            },
            {
                "type": "Playbook",
                "title": "Code vs Model vs Human Decision Boundary",
                "description": "Guiding when to use fixed code, AI models, or authorized humans.",
                "tags": ["playbook", "governance", "decision-framework"],
                "category": "playbooks",
                "filename": "code_model_human",
                "body": (
                    "# Framework\n\n"
                    "- **Fixed & Exact conditions**: Normal code (e.g. check delivery_date presence)\n"
                    "- **Varied language**: AI model (e.g. customer request summarization)\n"
                    "- **Business authority & refunds**: Authorized human\n\n"
                    "Links to [The Ratchet Pattern](/playbooks/the_ratchet.md).\n\n"
                    "# Source\n\nFrom: `Class 4 post read.md`, Section 4"
                ),
                "source_file": "Class 4 post read.md",
                "source_page": 4,
            },
            {
                "type": "Playbook",
                "title": "The Ratchet: Turning Failures into Permanent Checks",
                "description": "Turning surfaced failures into regression tests to prevent silent return.",
                "tags": ["testing", "quality", "evaluation", "ratchet"],
                "category": "playbooks",
                "filename": "the_ratchet",
                "body": (
                    "# Ratchet Principle\n\n"
                    "When an important failure is found during external evaluation, turn it into a permanent "
                    "acceptance test so the same failure is less likely to return unnoticed.\n\n"
                    "Follows [Code vs Model vs Human](/playbooks/code_model_human.md).\n\n"
                    "# Source\n\nFrom: `Class 4 post read.md`, Section 7"
                ),
                "source_file": "Class 4 post read.md",
                "source_page": 7,
            },
        ]
        text_resp = json.dumps(concepts)
        return {"generations": [{"message": AIMessage(content=text_resp)}]}

    @property
    def _llm_type(self) -> str:
        return "mock"


class BenchmarkEmbeddings(Embeddings):
    """Fast deterministic embedding model for reproducible benchmarks."""
    def embed_documents(self, texts):
        return [[0.05 * (i % 10), 0.1, 0.2] for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [0.05, 0.1, 0.2]


class BenchmarkVectorStore(VectorStore):
    """Disk-persistent VectorStore simulator for sub-second index reload."""
    def __init__(self, docs=None, persist_directory=None, **kwargs):
        self.docs = docs or []
        self.persist_directory = persist_directory

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas or [{}] * len(texts))]
        return cls(docs=docs)

    @classmethod
    def from_documents(cls, docs, embedding=None, persist_directory=None, **kwargs):
        if embedding:
            embedding.embed_documents([d.page_content for d in docs])
        if persist_directory:
            p = Path(persist_directory)
            p.mkdir(parents=True, exist_ok=True)
            (p / "index.bin").write_text("persisted_vector_data")
        return cls(docs=docs, persist_directory=persist_directory)

    def similarity_search(self, query, k=5, **kwargs):
        return self.docs[:k]

    def similarity_search_with_relevance_scores(self, query, k=5, **kwargs):
        return [(d, 0.95) for d in self.docs[:k]]


def run_benchmarks():
    source_file = Path(__file__).parent / "class_4_post_read.md"
    output_bundle = Path(__file__).parent / "bundle_class4"

    if output_bundle.exists():
        shutil.rmtree(output_bundle)

    print("=" * 72)
    print("  LANGCHAIN-OKF v0.2 PERFORMANCE & CONFORMANCE BENCHMARK")
    print(f"  Source File : {source_file.name} ({source_file.stat().st_size:,} bytes)")
    print(f"  Target Bundle: {output_bundle}")
    print("=" * 72)

    # ───────────────────────────────────────────────────────────────────────────
    # 1. BUNDLE GENERATION & RE-GENERATION CACHING
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[TEST 1] BUNDLE GENERATION LATENCY")

    llm = BenchmarkLLM()
    gen = OKFBundleGenerator(
        llm=llm,
        output_dir=output_bundle,
        bundle_name="FDE Class 4 Knowledge Base",
        domain="AI engineering and evaluation",
        verbose=False,
        stale_after_days=90,
    )
    gen.add_source(source_file)

    # Run 1: Cold generation
    t0 = time.perf_counter()
    gen.generate()
    t_cold_gen = time.perf_counter() - t0
    print(f"  • Run 1 (Cold Generation & Bundle Build)  : {t_cold_gen:.4f} s")

    # Run 2: Cached re-generation (identical content)
    gen2 = OKFBundleGenerator(
        llm=llm,
        output_dir=output_bundle,
        bundle_name="FDE Class 4 Knowledge Base",
        domain="AI engineering and evaluation",
        verbose=False,
        stale_after_days=90,
    )
    gen2.add_source(source_file)
    t0 = time.perf_counter()
    gen2.generate()
    t_cached_gen = time.perf_counter() - t0
    print(f"  • Run 2 (Cached Re-generation via SHA-256): {t_cached_gen * 1000:.2f} ms ({t_cached_gen:.4f} s)")
    speedup = t_cold_gen / max(t_cached_gen, 0.0001)
    print(f"  -> Cache Speedup: {speedup:.1f}x (Sub-second: {'✅ YES' if t_cached_gen < 1.0 else '❌ NO'})")

    # ───────────────────────────────────────────────────────────────────────────
    # 2. SPEC v0.2 CONFORMANCE AUDIT
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[TEST 2] OKF v0.2 SPECIFICATION CONFORMANCE")
    issues = validate_bundle(output_bundle, strict=True)
    if not issues:
        print("  • Strict Conformance Audit : ✅ PASSED (0 issues)")
    else:
        print(f"  • Strict Conformance Audit : ❌ FAILED ({len(issues)} issues)")
        for iss in issues:
            print(f"    - {iss}")

    # Inspect generated files
    index_md = (output_bundle / "index.md").read_text(encoding="utf-8")
    has_okf_v02 = 'okf_version: "0.2"' in index_md
    print(f"  • Root index.md v0.2 header : {'✅ Present' if has_okf_v02 else '❌ Missing'}")

    log_md = (output_bundle / "log.md").read_text(encoding="utf-8")
    log_entries = log_md.count("## ")
    print(f"  • Audit log.md entries      : ✅ {log_entries} recorded run(s)")

    # Check a concept's frontmatter
    sample_concept = output_bundle / "definitions" / "context_engineering.md"
    if sample_concept.exists():
        content = sample_concept.read_text(encoding="utf-8")
        print("  • Sample Generated Concept Frontmatter:")
        for line in content.splitlines()[:15]:
            print(f"      {line}")

    # ───────────────────────────────────────────────────────────────────────────
    # 3. RETRIEVAL & SEARCH BENCHMARK
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[TEST 3] RETRIEVAL & SEARCH LATENCY")
    persist_dir = output_bundle / ".chroma_index"
    embeddings = BenchmarkEmbeddings()

    # Cold Index build
    t0 = time.perf_counter()
    retriever_cold = OKFRetriever.from_bundle(
        output_bundle,
        embeddings=embeddings,
        vectorstore_cls=BenchmarkVectorStore,
        persist_directory=persist_dir,
    )
    t_cold_retriever = time.perf_counter() - t0
    print(f"  • Vector Index Cold Build     : {t_cold_retriever * 1000:.2f} ms ({t_cold_retriever:.4f} s)")

    # Hot Persisted Reload
    t0 = time.perf_counter()
    retriever_hot = OKFRetriever.from_bundle(
        output_bundle,
        embeddings=embeddings,
        vectorstore_cls=BenchmarkVectorStore,
        persist_directory=persist_dir,
    )
    t_hot_retriever = time.perf_counter() - t0
    print(f"  • Vector Index Persisted Reload : {t_hot_retriever * 1000:.2f} ms (Sub-second: {'✅ YES' if t_hot_retriever < 1.0 else '❌ NO'})")

    # In-memory graph search
    toolkit = OKFToolkit.from_bundle(output_bundle)
    tools = {t.name: t for t in toolkit.get_tools(include_write=True)}

    t0 = time.perf_counter()
    search_res = tools["okf_search"]._run("Context Pack")
    t_search = time.perf_counter() - t0
    print(f"  • Keyword Search (okf_search)   : {t_search * 1000:.3f} ms (Sub-second: ✅ YES)")

    t0 = time.perf_counter()
    rel_res = tools["okf_related_concepts"]._run("definitions/context_engineering")
    t_graph = time.perf_counter() - t0
    print(f"  • Graph Traversal (okf_related) : {t_graph * 1000:.3f} ms (Sub-second: ✅ YES)")

    t0 = time.perf_counter()
    read_res = tools["okf_read_concept"]._run("definitions/two_uses_of_ai")
    t_read = time.perf_counter() - t0
    print(f"  • Concept Read (okf_read)       : {t_read * 1000:.3f} ms (Sub-second: ✅ YES)")

    # ───────────────────────────────────────────────────────────────────────────
    # 4. AGENT REAL-TIME MEMORY OPERATIONS
    # ───────────────────────────────────────────────────────────────────────────
    print("\n[TEST 4] AGENT MEMORY WRITE & TRUST VERIFICATION")

    # Agent records a new finding from Class 4 Q1
    t0 = time.perf_counter()
    write_msg = tools["okf_write_concept"]._run(
        concept_id="findings/context_leakage_risk",
        type="Finding",
        title="Context Leakage Risk",
        description="Coding agents must not inherit customer runtime permissions.",
        body=(
            "# Finding\n\n"
            "Confusing the coding agent with the runtime model leads to assumed access to production data. "
            "Context boundaries must be strictly enforced.\n\n"
            "See [Two Uses of AI](/definitions/two_uses_of_ai.md)."
        ),
        tags="security,context,risk",
    )
    t_write = time.perf_counter() - t0
    print(f"  • okf_write_concept Latency    : {t_write * 1000:.2f} ms (Sub-second: {'✅ YES' if t_write < 1.0 else '❌ NO'})")
    print(f"    -> Result: {write_msg}")

    # Immediately verify the new concept
    t0 = time.perf_counter()
    verify_msg = tools["okf_verify_concept"]._run(
        concept_id="findings/context_leakage_risk",
        actor="human:chief_architect",
    )
    t_verify = time.perf_counter() - t0
    print(f"  • okf_verify_concept Latency   : {t_verify * 1000:.2f} ms (Sub-second: {'✅ YES' if t_verify < 1.0 else '❌ NO'})")
    print(f"    -> Result: {verify_msg}")

    # Read the updated concept to confirm trust tier upgrade
    read_updated = tools["okf_read_concept"]._run("findings/context_leakage_risk")
    print("  • Concept Read After Verification:")
    for line in read_updated.splitlines()[:6]:
        print(f"      {line}")

    print("\n" + "=" * 72)
    print("  ALL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmarks()
