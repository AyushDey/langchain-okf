# langchain-okf

[![PyPI version](https://img.shields.io/pypi/v/langchain-okf.svg?logo=pypi)](https://pypi.org/project/langchain-okf/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OKF v0.2 Compliant](https://img.shields.io/badge/OKF-v0.2%20Compliant-green.svg)](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Plug-and-play, self-improving memory layer and knowledge graph retriever for LangChain agents**, conforming strictly to Google's [Open Knowledge Format (OKF v0.2)](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md).

`langchain-okf` transforms static text and complex PDF documents into human-readable, Git-versionable Markdown knowledge graphs with structured YAML frontmatter. It equips LangChain and LangGraph agents with:
- **Graph-Aware Hybrid Retrieval**: Vector similarity search combined with bidirectional markdown cross-link hops (`[[category/concept_id]]`).
- **Self-Improving Agent Memory**: Autonomous tools for agents to search, read, write new learned concepts, and verify facts with thread-safe atomic writes and full async support.
- **Trust Tiers & Freshness Lifecycle**: Native filtering by confidence (`human-reviewed` > `machine-confirmed` > `unverified`) and automatic eviction of expired policies via `stale_after` timestamps.
- **Sub-Second Performance**: Content fingerprinting (`.okf_index_hash`) bypasses re-embedding on cold reloads, and incremental sync updates vector stores in $O(\Delta N)$.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │    Raw Sources (PDF, MD, TXT, CSV)     │
                      └───────────────────┬────────────────────┘
                                          │
                               OKFBundleGenerator
                               (SHA-256 Chunk Cache)
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │       OKF v0.2 Knowledge Bundle        │
                      │  ────────────────────────────────────  │
                      │  📁 index.md (root catalog & config)   │
                      │  📁 concepts/ (markdown + frontmatter) │
                      │  📁 assets/ (extracted diagrams/tables)│
                      │  📄 log.md (append-only audit log)     │
                      └───────────────────┬────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             ┌─────────────────────┐             ┌─────────────────────┐
             │ OKFKnowledgeGraph   │             │ Vector Store        │
             │ In-memory directed  │             │ (Chroma, pgvector,  │
             │ cross-link graph    │             │ Qdrant, Pinecone)   │
             └──────────┬──────────┘             └──────────┬──────────┘
                        └─────────────────┬─────────────────┘
                                          │
                                     OKFRetriever
                         (Vector Match + Graph Expansion +
                          Trust/Status/Stale Post-Filtering)
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │            LangChain Agent             │
                      │  ────────────────────────────────────  │
                      │  Tool: okf_search                      │
                      │  Tool: okf_read_concept                │
                      │  Tool: okf_related_concepts            │
                      │  Tool: okf_list_concepts               │
                      │  Tool: okf_write_concept (Thread-Safe) │
                      │  Tool: okf_verify_concept (Audit Log)  │
                      └────────────────────────────────────────┘
```

---

## Installation

Core package includes `langchain-core`, `pypdf`, and `pyyaml`:

```bash
pip install langchain-okf
```

Install optional integrations based on your stack:

```bash
# Vector Stores
pip install "langchain-okf[chroma]"       # ChromaDB (recommended file-based store)
pip install "langchain-okf[pgvector]"     # PostgreSQL + pgvector
pip install "langchain-okf[qdrant]"       # Qdrant
pip install "langchain-okf[pinecone]"     # Pinecone
pip install "langchain-okf[milvus]"       # Milvus
pip install "langchain-okf[faiss]"        # FAISS CPU

# LLM & Embedding Providers
pip install "langchain-okf[openai]"       # OpenAI / LM Studio / Local vLLM
pip install "langchain-okf[ollama]"       # Ollama
pip install "langchain-okf[openrouter]"   # OpenRouter

# Advanced Document Extractors
pip install "langchain-okf[loaders]"      # Docling (MIT) — structured text + tables + OCR
```

---

## Quick Start

### 1. Hybrid Retriever (Vector Search + Graph Hop + Trust Filtering)

`OKFRetriever` combines semantic similarity with markdown cross-link graph expansion. When post-filters (`min_trust_tier`, `exclude_stale`, `filter_status`) are active, it automatically over-fetches candidates to eliminate result starvation:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_okf import OKFRetriever

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Loads in ~7ms on subsequent runs via .okf_index_hash fingerprinting
retriever = OKFRetriever.from_bundle(
    "./my_knowledge/",
    embeddings=embeddings,
    persist_directory="./my_knowledge/.chroma",
    k=5,                                # Return top 5 relevant concepts
    expand_depth=1,                     # Traverse 1 hop along [[wikilinks]]
    min_trust_tier="machine-confirmed", # Exclude unverified drafts
    exclude_stale=True,                 # Automatically evict expired policies
    filter_status="stable",             # Only return active concepts
)

# Synchronous query
docs = retriever.invoke("What are the system timeout policies?")
for doc in docs:
    print(f"[{doc.metadata.get('trust_tier')}] {doc.metadata.get('title')}")
    print(f"Source: {doc.metadata.get('sources')}\n")

# Native Async query (FastAPI / LangGraph friendly)
# async_docs = await retriever.ainvoke("What are the system timeout policies?")
```

---

### 2. Autonomous Agent Memory Toolkit

Give your agent tools to query the knowledge graph, author new learned concepts, and upgrade trust ratings. All write tools enforce reentrant write locks (`threading.RLock`) and support non-blocking async execution (`_arun` via `asyncio.to_thread`):

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_okf import OKFToolkit

toolkit = OKFToolkit.from_bundle(
    "./my_knowledge/",
    actor="agent/customer-support-v1",
)

# Include write and verify tools (default is read-only)
tools = toolkit.get_tools(include_write=True)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=tools,
    system_prompt=(
        "You are an enterprise support agent with persistent OKF memory. "
        "Search existing policies before answering. When learning a new edge case, "
        "persist it using okf_write_concept with sources."
    ),
)
```

#### Available Agent Tools

| Tool | Mode | Purpose | Description |
| :--- | :---: | :--- | :--- |
| `okf_search` | Read | Keyword & Title Search | Fast lookup across titles, tags, aliases, and markdown bodies. |
| `okf_read_concept` | Read | Concept Inspection | Retrieves full concept body, YAML frontmatter, trust tier, and sources. |
| `okf_related_concepts`| Read | Graph Edge Traversal | Returns outgoing cross-links and incoming backlinks for a concept. |
| `okf_list_concepts` | Read | Filtered Catalog | Browses concepts filtered by `type:`, `tag:`, `trust:`, or `status:`. |
| `okf_write_concept` | Write | Memory Authoring | Creates a new concept file with frontmatter and appends event to `log.md`. |
| `okf_verify_concept`| Write | Trust Verification | Upgrades trust tier (`unverified` → `machine-confirmed` → `human-reviewed`). |

---

### 3. Generate Knowledge Bundles from Raw Documents

Ingest PDFs, markdown files, CSVs, or text documents and generate structured OKF concepts using an LLM. Repeated runs on unchanged chunks complete in milliseconds via SHA-256 chunk caching:

```python
from langchain_openai import ChatOpenAI
from langchain_okf import OKFBundleGenerator

generator = OKFBundleGenerator(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    output_dir="./my_knowledge/",
    bundle_name="Enterprise Platform Architecture",
    domain="Cloud infrastructure, auth tokens, microservices, and SLAs",
    stale_after_days=90,       # Sets default stale_after on generated concepts
    max_concurrency=4,         # Worker threads for concurrent LLM extraction
)

generator.add_sources([
    "architecture_spec.pdf",
    "security_guide.md",
    "api_reference.txt",
])

# Extracts concepts, writes index.md, concepts/, and log.md
bundle_path = generator.generate()
```

---

### 4. Direct In-Memory Knowledge Graph Queries

You can query the directed cross-link graph directly without setting up an LLM or vector store:

```python
from langchain_okf import OKFKnowledgeGraph

graph = OKFKnowledgeGraph.from_bundle("./my_knowledge/")

print(f"Total concepts: {len(graph.concept_ids)}")
print(f"Total cross-links: {graph.edge_count}")

# Inspect neighbors and incoming backlinks
neighbors = graph.neighbors("auth/jwt_tokens")
backlinks = graph.backlinks("auth/jwt_tokens")
print(f"Referenced concepts: {neighbors}")
print(f"Concepts referencing JWT: {backlinks}")

# Subgraph expansion around a key component
subgraph = graph.subgraph("auth/jwt_tokens", depth=2)
```

---

### 5. Memory-Efficient Document Streaming

Use `OKFBundleLoader` to load concepts into LangChain `Document` objects with metadata filtering:

```python
from langchain_okf import OKFBundleLoader

loader = OKFBundleLoader(
    "./my_knowledge/",
    include_types=["policy", "architecture"],
    exclude_stale=True,
    include_status="stable",
)

# Generator-based streaming: streams documents without loading all files into RAM
for doc in loader.lazy_load():
    print(f"Loaded: {doc.metadata['title']} ({doc.metadata['concept_id']})")
```

---

### 6. Spec Conformance Validation

Audit your knowledge bundle against the official OKF v0.2 specification:

```python
from langchain_okf import validate_bundle

# Strict mode validates ISO 8601 timestamps, actor formats, and source schemas
issues = validate_bundle("./my_knowledge/", strict=True)
if not issues:
    print("✅ 100% OKF v0.2 compliant")
else:
    for issue in issues:
        print(f"⚠️ {issue}")
```

---

## Architectural Performance & Complexity

Instead of relying on fragile, hardware-dependent micro-benchmarks, `langchain-okf` achieves deterministic performance through algorithmic guarantees:

| Operation | Time Complexity | Mechanism | Production Impact |
| :--- | :---: | :--- | :--- |
| **Index Cold Reload** | **$O(1)$** | Content fingerprinting via `.okf_index_hash` | Bypasses vector re-embedding on app restarts (~7–10 ms). |
| **Incremental Vector Sync** | **$O(\Delta N)$** | Content manifest diffing via `.okf_manifest.json` | Adding 1 concept to 5,000 concepts only embeds the 1 modified file. |
| **Graph Traversal & Hops** | **$O(E)$** | In-memory directed adjacency map + link cache | Traverses cross-links and backlinks in sub-millisecond memory hops. |
| **Retriever Candidate Filtering** | **$O(K)$** | Over-fetching candidate pool (`fetch_k = max(3k, 20)`) | Prevents result starvation when strict trust/staleness filters are applied. |
| **Document Ingestion** | **$O(C)$** | Zero-copy `pypdf` streams + SHA-256 `.okf_cache.json` | 100 unchanged PDFs re-run in milliseconds with 0 LLM API calls. |
| **Memory Footprint** | **$O(V + E)$** | Lightweight dataclasses + `__slots__` parsing | Graph of 10,000 concepts consumes less than 25 MB RAM. |

*Notation: $N$ = total concepts, $\Delta N$ = modified concepts, $E$ = edges/links, $K$ = top-k results, $C$ = content chunks, $V$ = vertices.*

---

## Production Patterns & Real-World Use Cases

### 1. Human-in-the-Loop Enterprise Policy Memory
In autonomous customer operations, agents inevitably encounter undocumented edge cases. Traditional vector memory indiscriminately embeds hallucinated answers. With `langchain-okf`:
1. The agent persists the new rule as an `unverified` draft using `okf_write_concept`.
2. The change creates a human-readable Markdown file in Git and appends an event to `log.md`.
3. A domain supervisor inspects the file, verifies compliance, and triggers `okf_verify_concept` (or approves a GitHub PR) promoting the concept to `human-reviewed`.
4. Downstream agents configured with `min_trust_tier="human-reviewed"` immediately access the vetted policy.

### 2. Regulatory Compliance & Policy Expiration (TTL Eviction)
Internal policies and medical or legal guidelines have strict expiration dates. In traditional vector databases, stale policies remain permanently embedded until manually scrubbed. 
- In OKF, every concept specifies `stale_after: "2026-12-31"`.
- `OKFRetriever` with `exclude_stale=True` compares against UTC wall-clock time and evicts expired policies at query time without destroying historical audit trails.

### 3. Multi-Agent Shared Context in LangGraph
In multi-agent systems, agents need a shared, auditable knowledge base:
- **Research Agent**: Scrapes documentation, runs `OKFBundleGenerator`, and writes structured concepts.
- **Reviewer Agent**: Audits concept citations against raw sources and runs `okf_verify_concept`.
- **Execution Agent**: Uses `OKFRetriever` to answer user inquiries strictly grounded in verified concepts.

---

## Contributing & Testing

Run tests and linters locally:

```bash
uv sync --dev --extra chroma
uv run pytest tests/ -v
uv run ruff check langchain_okf/
uv run mypy langchain_okf/
```

To run the automated pre-flight release check:

```bash
python3 scripts/release_check.py
```

---

## License

MIT © Ayush Dey
