# Changelog

<!-- version list -->

## v0.2.0 (2026-11-15)

### Features & Architecture
- Prune dependencies, add retriever over-fetching, and activate CI/CD workflows ([#1](https://github.com/AyushDey/langchain-okf/pull/1), [`be14087`](https://github.com/AyushDey/langchain-okf/commit/be140877197a61f42a24a94642d694a07ab21206))
- **Dependency Pruning**: Pruned phantom dependencies (`langchain-ollama`, `langchain-text-splitters`) from core dependencies, moved `langchain-chroma` to optional extra (`[project.optional-dependencies] chroma`), and kept core lean (`langchain-core`, `pypdf`, `pyyaml`).
- **Retriever Over-Fetching**: Dynamically scale candidate pool (`fetch_k = max(k * 3, 20)`) in `OKFRetriever._get_relevant_documents()` and `_aget_relevant_documents()` when post-filters (`trust_tier`, `stale_after`, `status`, `tags`) are active to prevent result starvation.
- **Thread-Safe & Async Memory Tools**: Wrapped `OKFWriteConceptTool` and `OKFVerifyConceptTool` writes in reentrant locks (`threading.RLock`) and implemented native non-blocking async execution in `_arun()` via `asyncio.to_thread()`.
- **In-Memory Cross-Link Caching**: Cached parsed Markdown link references (`OKFConcept.cross_links()`) to eliminate repetitive regex scans during graph expansion.
- **Automated CI/CD**: Configured GitHub Actions matrix testing across Python 3.10–3.13 and automated PyPA OIDC Trusted Publishing with Python Semantic Release.

### Core OKF v0.2 Extraction Engine
- **Millisecond Reruns via Caching**: `OKFBundleGenerator` hashes content chunks (SHA-256) and caches LLM results in `.okf_cache.json` in the bundle directory. If chunks or images are unchanged, extraction runs in milliseconds and uses 0 LLM API calls.
- **Parallel Extraction**: `OKFBundleGenerator` executes LLM calls concurrently using thread pools (controlled via `max_concurrency`, default `4`) to speed up first-run extraction.
- **Decorative Image Filtering**: `OKFBundleGenerator` automatically skips analyzing decorative graphics, logos, banners, headers, and footers to save API costs and processing time. Uses a combination of size/aspect-ratio heuristics and multimodal LLM classification.
- **Asset Directory Cleanup**: `OKFBundleGenerator` removes any unused/irrelevant extracted images from the bundle's `assets/` directory at the end of the run.
- **Model-Agnostic Chunking**: `OKFBundleGenerator` accepts a `chunk_size` at construction, and `generate(chunk_size=...)` overrides per call. When unset, a conservative default is used that is reliable on any model, local or hosted.
- **Output-Truncation Recovery**: When a model hits its output-token limit mid-answer (`finish_reason == "length"` and equivalents), the chunk is split in half and re-extracted recursively so no concepts are lost.
- **Concept Normalization**: Every extracted concept is guaranteed to have the keys the pipeline depends on (`category`, `filename`, `type`, `title`, `body`); concepts with no usable content are dropped.
- **JSON Salvage**: Complete concept objects are recovered from truncated or malformed JSON arrays (unescaped newlines, trailing prose, cut-off output).
- **Oversized-Passage Handling**: A single paragraph/table larger than the chunk budget is hard-split so no call ever exceeds the model's limits.
- **Transient-Error Resilience**: Rate limits and 5xx provider errors are retried with exponential backoff + jitter. Verbose multi-line provider errors are collapsed to a single readable line.

### Changed
- **Progress Feedback**: The chunk-size line notes whether the size is default or custom; per-chunk lines show running elapsed clock; zero-concept chunks and images reported distinctly; and final "no concepts" warning suggests next steps.

### Fixed
- `FileExistsError` / nested-directory bug when a model returned a `category` or `filename` containing slashes. Category and filename are now sanitized to a single flat, lowercased path segment. Writes disambiguate genuine collisions with a numeric suffix.
- `KeyError: 'category'` (and other missing keys) when building the incremental-enrichment prompt from previously stored concepts.
- `shutil.SameFileError` when Docling-extracted images already live in the bundle's `assets/` directory.
- Message content returned as a list of content blocks (LangChain v1 chat models) is coerced to text before JSON parsing.

## v0.1.0 (2026-06-27)

Initial baseline release of `langchain-okf`.

### Components
- `OKFBundleLoader` — Load OKF bundles as LangChain Documents with type/tag filtering
- `OKFKnowledgeGraph` — Directed graph from OKF cross-links with neighbor/backlink/subgraph queries
- `OKFRetriever` — Graph-aware retriever combining vector similarity with cross-link expansion
- `OKFToolkit` — 4 agent tools (`list`, `read`, `related`, `search`) with source citations
- `OKFBundleGenerator` — Generate OKF bundles from PDFs, text, images, CSVs, URLs using an LLM
  - Auto-detect incremental mode: re-running on an existing bundle merges new content
  - Docling (MIT) as preferred extraction backend — structured text, tables, images, OCR
  - Fallback chain: Docling → pymupdf → pypdf (choose by license needs)
  - Domain-aware extraction with 7 presets + custom free-text domains
  - Error recovery with automatic retry on LLM failures
  - Progress logging with `verbose` flag and custom `logger` support
  - Smart chunking at paragraph boundaries with tiny-chunk merging
  - Anti-hallucination prompts strictly grounded in source material
  - PDF image extraction with classification (diagram/table/chart/photo)
  - Source citations embedded in every concept for traceability
  - Works fully offline after initial model download
- `validate_bundle()` — Check OKF v0.1 conformance of a bundle directory

### Integrations
- Embedding providers: OpenAI, Ollama, OpenRouter
- Vector stores: Chroma, pgvector, Pinecone, Qdrant, FAISS, Milvus
- Agent framework: LangChain v1 `create_agent` with middleware support
- Source extraction: Docling (MIT, recommended), pymupdf (AGPL), pypdf (BSD)
- Async support: `OKFRetriever` and all toolkit tools support async (`ainvoke`, `_arun`)

### Infrastructure
- Built with hatchling, compatible with uv and pip
- PEP 735 `[dependency-groups]` for dev dependencies
- `py.typed` marker for type-checking support
- Complete unit test suite with mock LLM (no API keys needed)
- Python 3.10+ (aligned with LangChain v1)
- All core dependencies MIT/BSD licensed
