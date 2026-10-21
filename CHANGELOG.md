# Changelog

## Unreleased

### Added
- Millisecond-level reruns via caching: `OKFBundleGenerator` hashes content chunks (SHA-256) and caches LLM results in `.okf_cache.json` in the bundle directory. If chunks or images are unchanged, extraction runs in milliseconds and uses 0 LLM API calls.
- Parallel LLM extraction: `OKFBundleGenerator` executes LLM calls concurrently using thread pools (controlled via `max_concurrency`, default `4`) to speed up first-run extraction.
- Decorative image filtering: `OKFBundleGenerator` automatically skips analyzing decorative graphics, logos, banners, headers, and footers to save API costs and processing time. Uses a combination of size/aspect-ratio heuristics and multimodal LLM classification.
- Asset directory cleanup: `OKFBundleGenerator` removes any unused/irrelevant extracted images from the bundle's `assets/` directory at the end of the run.
- Model-agnostic chunking: `OKFBundleGenerator` accepts a `chunk_size` at
  construction, and `generate(chunk_size=...)` overrides per call. When
  unset, a conservative default is used that is reliable on any model, local
  or hosted — no guessing from model names or context windows.
- Output-truncation recovery: when a model hits its output-token limit
  mid-answer (`finish_reason == "length"` and equivalents), the chunk is split
  in half and re-extracted recursively so no concepts are lost.
- Concept normalization: every extracted concept is guaranteed to have the
  keys the pipeline depends on (`category`, `filename`, `type`, `title`,
  `body`); concepts with no usable content are dropped.
- JSON salvage: complete concept objects are recovered from truncated or
  malformed JSON arrays (unescaped newlines, trailing prose, cut-off output).
- Oversized-passage handling: a single paragraph/table larger than the chunk
  budget is hard-split so no call ever exceeds the model's limits.
- Transient-error resilience: rate limits and 5xx provider errors (capacity,
  timeouts, gateway failures) are retried with exponential backoff + jitter,
  and the default attempt count was raised so a brief provider outage can
  recover instead of failing instantly. Verbose multi-line SDK/provider errors
  (e.g. Pydantic validation dumps) are collapsed to a single readable line with
  the error type and any HTTP status code.

### Changed
- Progress feedback: the chunk-size line notes whether the size is the default
  or custom; per-chunk lines show a running elapsed clock; zero-concept chunks
  and images are reported distinctly (pointing to the warnings above); image
  extraction is announced with a note that it needs a multimodal LLM; and the
  final "no concepts" warning suggests concrete next steps.

### Fixed
- `FileExistsError` / nested-directory bug when a model returned a `category`
  or `filename` containing slashes (e.g. `research/r_swa`), a `.md` suffix, or
  other path characters. Category and filename are now sanitized to a single
  flat, lowercased path segment, and writes disambiguate genuine collisions
  (file-vs-directory, or two concepts resolving to the same path) with a
  numeric suffix. Index links use the same sanitized names.
- `KeyError: 'category'` (and other missing keys) when building the
  incremental-enrichment prompt from previously stored concepts — the prompt
  now tolerates concepts with missing fields, and normalization prevents such
  concepts from being stored in the first place.
- `shutil.SameFileError` when Docling-extracted images already live in the
  bundle's `assets/` directory — `_copy_image_asset` now skips same-path copies.
- Message content returned as a list of content blocks (LangChain v1 chat
  models) is now coerced to text before JSON parsing.

## 0.1.0 (2026-06-27)

Initial release.

### Components
- `OKFBundleLoader` — Load OKF bundles as LangChain Documents with type/tag filtering
- `OKFKnowledgeGraph` — Directed graph from OKF cross-links with neighbor/backlink/subgraph queries
- `OKFRetriever` — Graph-aware retriever combining vector similarity with cross-link expansion
- `OKFToolkit` — 4 agent tools (list, read, related, search) with source citations
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
- 90+ unit tests with mock LLM (no API keys needed)
- Integration test stubs (run with `OPENAI_API_KEY` set)
- Python 3.10+ (aligned with LangChain v1)
- All core dependencies MIT/BSD licensed
