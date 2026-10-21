"""Automatic OKF v0.2 bundle generation from arbitrary sources.

Uses LangChain document loaders to ingest source files and an LLM to
extract concepts, generate frontmatter, and produce OKF v0.2-compliant
markdown documents with provenance, trust, and lifecycle metadata.

Sub-second performance notes:
- Subsequent ``generate()`` calls on unchanged content are sub-second
  thanks to SHA-256 chunk/image caching (.okf_cache.json).
- For retrieval, pass ``persist_directory`` to ``OKFRetriever.from_bundle``
  so the vector index is loaded from disk rather than rebuilt.

Example:
    .. code-block:: python

        from langchain_okf import OKFBundleGenerator
        from langchain_openai import ChatOpenAI

        generator = OKFBundleGenerator(
            llm=ChatOpenAI(model="gpt-4o", temperature=0),
            output_dir="./my_bundle/",
            stale_after_days=30,   # auto-set stale_after per concept
        )
        generator.add_sources(["report.pdf", "notes.txt", "schema.png"])
        bundle_path = generator.generate()
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import mimetypes
import random
import re
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_\-]")

# Sentinel for lazy Docling initialization.
_DOCLING_UNINITIALIZED = object()


def _sanitize_segment(value: Any, fallback: str) -> str:
    """Turn a model-provided category/filename into one safe path segment.

    Models sometimes return slashes (``research/r_swa``), a ``.md`` suffix, or
    other filesystem-hostile characters. This collapses everything to a single
    flat segment — no path separators, no leading dots — so a value can never
    create nested directories or collide a file with a directory.
    """
    text = str(value or "").strip()
    # Drop a trailing .md the model may have added.
    if text.lower().endswith(".md"):
        text = text[:-3]
    text = _FILENAME_RE.sub("_", text).strip("_")
    return text.lower() or fallback


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Default characters per LLM call. Deliberately conservative so extraction is
# reliable on ANY model — a small local model and a huge hosted one alike.
# Small chunks keep each call's output well under any output-token cap (the
# real failure mode), and more calls simply cost a little more time. If a chunk
# still truncates, it is split and retried automatically. Override per call via
# ``generate(chunk_size=...)`` or per instance via ``OKFBundleGenerator(
# chunk_size=...)``.
_DEFAULT_CHUNK_CHARS = 4000

_CONCEPT_EXTRACTION_PROMPT = """\
You are a knowledge engineer creating documentation from source material.

STRICT RULES:
- ONLY document information that is EXPLICITLY stated in the source material below.
- Do NOT infer, assume, or add any information not directly present in the text.
- If something is unclear or ambiguous in the source, note it as unclear rather \
than guessing.
- Every claim in the body must be directly traceable to the source material.
- Do NOT add examples, sample data, or hypothetical scenarios unless they \
appear verbatim in the source.

WHAT TO EXTRACT:
Identify each distinct concept — any meaningful, self-contained unit of \
knowledge. Adapt the type and structure to match the domain of the source. \
Examples across domains:

Data/Engineering: Table, Dataset, API, Metric, Pipeline, Schema, Query
Research/Academic: Finding, Method, Hypothesis, Experiment, Dataset, Theory, \
Author, Citation, Model
Product/Business: Product, Feature, Specification, Pricing, Category, Variant, \
Competitor, Market Segment
Project/Team: Requirement, Decision, Milestone, Risk, Stakeholder, Epic, \
User Story, Sprint Goal
Operations: Process, Procedure, Policy, Runbook, Playbook, SLA, Incident, \
Compliance Rule
General: Concept, Definition, Glossary Term, FAQ, Guide, Tutorial, Reference, \
Overview, Summary

For EACH concept, output a JSON object with:
- "type": A short descriptive type that fits the content (use the examples \
above as guidance, or create your own if none fit)
- "title": Human-readable name exactly as it appears in the source
- "description": One-sentence summary using only source language
- "tags": List of short categorization strings derived from the source
- "category": Directory grouping that matches the domain. Lowercase, plural, \
no spaces. Examples: "tables", "findings", "products", "requirements", \
"processes", "definitions", "methods", "features", "policies"
- "filename": A snake_case filename (no extension)
- "body": Structured markdown body adapted to the content type. Use the most \
appropriate structure:
  - Data concepts: # Schema with column tables, # Joins, # Examples
  - Research: # Abstract, # Method, # Results, # Conclusions, # Citations
  - Products: # Specifications, # Features, # Pricing, # Variants
  - Processes: # Steps, # Prerequisites, # Expected Outcome
  - Decisions: # Context, # Decision, # Rationale, # Consequences
  - General: # Overview, # Details, # Related Concepts
  When one concept references another, use [Title](/category/filename.md). \
  ONLY include information from the source material.
  IMPORTANT: End every concept body with a # Source section that cites \
  exactly where the information came from. Include the source filename \
  and page/section if available. Example: \
  "# Source\\n\\nFrom: `report.pdf`, page 5"

Output a JSON array. Output ONLY valid JSON, no commentary.

SOURCE MATERIAL:
{content}
"""

_ENRICHMENT_PROMPT = """\
You are a knowledge engineer. You have an existing knowledge base with these \
concepts:
{existing_concepts}

STRICT RULES:
- ONLY use information EXPLICITLY stated in the new source material below.
- Do NOT infer, assume, or fabricate any details.
- If new material updates an existing concept, preserve existing information and \
add only what the new source explicitly states.
- Add cross-links between related concepts using [Title](/category/filename.md).
- Maintain the same type and structural style as the existing concepts.

For each distinct concept in the new material:
1. If it matches an existing concept, output it with the SAME category and \
filename but with an enriched body that merges old and new information.
2. If it's a new concept, create a new entry with a type and structure that \
fits the content domain.

Output a JSON array of concept objects. Include ONLY new or updated concepts.

NEW SOURCE MATERIAL:
{content}
"""

_IMAGE_ANALYSIS_PROMPT = """\
You are a knowledge engineer analyzing an image from a document.

STRICT RULES:
- ONLY describe what is VISUALLY PRESENT in the image.
- Do NOT guess, infer, or add information not visible.
- Be precise about labels, numbers, arrows, and relationships shown.
- If the image is a logo, banner, graphic, header, footer, or purely decorative element with no useful core information to document, you MUST return an empty JSON array [] and extract no concepts.

First, classify the image as one of these types:
- "Diagram" — architecture, flow, system, network, or ER diagrams
- "Table" — tabular data, matrices, grids
- "Chart" — bar charts, line charts, pie charts, graphs with axes
- "Screenshot" — UI screenshots, tool interfaces, dashboards
- "Flowchart" — process flows, decision trees, swim lanes
- "Infographic" — visual summaries, annotated graphics
- "Photo" — real-world photographs
- "Code" — code snippets, terminal output
- "Logo/Banner" — logos, banners, headers, footers, or decorative graphics that do not contain core documentation content
- "Other" — anything that doesn't fit above

For EACH distinct concept visible in the image, output a JSON object with:
- "type": The image classification from above (e.g. "Diagram", "Table")
- "title": A descriptive title for what the image shows
- "description": One-sentence summary of the image content
- "tags": Relevant tags derived from visible content
- "category": Directory grouping. Use "diagrams", "tables", "charts", \
"screenshots", or another fitting category. Lowercase, plural, no spaces.
- "filename": A snake_case filename (no extension)
- "image_type": The classification (Diagram, Table, Chart, etc.)
- "body": Structured markdown body that captures ALL information visible \
in the image. For diagrams: list all components and their connections. \
For tables: reproduce the full table in markdown. For charts: describe \
axes, data points, trends. For flowcharts: list all steps and branches. \
When a concept references another, use [Title](/category/filename.md). \
End with a # Source section containing: \
"Extracted from image: `{filename}` (original path: `{filepath}`)"

Output a JSON array. Output ONLY valid JSON, no commentary.
"""


def _default_logger(message: str) -> None:
    print(message, flush=True)


def _llm_actor_name(llm: Any) -> str:
    """Derive an OKF actor string for ``generated.by`` from a LangChain LLM.

    Returns a string like ``okf-generator/gpt-4o`` or falls back to
    ``okf-generator/unknown``. Actor convention: ``<producer>/<version>`` (§7).
    """
    # Try common LangChain model-name attributes
    for attr in ("model_name", "model", "model_id", "deployment_name"):
        val = getattr(llm, attr, None)
        if isinstance(val, str) and val:
            # Sanitize: replace spaces and slashes that would break the path segment
            safe = val.strip().replace(" ", "-").replace("/", "-")
            return f"okf-generator/{safe}"
    return "okf-generator/unknown"


def _detect_loader(file_path: Path) -> Any:
    """Return an appropriate LangChain loader for the file type."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return None  # handled by _ingest_pdf
    elif suffix == ".csv":
        try:
            from langchain_community.document_loaders import CSVLoader

            return CSVLoader(str(file_path))
        except ImportError as err:
            raise ImportError(
                "CSV support requires langchain-community. Install with: "
                "pip install langchain-community"
            ) from err
    elif suffix in (".docx", ".doc"):
        try:
            from langchain_community.document_loaders import Docx2txtLoader

            return Docx2txtLoader(str(file_path))
        except ImportError as err:
            raise ImportError(
                "Word doc support requires docx2txt. Install with: "
                "pip install langchain-community docx2txt"
            ) from err
    elif suffix in (".html", ".htm"):
        try:
            from langchain_community.document_loaders import BSHTMLLoader

            return BSHTMLLoader(str(file_path))
        except ImportError as err:
            raise ImportError(
                "HTML support requires beautifulsoup4. Install with: "
                "pip install langchain-community beautifulsoup4"
            ) from err
    elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"):
        return None
    elif suffix == ".md":
        return _TextFileLoader(file_path)
    else:
        return _TextFileLoader(file_path)


class _TextFileLoader:
    """Simple text file loader that doesn't require langchain-community."""

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path

    def load(self) -> list[Document]:
        text = self.file_path.read_text(encoding="utf-8", errors="replace")
        return [Document(page_content=text, metadata={"source": str(self.file_path)})]


class _URLLoader:
    """Simple URL loader that doesn't require langchain-community."""

    def __init__(self, url: str) -> None:
        self.url = url

    def load(self) -> list[Document]:
        try:
            from langchain_community.document_loaders import WebBaseLoader

            docs: list[Document] = list(WebBaseLoader(self.url).load())
            return docs
        except ImportError as err:
            raise ImportError(
                "URL loading requires langchain-community and beautifulsoup4. "
                "Install with: pip install langchain-community beautifulsoup4"
            ) from err


class OKFBundleGenerator:
    """Generate OKF bundles from arbitrary source files using an LLM.

    Ingests PDFs, text files, CSVs, images, URLs, and more. Uses an LLM
    to identify concepts in the source material and produce spec-compliant
    OKF markdown documents with proper frontmatter and cross-links.

    Args:
        llm: A LangChain chat model for concept extraction.
        output_dir: Directory where the OKF bundle will be written.
        bundle_name: Display name for the bundle (used in index.md).
        domain: Optional domain hint to focus concept extraction.
            Can be a preset like ``"research"``, ``"product"``,
            ``"project"`` or a free-text description like
            ``"medical device regulatory documentation"``.
        verbose: If True, print progress messages during generation.
        logger: Custom logging function. Receives a string message.
            Defaults to ``print``.

    Example:
        .. code-block:: python

            from langchain_okf import OKFBundleGenerator
            from langchain_openai import ChatOpenAI

            generator = OKFBundleGenerator(
                llm=ChatOpenAI(model="gpt-4o", temperature=0),
                output_dir="./my_bundle/",
                domain="research",
            )
            generator.add_sources(["paper.pdf"])
            bundle_path = generator.generate()
    """

    DOMAIN_PRESETS: dict[str, str] = {
        "data": (
            "This source material is about DATA AND ENGINEERING. "
            "Focus on: tables, datasets, schemas, columns, data types, "
            "joins, queries, metrics, KPIs, pipelines, and data models. "
            "Use types like: Table, Dataset, API, Metric, Pipeline, Schema, Query. "
            "Use categories like: tables, datasets, metrics, apis, pipelines."
        ),
        "research": (
            "This source material is ACADEMIC/RESEARCH content. "
            "Focus on: findings, methods, hypotheses, experiments, results, "
            "datasets, models, authors, citations, and theoretical frameworks. "
            "Use types like: Finding, Method, Hypothesis, Experiment, Dataset, "
            "Theory, Model, Author, Citation. "
            "Use categories like: findings, methods, experiments, datasets, models, "
            "citations."
        ),
        "product": (
            "This source material is about PRODUCTS AND BUSINESS. "
            "Focus on: products, features, specifications, pricing, categories, "
            "variants, competitors, market segments, and customer needs. "
            "Use types like: Product, Feature, Specification, Pricing, Category, "
            "Variant, Competitor, Market Segment. "
            "Use categories like: products, features, specifications, pricing, "
            "competitors."
        ),
        "project": (
            "This source material is PROJECT DOCUMENTATION. "
            "Focus on: requirements, decisions, milestones, risks, stakeholders, "
            "epics, user stories, architecture decisions, and sprint goals. "
            "Use types like: Requirement, Decision, Milestone, Risk, Stakeholder, "
            "Epic, User Story, Architecture Decision. "
            "Use categories like: requirements, decisions, milestones, risks, "
            "stakeholders."
        ),
        "operations": (
            "This source material is about OPERATIONS AND PROCESSES. "
            "Focus on: processes, procedures, policies, runbooks, playbooks, "
            "SLAs, incidents, compliance rules, and workflows. "
            "Use types like: Process, Procedure, Policy, Runbook, Playbook, "
            "SLA, Incident, Compliance Rule. "
            "Use categories like: processes, policies, runbooks, incidents, "
            "workflows."
        ),
        "legal": (
            "This source material is LEGAL/COMPLIANCE content. "
            "Focus on: clauses, regulations, obligations, definitions, "
            "parties, terms, conditions, rights, and penalties. "
            "Use types like: Clause, Regulation, Obligation, Definition, "
            "Party, Term, Right, Penalty. "
            "Use categories like: clauses, regulations, obligations, "
            "definitions, terms."
        ),
        "medical": (
            "This source material is MEDICAL/HEALTHCARE content. "
            "Focus on: conditions, treatments, medications, procedures, "
            "symptoms, diagnoses, guidelines, and clinical trials. "
            "Use types like: Condition, Treatment, Medication, Procedure, "
            "Symptom, Diagnosis, Guideline, Clinical Trial. "
            "Use categories like: conditions, treatments, medications, "
            "procedures, guidelines."
        ),
    }

    def __init__(
        self,
        llm: BaseChatModel,
        output_dir: str | Path,
        bundle_name: str | None = None,
        domain: str | None = None,
        verbose: bool = True,
        logger: Callable[[str], None] | None = None,
        chunk_size: int | None = None,
        max_concurrency: int = 4,
        stale_after_days: int | None = None,
    ) -> None:
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.bundle_name = bundle_name or self.output_dir.name
        self.domain = domain
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.max_concurrency = max_concurrency
        self.stale_after_days = stale_after_days
        self._log_fn = logger or _default_logger
        self._sources: list[str | Path] = []
        self._raw_documents: list[Document] = []
        self._image_paths: list[Path] = []
        self._pdf_image_sources: dict[str, dict[str, Any]] = {}
        self._concepts: list[dict[str, Any]] = []
        self._concept_keys: dict[str, int] = {}
        self._docling_converter: Any = _DOCLING_UNINITIALIZED
        self._cached_domain_prefix: str | None = None
        self._existing_concept_count: int = 0
        self._cache: dict[str, Any] = {"version": 1, "chunks": {}, "images": {}}
        # Derive a short model identifier for generated.by actor
        self._llm_actor = _llm_actor_name(llm)

    def _resolve_chunk_size(self, override: int | None) -> int:
        """Pick the per-call chunk size in characters.

        Priority: explicit ``generate(chunk_size=...)`` argument, then the
        instance ``chunk_size`` set at construction, then a conservative
        model-agnostic default that is reliable on any model. No guessing
        from model names or context windows — small chunks work everywhere,
        and truncation is handled at runtime if it ever occurs.
        """
        for candidate in (override, self.chunk_size):
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        return _DEFAULT_CHUNK_CHARS

    def _domain_prefix(self) -> str:
        """Build a domain-specific prompt prefix. Cached after first call."""
        if self._cached_domain_prefix is not None:
            return self._cached_domain_prefix
        if not self.domain:
            self._cached_domain_prefix = ""
            return ""
        preset = self.DOMAIN_PRESETS.get(self.domain.lower())
        if preset:
            self._cached_domain_prefix = f"\nDOMAIN CONTEXT:\n{preset}\n"
        else:
            self._cached_domain_prefix = (
                f"\nDOMAIN CONTEXT:\n"
                f"This source material is about: {self.domain}. "
                f"Focus on concepts, terminology, and structures specific to "
                f"this domain. Choose types and categories that are natural "
                f"for this field.\n"
            )
        return self._cached_domain_prefix

    def _ensure_docling(self) -> Any:
        """Lazily initialize Docling's DocumentConverter on first use.

        Docling (MIT) is the preferred extraction backend. It provides
        structured text, table recognition, image classification, and
        OCR in a single pass. The converter is initialized once and
        reused across all files for efficiency.

        Returns the converter or None if Docling is not installed.
        """
        if self._docling_converter is not _DOCLING_UNINITIALIZED:
            return self._docling_converter
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
            )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True
            pipeline_options.generate_page_images = False

            self._docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        except ImportError:
            self._docling_converter = None
        return self._docling_converter

    def _ingest_with_docling(self, path: Path, documents: list[Document]) -> None:
        """Extract text and images using Docling.

        Docling provides structured markdown with tables, reading order,
        and classified images in a single pass.
        """
        from docling_core.types.doc import PictureItem

        conv_result = self._ensure_docling().convert(str(path))
        doc = conv_result.document

        markdown = doc.export_to_markdown()

        # Extract actual page count from Docling's document model.
        try:
            page_count = doc.num_pages()
        except (AttributeError, TypeError):
            # Fallback: count page-level provenance items.
            try:
                page_nos = set()
                for element, _ in doc.iterate_items():
                    if hasattr(element, "prov") and element.prov:
                        for prov in element.prov:
                            pn = getattr(prov, "page_no", None)
                            if pn is not None:
                                page_nos.add(pn)
                page_count = len(page_nos) if page_nos else 1
            except Exception:
                page_count = 1

        documents.append(
            Document(
                page_content=markdown,
                metadata={
                    "source": str(path),
                    "source_file": path.name,
                    "extraction_method": "docling",
                    "total_pages": page_count,
                },
            )
        )

        assets_dir = self.output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        picture_count = 0

        for element, _level in doc.iterate_items():
            if isinstance(element, PictureItem):
                picture_count += 1
                img = element.get_image(doc)
                if img is None:
                    continue
                img_name = f"{path.stem}_fig{picture_count}.png"
                img_path = assets_dir / img_name
                img.save(str(img_path), "PNG")
                self._image_paths.append(img_path)
                self._pdf_image_sources[str(img_path)] = {
                    "source_file": path.name,
                    "page": getattr(element.prov[0], "page_no", 0)
                    if element.prov
                    else 0,
                }

        self._log(
            f"  Loading: {path.name} -> "
            f"{page_count} page(s), {len(markdown):,} chars (structured), "
            f"{picture_count} figure(s)"
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            self._log_fn(message)

    @staticmethod
    def _summarize_error(exc: Exception) -> str:
        """Collapse a verbose exception into a single readable line.

        Provider outages and SDK validation failures can produce multi-line
        dumps (e.g. Pydantic's per-field errors). This keeps logs readable by
        surfacing the error type, any embedded HTTP status code, and a short
        message, dropping the noise.
        """
        text = " ".join(str(exc).split())
        status = ""
        match = re.search(r"['\"]?code['\"]?\s*[:=]\s*(\d{3})", text)
        if not match:
            match = re.search(r"\b(429|500|502|503|504)\b", text)
        if match:
            status = f" (HTTP {match.group(1)})"
        if len(text) > 160:
            text = text[:157] + "..."
        return f"{type(exc).__name__}{status}: {text}"

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """Whether an error is a transient provider/server issue worth waiting for.

        Transient: rate limits and 5xx server errors (capacity, timeouts,
        gateway failures). Non-transient issues still get retried, but without
        the longer backoff, since an immediate retry is usually harmless.
        """
        text = str(exc).lower()
        transient_markers = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "rate limit",
            "capacity",
            "overloaded",
            "timeout",
            "timed out",
            "temporarily",
            "unavailable",
            "try again",
            "internal server",
            "bad gateway",
            "gateway timeout",
        )
        return any(marker in text for marker in transient_markers)

    def _retry_backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff + jitter before the next attempt."""
        delay = min(1.5 * (2**attempt), 30.0)
        delay += random.uniform(0, 0.5 * delay)
        self._log(f"    Waiting {delay:.1f}s before retry (provider issue)...")
        time.sleep(delay)

    def add_source(self, source: str | Path) -> OKFBundleGenerator:
        """Add a source file or URL.

        Args:
            source: Path to a file, or an HTTP(S) URL.

        Returns:
            self, for chaining.
        """
        self._sources.append(source)
        return self

    def add_sources(self, sources: list[str | Path]) -> OKFBundleGenerator:
        """Add multiple source files or URLs.

        Args:
            sources: List of file paths or URLs.

        Returns:
            self, for chaining.
        """
        self._sources.extend(sources)
        return self

    def add_documents(self, documents: list[Document]) -> OKFBundleGenerator:
        """Add pre-loaded LangChain Documents directly.

        Useful when you've already loaded content with a custom loader.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            self, for chaining.
        """
        self._raw_documents.extend(documents)
        return self

    def _ingest_sources(self) -> list[Document]:
        """Load all sources into Documents."""
        documents = list(self._raw_documents)

        if self._ensure_docling() is not None:
            self._log("  Backend: Docling")
        else:
            self._log("  Backend: pymupdf/pypdf")

        if self._raw_documents:
            total_chars = sum(len(d.page_content) for d in self._raw_documents)
            self._log(
                f"  Pre-loaded: {len(self._raw_documents)} document(s), "
                f"{total_chars:,} chars"
            )

        for source in self._sources:
            source_str = str(source)

            if source_str.startswith(("http://", "https://")):
                self._log(f"  Loading URL: {source_str}")
                loader = _URLLoader(source_str)
                docs = loader.load()
                total_chars = sum(len(d.page_content) for d in docs)
                self._log(f"    -> {total_chars:,} chars")
                documents.extend(docs)
                continue

            path = Path(source_str)
            if not path.exists():
                raise FileNotFoundError(f"Source not found: {path}")

            if path.is_dir():
                dir_files: dict[str, int] = {}
                before = len(documents)
                for f in sorted(path.rglob("*")):
                    if f.is_file() and not f.name.startswith("."):
                        self._ingest_single_file(f, documents)
                        ext = f.suffix.lower() or "other"
                        dir_files[ext] = dir_files.get(ext, 0) + 1
                parts = ", ".join(
                    f"{count} {ext}" for ext, count in sorted(dir_files.items())
                )
                new_docs = len(documents) - before
                self._log(
                    f"  Loading directory: {path.name}/ "
                    f"({parts}) -> {new_docs} pages/chunks"
                )
            else:
                before = len(documents)
                self._ingest_single_file(path, documents)
                new_docs = len(documents) - before
                suffix = path.suffix.lower()
                self_logged = suffix == ".pdf" or (
                    suffix in (".docx", ".doc", ".pptx", ".html", ".htm")
                    and self._ensure_docling() is not None
                )
                if not self_logged:
                    self._log_file_result(path, new_docs)

        return documents

    def _log_file_result(self, path: Path, doc_count: int) -> None:
        """Log the result of loading a single file with type-aware units."""
        suffix = path.suffix.lower()
        name = path.name

        if suffix == ".pdf":
            self._log(f"  Loading: {name} -> {doc_count} page(s)")
        elif suffix == ".csv":
            self._log(f"  Loading: {name} -> {doc_count} row(s)")
        elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            self._log(f"  Loading: {name} -> queued for image processing")
        elif suffix in (".html", ".htm"):
            self._log(f"  Loading: {name} -> 1 web page")
        elif suffix in (".docx", ".doc"):
            self._log(f"  Loading: {name} -> {doc_count} section(s)")
        elif suffix == ".json":
            self._log(f"  Loading: {name} -> {doc_count} record(s)")
        else:
            size = path.stat().st_size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            self._log(f"  Loading: {name} -> {size_str}")

    def _ingest_single_file(self, path: Path, documents: list[Document]) -> None:
        """Ingest a single file, handling PDFs and images separately."""
        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            self._image_paths.append(path)
        elif suffix == ".pdf":
            self._ingest_pdf(path, documents)
        elif suffix in (".docx", ".doc", ".pptx", ".html", ".htm"):
            if self._ensure_docling() is not None:
                self._ingest_with_docling(path, documents)
            else:
                loader = _detect_loader(path)
                if loader is not None:
                    docs = loader.load()
                    for doc in docs:
                        if "source" not in doc.metadata:
                            doc.metadata["source"] = str(path)
                    documents.extend(docs)
        else:
            loader = _detect_loader(path)
            if loader is not None:
                docs = loader.load()
                for doc in docs:
                    if "source" not in doc.metadata:
                        doc.metadata["source"] = str(path)
                documents.extend(docs)

    def _ingest_pdf(self, path: Path, documents: list[Document]) -> None:
        """Extract text and images from a PDF.

        Fallback chain:
        1. Docling (MIT) — structured text + images + tables + OCR
        2. pymupdf (AGPL) — fast text + image extraction
        3. pypdf (BSD) — text only, no images
        """
        if self._ensure_docling() is not None:
            self._ingest_with_docling(path, documents)
            return

        try:
            import fitz

            doc = fitz.open(str(path))
            page_count = len(doc)
            image_count = 0
            min_image_bytes = 5 * 1024
            assets_dir = self.output_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            for page_num in range(page_count):
                page = doc[page_num]

                text = page.get_text()
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(path),
                                "source_file": path.name,
                                "page": page_num + 1,
                                "total_pages": page_count,
                            },
                        )
                    )

                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    img_bytes = base_image["image"]
                    if len(img_bytes) < min_image_bytes:
                        continue
                    img_ext = base_image.get("ext", "png")
                    img_name = (
                        f"{path.stem}_p{page_num + 1}_img{img_index + 1}.{img_ext}"
                    )
                    img_path = assets_dir / img_name
                    img_path.write_bytes(img_bytes)
                    self._image_paths.append(img_path)
                    self._pdf_image_sources[str(img_path)] = {
                        "source_file": path.name,
                        "page": page_num + 1,
                    }
                    image_count += 1

            doc.close()
            self._log(
                f"  Loading: {path.name} -> {page_count} page(s), "
                f"{image_count} image(s) extracted"
            )
            return

        except ImportError:
            pass

        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            page_count = len(reader.pages)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(path),
                                "source_file": path.name,
                                "page": page_num + 1,
                                "total_pages": page_count,
                            },
                        )
                    )
            self._log(f"  Loading: {path.name} -> {page_count} page(s) via pypdf")
            return
        except ImportError:
            pass

        try:
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(str(path)).load()
            for doc in docs:
                doc.metadata["source_file"] = path.name
            documents.extend(docs)
        except ImportError as err:
            raise ImportError(
                "PDF support requires one of these (in order of recommendation):\n"
                "  pip install docling       (MIT, best quality, text+images+tables+OCR)\n"
                "  pip install pymupdf       (AGPL, fast, text+images)\n"
                "  pip install pypdf         (BSD, text only)"
            ) from err

    def _extract_concepts_from_text(self, content: str) -> tuple[list[dict], bool]:
        """Use the LLM to extract concepts from text content.

        Returns a ``(concepts, truncated)`` tuple. ``truncated`` is True when
        the model stopped because it hit its output-token limit, signalling
        the caller to retry with a smaller chunk.
        """
        domain_prefix = self._domain_prefix()
        # Only use the heavier enrichment prompt when doing an incremental
        # update on a pre-existing bundle.  Within a single generation run
        # the extraction prompt is used for every chunk — deduplication is
        # handled by _merge_concepts().  This avoids bloating prompts 2-N
        # with a growing list of already-extracted concepts, which is the
        # single biggest performance penalty.
        if self._existing_concept_count > 0:
            recent = self._concepts[: self._existing_concept_count][-50:]
            existing_summary = "\n".join(
                f"- {c.get('category', 'general')}/{c.get('filename', 'unnamed')} "
                f"[{c.get('type', 'Concept')}]: {c.get('title', 'Untitled')}"
                for c in recent
            )
            if self._existing_concept_count > 50:
                existing_summary = (
                    f"(showing 50 of {self._existing_concept_count} concepts)\n"
                    + existing_summary
                )
            prompt = domain_prefix + _ENRICHMENT_PROMPT.format(
                existing_concepts=existing_summary,
                content=content,
            )
        else:
            prompt = domain_prefix + _CONCEPT_EXTRACTION_PROMPT.format(
                content=content,
            )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        concepts = self._parse_llm_json(response.content)
        return concepts, self._response_was_truncated(response)

    @staticmethod
    def _response_was_truncated(response: Any) -> bool:
        """Detect whether the model stopped due to an output-length limit."""
        metadata = getattr(response, "response_metadata", None) or {}
        for key in ("finish_reason", "stop_reason", "done_reason"):
            reason = metadata.get(key)
            if isinstance(reason, str) and reason.lower() in (
                "length",
                "max_tokens",
                "max_output_tokens",
            ):
                return True
        return False

    def _extract_concepts_from_image(self, image_path: Path) -> list[dict]:
        """Use a multimodal LLM to analyze an image and extract concepts.

        Classifies the image type (diagram, table, chart, etc.),
        extracts all visible information, and stores the source path
        for citation.
        """
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        domain_prefix = self._domain_prefix()
        prompt_text = domain_prefix + _IMAGE_ANALYSIS_PROMPT.format(
            filename=image_path.name,
            filepath=str(image_path),
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
            ]
        )

        response = self.llm.invoke([message])
        concepts = self._parse_llm_json(response.content)

        pdf_info = self._pdf_image_sources.get(str(image_path))
        for concept in concepts:
            concept["image_source"] = str(image_path)
            if pdf_info:
                concept["source_file"] = pdf_info["source_file"]
                concept["source_page"] = pdf_info["page"]

        return concepts

    def _parse_llm_json(self, text: Any) -> list[dict]:
        """Extract a JSON array from LLM output, tolerating markdown fences.

        Handles the common small-model failure modes: content returned as a
        list of blocks, markdown-fenced JSON, prose before/after the array,
        and trailing garbage after the closing bracket.
        """
        text = self._content_to_str(text).strip()
        if not text:
            self._log("  Warning: empty LLM response, no concepts extracted")
            return []

        json_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        bracket_start = text.find("[")
        if bracket_start != -1:
            text = text[bracket_start:]

        candidates = [text]
        bracket_end = text.rfind("]")
        if bracket_end != -1 and bracket_end < len(text) - 1:
            candidates.append(text[: bracket_end + 1])

        for candidate in candidates:
            for strict in (True, False):
                try:
                    result = json.loads(candidate, strict=strict)
                except json.JSONDecodeError:
                    continue
                if isinstance(result, list):
                    return self._normalize_concepts(result)
                if isinstance(result, dict):
                    return self._normalize_concepts([result])

        salvaged = self._salvage_json_objects(text)
        if salvaged:
            self._log(
                f"  Recovered {len(salvaged)} concept(s) from malformed/truncated JSON"
            )
            return self._normalize_concepts(salvaged)

        preview = text[:200].replace("\n", " ")
        self._log(
            f"  Warning: could not parse JSON from LLM response "
            f"(0 concepts). Response started with: {preview!r}"
        )
        return []

    @staticmethod
    def _normalize_concepts(raw: list) -> list[dict]:
        """Guarantee every concept has the keys the pipeline depends on.

        Models vary in what they return; a concept missing ``category``,
        ``filename``, ``type``, ``title``, or ``body`` would otherwise crash
        downstream code. Concepts with no usable content are dropped.
        """
        normalized: list[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            body = str(item.get("body") or "").strip()
            if not title and not body:
                continue
            filename = str(item.get("filename") or title or "unnamed")
            item["type"] = str(item.get("type") or "Concept").strip() or "Concept"
            item["title"] = title or filename.strip()
            item["description"] = str(item.get("description") or "").strip()
            item["category"] = _sanitize_segment(item.get("category"), "general")
            item["filename"] = _sanitize_segment(filename, "unnamed")
            item["body"] = body
            normalized.append(item)
        return normalized

    @staticmethod
    def _salvage_json_objects(text: str) -> list[dict]:
        """Recover complete top-level JSON objects from a malformed array.

        Walks the string tracking brace depth (ignoring braces inside
        strings) and parses each balanced ``{...}`` object independently.
        Tolerates a truncated final object and control characters inside
        string values. Used when the model's output is cut off mid-array
        or contains unescaped newlines.
        """
        objects: list[dict] = []
        depth = 0
        start = -1
        in_string = False
        escape = False

        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    chunk = text[start : i + 1]
                    try:
                        obj = json.loads(chunk, strict=False)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        objects.append(obj)
                    start = -1

        # Truncation recovery: if text ended inside an unclosed object
        if start != -1 and depth > 0:
            raw_tail = text[start:].strip()
            if raw_tail:
                tail_candidate = raw_tail
                if in_string:
                    tail_candidate += '"'
                tail_candidate = tail_candidate.rstrip().rstrip(",")
                open_brackets = tail_candidate.count("[") - tail_candidate.count("]")
                open_braces = tail_candidate.count("{") - tail_candidate.count("}")
                if open_brackets > 0:
                    tail_candidate += "]" * open_brackets
                if open_braces > 0:
                    tail_candidate += "}" * open_braces

                try:
                    obj = json.loads(tail_candidate, strict=False)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass

        return objects

    @staticmethod
    def _content_to_str(content: Any) -> str:
        """Coerce LangChain message content into plain text.

        Chat models may return content as a string or as a list of content
        blocks (dicts with a "text" key, or plain strings).
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return str(content) if content is not None else ""

    def _smart_chunk(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks at paragraph boundaries.

        Merges trailing chunks shorter than 200 chars into the previous
        chunk to avoid wasting LLM calls on tiny fragments.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        paragraphs = re.split(r"\n\n+", text)
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            # A single paragraph larger than chunk_size (e.g. a dense table or
            # text with no blank lines) is hard-split so no chunk ever exceeds
            # the model's budget.
            if len(para) > chunk_size:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                for start in range(0, len(para), chunk_size):
                    chunks.append(para[start : start + chunk_size])
                continue

            para_len = len(para) + 2
            if current_len + para_len > chunk_size and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_len = para_len
            else:
                current.append(para)
                current_len += para_len

        if current:
            tail = "\n\n".join(current)
            if len(tail) < 200 and chunks:
                chunks[-1] += "\n\n" + tail
            else:
                chunks.append(tail)

        return chunks

    def _write_bundle(self) -> Path:
        """Write all extracted concepts to disk as an OKF v0.2 bundle."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        categories: dict[str, list[dict]] = {}
        copied_images: set[str] = set()
        used_paths: set = set()
        new_concepts_written: list[str] = []

        # Compute stale_after if configured
        stale_after_str: str | None = None
        if self.stale_after_days:
            from datetime import timedelta

            stale_dt = datetime.now(timezone.utc) + timedelta(
                days=self.stale_after_days
            )
            stale_after_str = stale_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        for concept in self._concepts:
            cat = _sanitize_segment(concept.get("category"), "general")
            cat_dir = self.output_dir / cat
            cat_dir.mkdir(parents=True, exist_ok=True)

            filename = _sanitize_segment(concept.get("filename"), "unnamed")

            # ── OKF v0.2 frontmatter ─────────────────────────────────────────
            frontmatter: dict[str, Any] = {
                "type": concept.get("type", "Concept"),
                "title": concept.get("title", filename),
                "description": concept.get("description", ""),
            }
            if concept.get("tags"):
                frontmatter["tags"] = concept["tags"]

            # §5.4 lifecycle
            frontmatter["status"] = concept.get("status", "stable")

            # §5.5 stale_after
            if stale_after_str:
                frontmatter["stale_after"] = stale_after_str

            # §5.2 generated (supersedes v0.1 timestamp)
            frontmatter["generated"] = {
                "by": self._llm_actor,
                "at": now,
            }

            # §5.1 sources — build from source_file/source_page metadata
            sources: list[dict[str, Any]] = []
            if concept.get("source_file"):
                src_entry: dict[str, Any] = {
                    "resource": concept["source_file"],
                    "title": concept["source_file"],
                }
                if concept.get("source_page"):
                    src_entry["description"] = f"page {concept['source_page']}"
                sources.append(src_entry)
            if concept.get("image_source"):
                image_src = Path(concept["image_source"])
                asset_name = image_src.name
                frontmatter["image_source"] = f"assets/{asset_name}"
                sources.append(
                    {
                        "resource": f"assets/{asset_name}",
                        "title": asset_name,
                    }
                )
                if str(image_src) not in copied_images:
                    self._copy_image_asset(image_src)
                    copied_images.add(str(image_src))
                if concept.get("image_type"):
                    frontmatter["image_type"] = concept["image_type"]
            if sources:
                frontmatter["sources"] = sources

            body = concept.get("body", "")
            fm_text = yaml.safe_dump(
                frontmatter, sort_keys=False, allow_unicode=True
            ).rstrip()
            file_content = f"---\n{fm_text}\n---\n\n{body}\n"

            file_path = self._unique_path(cat_dir, filename, used_paths)
            file_path.write_text(file_content, encoding="utf-8")
            rel = file_path.relative_to(self.output_dir)
            self._log(f"  Wrote: {rel}")
            new_concepts_written.append(str(rel))

            categories.setdefault(cat, []).append(concept)

        self._write_index(categories)
        self._append_log(new_concepts_written, now)

        # Save cache
        cache_path = self.output_dir / ".okf_cache.json"
        with contextlib.suppress(Exception):
            cache_path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

        # Cleanup unused assets
        assets_dir = self.output_dir / "assets"
        if assets_dir.exists():
            used_assets = set()
            for concept in self._concepts:
                if concept.get("image_source"):
                    used_assets.add(Path(concept["image_source"]).name)
            for f in assets_dir.iterdir():
                if f.is_file() and f.name not in used_assets:
                    with contextlib.suppress(Exception):
                        f.unlink()
                        self._log(f"  Removed unused asset: assets/{f.name}")

        return self.output_dir

    @staticmethod
    def _unique_path(cat_dir: Path, filename: str, used: set) -> Path:
        """Return a writable ``.md`` path, avoiding file/dir collisions.

        Guards against two model-caused problems within a single run: two
        different concepts resolving to the same file, and a filename colliding
        with a directory. An existing ``.md`` file not yet written this run is
        overwritten on purpose — that is the incremental-update behavior.
        Appends ``_2``, ``_3``, ... only to resolve genuine collisions.
        """
        candidate = cat_dir / f"{filename}.md"
        n = 2
        while str(candidate) in used or candidate.is_dir():
            candidate = cat_dir / f"{filename}_{n}.md"
            n += 1
        used.add(str(candidate))
        return candidate

    def _copy_image_asset(self, image_path: Path) -> None:
        """Copy a source image into the bundle's assets/ directory."""
        assets_dir = self.output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        dest = assets_dir / image_path.name
        if not image_path.exists():
            return
        if image_path.resolve() == dest.resolve():
            return
        shutil.copy2(image_path, dest)
        self._log(f"  Copied asset: assets/{image_path.name}")

    def _write_index(self, categories: dict[str, list[dict]]) -> None:
        """Generate OKF v0.2-compliant index.md files.

        Root ``index.md`` carries an ``okf_version: "0.2"`` frontmatter
        block per §12 so consumers can self-identify the bundle version.
        """
        # Root index.md — with okf_version frontmatter (§12)
        root_lines = [
            "---",
            'okf_version: "0.2"',
            "---",
            "",
            f"# {self.bundle_name}",
            "",
        ]
        for cat, concepts in sorted(categories.items()):
            root_lines.append(f"\n## {cat.replace('_', ' ').title()}\n")
            for c in concepts:
                fname = _sanitize_segment(c.get("filename"), "unnamed")
                title = c.get("title", fname)
                desc = c.get("description", "")
                root_lines.append(f"* [{title}](/{cat}/{fname}.md) - {desc}")

            # Per-category index.md (no frontmatter — reserved files can't be concepts)
            cat_index_lines = [f"# {cat.replace('_', ' ').title()}\n"]
            for c in concepts:
                fname = _sanitize_segment(c.get("filename"), "unnamed")
                title = c.get("title", fname)
                desc = c.get("description", "")
                cat_index_lines.append(f"* [{title}]({fname}.md) - {desc}")
            (self.output_dir / cat / "index.md").write_text(
                "\n".join(cat_index_lines) + "\n", encoding="utf-8"
            )

        (self.output_dir / "index.md").write_text(
            "\n".join(root_lines) + "\n", encoding="utf-8"
        )
        self._log("  Wrote: index.md")

    def _append_log(self, concepts_written: list[str], now: str) -> None:
        """Append an entry to log.md per OKF v0.2 §9.

        The log records each generation run with the date, mode, and
        list of concepts written, newest first.
        """
        log_path = self.output_dir / "log.md"
        date_str = now[:10]  # YYYY-MM-DD

        # Build new entry
        mode = (
            "Incremental update"
            if self._existing_concept_count > 0
            else "Initial generation"
        )
        entry_lines = [f"\n## {date_str}\n"]
        entry_lines.append(
            f"* **{mode}**: {len(concepts_written)} concept(s) written "
            f"by `{self._llm_actor}`."
        )
        for rel in concepts_written[:20]:  # cap to avoid huge logs
            entry_lines.append(f"  - `{rel}`")
        if len(concepts_written) > 20:
            entry_lines.append(f"  - ... and {len(concepts_written) - 20} more.")

        new_block = "\n".join(entry_lines) + "\n"

        if log_path.exists():
            existing = log_path.read_text(encoding="utf-8")
            # Insert after the header line (newest first)
            header_end = existing.find("\n## ")
            if header_end == -1:
                log_path.write_text(existing.rstrip() + new_block, encoding="utf-8")
            else:
                log_path.write_text(
                    existing[:header_end] + new_block + existing[header_end:],
                    encoding="utf-8",
                )
        else:
            header = "# Bundle Update Log\n"
            log_path.write_text(header + new_block, encoding="utf-8")
        self._log("  Wrote: log.md")

    def _load_existing_bundle(self) -> int:
        """Load existing concepts from the output directory if it's a bundle.

        Also loads chunk/image cache (.okf_cache.json) if present.

        Returns the number of existing concepts loaded.
        """
        from langchain_okf._parser import parse_bundle

        cache_path = self.output_dir / ".okf_cache.json"
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {"version": 1, "chunks": {}, "images": {}}

        if not self.output_dir.exists():
            return 0

        existing = parse_bundle(self.output_dir)
        if not existing:
            return 0

        for concept in existing:
            entry = {
                "type": concept.type,
                "title": concept.title,
                "description": concept.description,
                "tags": concept.tags,
                "category": "/".join(concept.concept_id.split("/")[:-1]) or "general",
                "filename": concept.concept_id.split("/")[-1],
                "body": concept.body,
                **{
                    k: v
                    for k, v in concept.frontmatter.items()
                    if k not in ("type", "title", "description", "tags")
                },
            }
            key = f"{entry.get('category', '')}/{entry.get('filename', '')}"
            self._concept_keys[key] = len(self._concepts)
            self._concepts.append(entry)
        count = len(existing)
        self._existing_concept_count = count
        return count

    def _extract_with_retry(
        self, content: str, *, max_retries: int = 3, _depth: int = 0
    ) -> list[dict]:
        """Extract concepts, retrying on error and adapting to truncation.

        Transient provider errors (rate limits, 5xx, capacity) are retried
        with exponential backoff so an outage can recover instead of burning
        all attempts in a couple of seconds. If the model's output is
        truncated (it hit its token cap), the chunk is split in half and each
        half is extracted independently. Recursion is bounded so a pathological
        chunk can't split forever.
        """
        for attempt in range(max_retries + 1):
            try:
                concepts, truncated = self._extract_concepts_from_text(content)
            except Exception as e:
                summary = self._summarize_error(e)
                if attempt < max_retries:
                    self._log(f"    Retry {attempt + 1}/{max_retries}: {summary}")
                    if self._is_retryable_error(e):
                        self._retry_backoff(attempt)
                    continue
                self._log(f"    Failed after {max_retries + 1} attempts: {summary}")
                return []

            if (truncated or not concepts) and _depth < 4 and len(content) > 800:
                reason = "truncated" if truncated else "unparseable/empty"
                self._log(
                    f"    Output {reason} at this size ({len(content):,} chars); "
                    f"splitting chunk and retrying smaller..."
                )
                left, right = self._split_in_half(content)
                out = self._extract_with_retry(
                    left, max_retries=max_retries, _depth=_depth + 1
                )
                out += self._extract_with_retry(
                    right, max_retries=max_retries, _depth=_depth + 1
                )
                return out

            return concepts
        return []

    @staticmethod
    def _split_in_half(text: str) -> tuple[str, str]:
        """Split text near the midpoint at a paragraph or line boundary."""
        mid = len(text) // 2
        for sep in ("\n\n", "\n", ". "):
            idx = text.rfind(sep, 0, mid + len(sep))
            alt = text.find(sep, mid)
            candidate = idx if idx > mid // 2 else alt
            if candidate != -1:
                cut = candidate + len(sep)
                return text[:cut], text[cut:]
        return text[:mid], text[mid:]

    def _extract_image_with_retry(
        self, image_path: Path, *, max_retries: int = 3
    ) -> list[dict]:
        """Extract concepts from an image, retrying with backoff on failure."""
        for attempt in range(max_retries + 1):
            try:
                return self._extract_concepts_from_image(image_path)
            except Exception as e:
                summary = self._summarize_error(e)
                if attempt < max_retries:
                    self._log(f"    Retry {attempt + 1}/{max_retries}: {summary}")
                    if self._is_retryable_error(e):
                        self._retry_backoff(attempt)
                else:
                    self._log(f"    Failed after {max_retries + 1} attempts: {summary}")
                    return []
        return []

    def _is_likely_decorative_or_logo(self, image_path: Path) -> bool:
        """Heuristic check to identify logos/banners/icons without calling the LLM.

        Checks file size, and if PIL is available, aspect ratio and dimensions.
        """
        # 1. Byte size check
        try:
            size_bytes = image_path.stat().st_size
        except OSError:
            return False

        # If it's extremely small (under 3KB), it's likely a tiny icon or bullet graphic.
        if size_bytes < 3 * 1024:
            self._log(
                f"  Skipping image {image_path.name}: too small ({size_bytes} bytes), likely a logo or bullet graphic."
            )
            return True

        # 2. PIL dimension check
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                w, h = img.size

                # Extreme aspect ratios (e.g., very wide/tall banners or thin dividers)
                if h > 0 and w > 0:
                    aspect_ratio = w / h
                    # Banners are typically very wide
                    if aspect_ratio > 4.5 or aspect_ratio < 0.22:
                        self._log(
                            f"  Skipping image {image_path.name}: extreme aspect ratio ({w}x{h}), likely a banner or header."
                        )
                        return True

                    # Typical logos are small square-ish icons
                    if w < 120 and h < 120:
                        self._log(
                            f"  Skipping image {image_path.name}: logo dimensions ({w}x{h}), likely a small icon."
                        )
                        return True
        except ImportError:
            # PIL not available, fall back to byte size only
            pass
        except Exception:
            pass

        return False

    def generate(self, *, chunk_size: int | None = None) -> Path:
        """Run the full pipeline: ingest sources, extract concepts, write bundle.

        Automatically detects whether this is a fresh generation or an
        incremental update. If the output directory already contains an
        OKF bundle, existing concepts are loaded first and new sources
        are merged in.

        Args:
            chunk_size: Maximum characters per LLM call. If omitted, uses the
                instance ``chunk_size`` or a conservative default that is
                reliable on any model. Smaller chunks are more reliable and
                faster per call on slow local models; larger chunks mean fewer
                calls on fast models. If a chunk's output is truncated, it is
                split and retried automatically regardless of this value.

        Returns:
            Path to the generated OKF bundle directory.
        """
        total_start = time.time()

        # Step 0: Detect mode
        existing_count = self._load_existing_bundle()
        if existing_count > 0:
            self._log(
                f"[0/4] Incremental mode: loaded {existing_count} existing concept(s)"
            )
        else:
            self._log("[0/4] Fresh bundle")

        # Step 1: Ingest
        self._log("[1/4] Ingesting sources...")
        documents = self._ingest_sources()
        total_chars = sum(len(d.page_content) for d in documents)

        # Count sources accurately: each file/URL is one source,
        # pre-loaded documents count by distinct source metadata.
        raw_sources = (
            {d.metadata.get("source", "unknown") for d in self._raw_documents}
            if self._raw_documents
            else set()
        )
        source_count = len(self._sources) + len(raw_sources)

        # Count pages accurately: use total_pages metadata when
        # available (Docling, pymupdf), fall back to 1 per Document.
        total_pages = sum(d.metadata.get("total_pages", 1) for d in documents)

        summary_parts = [f"{source_count} source(s)"]
        summary_parts.append(f"{total_pages} page(s)")
        summary_parts.append(f"{total_chars:,} chars")
        if self._image_paths:
            summary_parts.append(f"{len(self._image_paths)} image(s)")
        self._log(f"  Summary: {', '.join(summary_parts)}")

        # Step 2: Chunk
        self._log("[2/4] Preparing text chunks...")
        resolved = self._resolve_chunk_size(chunk_size)
        origin = (
            "default" if chunk_size is None and self.chunk_size is None else "custom"
        )
        chunk_size = resolved
        self._log(f"  Chunk size: {chunk_size:,} chars ({origin})")
        all_text = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in documents
        )
        chunks = self._smart_chunk(all_text, chunk_size)
        self._log(f"  Split into {len(chunks)} chunk(s)")

        # Step 3: Extract concepts
        self._log("[3/4] Extracting concepts from text...")
        chunks_to_extract = []
        chunk_indices = []

        for i, chunk in enumerate(chunks):
            chunk_hash = _sha256_text(chunk)
            if chunk_hash in self._cache.get("chunks", {}):
                self._log(
                    f"  Chunk {i + 1}/{len(chunks)} hit cache (millisecond-level lookup)"
                )
                cached_concepts = self._cache["chunks"][chunk_hash]
                self._merge_concepts(cached_concepts)
            else:
                chunks_to_extract.append(chunk)
                chunk_indices.append(i)

        if chunks_to_extract:
            self._log(
                f"  Extracting {len(chunks_to_extract)} uncached chunk(s) concurrently "
                f"(max_concurrency={self.max_concurrency})..."
            )

            def run_extraction(
                idx_and_chk: tuple[int, str],
            ) -> tuple[int, str, list[dict[str, Any]]]:
                idx, chk = idx_and_chk
                try:
                    res = self._extract_with_retry(chk)
                    return idx, chk, res
                except Exception as e:
                    self._log(f"    Error extracting chunk {idx + 1}: {e}")
                    return idx, chk, []

            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                results = list(
                    executor.map(
                        run_extraction,
                        zip(chunk_indices, chunks_to_extract, strict=False),
                    )
                )

            for _, chk, new_concepts in results:
                chunk_hash = _sha256_text(chk)
                self._cache.setdefault("chunks", {})[chunk_hash] = new_concepts
                self._merge_concepts(new_concepts)

        # Filter out images that are likely logos/banners/icons before processing
        active_images = [
            img
            for img in self._image_paths
            if not self._is_likely_decorative_or_logo(img)
        ]

        # 1. Identify cached vs uncached images
        images_to_extract = []
        image_indices = []

        for i, img_path in enumerate(active_images):
            try:
                img_bytes = img_path.read_bytes()
                img_hash = _sha256_bytes(img_bytes)
            except Exception:
                continue

            if img_hash in self._cache.get("images", {}):
                self._log(
                    f"  Image {i + 1}/{len(active_images)} ({img_path.name}) hit cache (millisecond-level lookup)"
                )
                cached_concepts = self._cache["images"][img_hash]
                self._merge_concepts(cached_concepts)
            else:
                images_to_extract.append((img_path, img_hash))
                image_indices.append(i)

        # 2. Extract uncached images concurrently
        if images_to_extract:
            self._log(
                f"  Processing {len(images_to_extract)} uncached image(s) concurrently "
                f"(max_concurrency={self.max_concurrency})..."
            )

            def run_img_extraction(
                idx_path_hash: tuple[int, Path, str],
            ) -> tuple[int, Path, str, list[dict[str, Any]]]:
                idx, path, hsh = idx_path_hash
                try:
                    res = self._extract_image_with_retry(path)
                    return idx, path, hsh, res
                except Exception as e:
                    self._log(f"    Error extracting image {path.name}: {e}")
                    return idx, path, hsh, []

            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                inputs = [
                    (idx, path, hsh)
                    for idx, (path, hsh) in zip(
                        image_indices, images_to_extract, strict=False
                    )
                ]
                img_results = list(executor.map(run_img_extraction, inputs))

            # 3. Store results in cache and merge
            for _idx, _path, hsh, new_concepts in img_results:
                self._cache.setdefault("images", {})[hsh] = new_concepts
                self._merge_concepts(new_concepts)

        if not self._concepts:
            self._log(
                "  Warning: No concepts extracted. The model may have "
                "returned unparseable output — check the warnings above, or "
                "try a smaller chunk_size or a more capable model."
            )
            return self.output_dir

        # Step 4: Write bundle
        new_count = len(self._concepts) - existing_count
        self._log(f"[4/4] Writing {len(self._concepts)} concepts to bundle...")
        result = self._write_bundle()

        total_elapsed = time.time() - total_start
        if existing_count > 0:
            self._log(
                f"\nDone! {new_count} new/updated + {existing_count} existing "
                f"= {len(self._concepts)} total concepts "
                f"in {total_elapsed:.1f}s -> {result}"
            )
        else:
            self._log(
                f"\nDone! Generated {len(self._concepts)} concepts "
                f"in {total_elapsed:.1f}s -> {result}"
            )
        return result

    def _merge_concepts(self, new_concepts: list[dict]) -> None:
        """Merge new concepts into the existing list, updating duplicates.

        Uses the persistent ``_concept_keys`` dict for O(1) lookups
        instead of rebuilding it from scratch on every call.
        """
        for concept in new_concepts:
            key = f"{concept.get('category', '')}/{concept.get('filename', '')}"
            if key in self._concept_keys:
                self._concepts[self._concept_keys[key]] = concept
            else:
                self._concept_keys[key] = len(self._concepts)
                self._concepts.append(concept)
