# Coding Agent Guidelines & Commit Rules

This file establishes operating principles and commit conventions for any AI coding agent (Antigravity, Cursor, Claude Code, Codex, Copilot, etc.) working in `langchain-okf`.

---

## 1. Commit Message Conventions & SemVer Rules

Every commit created by an agent (and every Pull Request title) must strictly adhere to the **Conventional Commits 1.0.0** specification:

```text
<type>(<optional-scope>): <imperative subject>
```

### Release Impact Matrix

| Commit Type | SemVer Impact | Triggers Release? | When to Use |
| :--- | :--- | :---: | :--- |
| **`feat`** (standard) | **MINOR** (`0.2.0` $\rightarrow$ `0.3.0`) | **YES** | User-facing feature, new retriever functionality, new agent tool |
| **`test`** | **PATCH** (`0.2.0` $\rightarrow$ `0.2.1`) | **YES** | Adding/updating test cases, fixtures, or test utilities |
| **`feat(testing)`** | **PATCH** (`0.2.0` $\rightarrow$ `0.2.1`) | **YES** | Adding test framework compliance suites (e.g. `langchain-tests`) |
| **`fix`** | **PATCH** (`0.2.0` $\rightarrow$ `0.2.1`) | **YES** | Bug fix, runtime exception, or unexpected behavior resolution |
| **`perf`** | **PATCH** (`0.2.0` $\rightarrow$ `0.2.1`) | **YES** | Performance optimization (caching, faster traversal, reduced memory) |
| **`feat!`** / **`BREAKING CHANGE:`** | **MAJOR** (`1.0.0` $\rightarrow$ `2.0.0`)* | **YES** | Incompatible breaking public API change |
| **`docs`** | None (`NO_RELEASE`) | **NO** | Documentation changes (README, docstrings, typing notes) |
| **`refactor`** | None (`NO_RELEASE`) | **NO** | Code reorganization without feature or bug changes |
| **`style`** | None (`NO_RELEASE`) | **NO** | Formatting, whitespace, imports (no code change) |
| **`chore`** | None (`NO_RELEASE`) | **NO** | Maintenance, dependencies, general housekeeping (`uv.lock`) |
| **`ci`** | None (`NO_RELEASE`) | **NO** | GitHub Actions workflows and CI scripts (`.github/workflows/`) |
| **`build`** | None (`NO_RELEASE`) | **NO** | Hatchling packaging configs, build system changes |

> **Critical SemVer Rule for Testing:**
> Both `test` commits and `feat(testing)` commits are configured to increment the **PATCH** version only (`0.2.0` $\rightarrow$ `0.2.1`). Minor version bumps (`0.3.0`) are reserved exclusively for library-level features.

---

## 2. Recommended Scopes

Use concise, lowercase scopes to clarify which subsystem was modified:
- `(retriever)`: `OKFRetriever`, post-filtering, over-fetching, hybrid expansion.
- `(generator)`: `OKFBundleGenerator`, document parsing, caching, image handling.
- `(graph)`: `OKFKnowledgeGraph`, edges, cross-links `[[category/concept_id]]`.
- `(memory)`: Agent memory tools (`okf_write_concept`, `okf_verify_concept`).
- `(testing)`: Test framework additions, compliance suites (`langchain-tests`).
- `(ci)` / `(release)`: Workflow files in `.github/workflows/`.
- `(deps)`: Dependency management in `pyproject.toml` or `uv.lock`.

---

## 3. Pre-Commit Quality Checks

Before committing or pushing changes, ensure all quality gates pass:

```bash
# Lint check
uv run ruff check langchain_okf/ tests/

# Format check
uv run ruff format --check langchain_okf/ tests/

# Type check
uv run mypy langchain_okf/

# Unit tests
uv run pytest tests/ -v
```

---

## 4. Release Automation Constraints

- **Do not manually edit version numbers** in `pyproject.toml` during standard feature work. `python-semantic-release` manages versions, tags, and CHANGELOG entries automatically upon merge to `main`.
- **Changelog source of truth**: Automated releases populate GitHub Release descriptions directly from `CHANGELOG.md`. Keep formatting compact and consistent without blank lines between list items.
