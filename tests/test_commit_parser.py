"""Unit tests for ScopedConventionalCommitParser.

Verifies:
1. `feat(testing)`, `feat(test)`, and `feat(tests)` are downgraded to LevelBump.PATCH.
2. Standard `feat` (e.g. `feat(retriever)`) retains LevelBump.MINOR.
3. Standard `fix` and `perf` retain LevelBump.PATCH.
4. `test(...)` retains LevelBump.PATCH when configured in patch_tags.
5. `ci(...)` commits are excluded from the changelog (include_in_changelog=False, LevelBump.NO_RELEASE).
6. Breaking changes (`!`) retain LevelBump.MAJOR.
7. Real historical commit 1390608 evaluates to LevelBump.PATCH.
"""

# ruff: noqa: E402
from __future__ import annotations

from typing import cast

import pytest

git = pytest.importorskip("git")
pytest.importorskip("semantic_release")

from semantic_release.commit_parser.conventional.options import (
    ConventionalCommitParserOptions,
)
from semantic_release.commit_parser.token import ParsedCommit
from semantic_release.enums import LevelBump

from scripts.commit_parser import ScopedConventionalCommitParser


class MockCommit:
    """Mock commit providing Git Commit attributes for PSR parsing."""

    def __init__(self, message: str, repo: git.Repo) -> None:
        self.message = message
        self.repo = repo
        self.binsha = b"0" * 20
        self.hexsha = "0" * 40
        self.parents = ()
        self.author = None
        self.authored_date = None
        self.committer = None
        self.committed_date = None
        self.tree = None
        self.encoding = "utf-8"
        self.gpgsig = None


def make_commit(message: str, repo: git.Repo) -> git.Commit:
    """Helper to construct a mock commit typed as git.Commit."""
    return cast(git.Commit, MockCommit(message, repo))


@pytest.fixture
def repo() -> git.Repo:
    return git.Repo(".")


@pytest.fixture
def parser() -> ScopedConventionalCommitParser:
    options = ConventionalCommitParserOptions(
        minor_tags=("feat",),
        patch_tags=("fix", "perf", "test"),
        allowed_tags=(
            "feat",
            "fix",
            "perf",
            "build",
            "chore",
            "ci",
            "docs",
            "style",
            "refactor",
            "test",
        ),
    )
    return ScopedConventionalCommitParser(options)


def test_feat_testing_downgraded_to_patch(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = make_commit("feat(testing): add langchain-tests compliance suite\n", repo)
    res = parser.parse(commit)

    assert isinstance(res, list)
    assert len(res) == 1
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.PATCH
    assert item.scope == "testing"
    assert item.include_in_changelog is True


def test_feat_test_downgraded_to_patch(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = make_commit("feat(test): add extra unit test fixtures\n", repo)
    res = parser.parse(commit)

    assert isinstance(res, list)
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.PATCH
    assert item.include_in_changelog is True


def test_feat_tests_downgraded_to_patch(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = make_commit("feat(tests): update integration tests\n", repo)
    res = parser.parse(commit)

    assert isinstance(res, list)
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.PATCH
    assert item.include_in_changelog is True


def test_feat_standard_remains_minor(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = make_commit("feat(retriever): add hybrid search support\n", repo)
    res = parser.parse(commit)

    assert isinstance(res, list)
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.MINOR
    assert item.include_in_changelog is True


def test_fix_standard_remains_patch(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = make_commit("fix(retriever): resolve none check issue\n", repo)
    res = parser.parse(commit)

    assert isinstance(res, list)
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.PATCH
    assert item.include_in_changelog is True


def test_test_commit_is_patch(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = make_commit(
        "test(retriever): add test cases for retriever filtering\n", repo
    )
    res = parser.parse(commit)

    assert isinstance(res, list)
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.PATCH
    assert item.include_in_changelog is True


def test_ci_commit_excluded_from_changelog(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = make_commit(
        "ci(release): configure checkout branch and fetch tags\n", repo
    )
    res = parser.parse(commit)

    assert isinstance(res, list)
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.NO_RELEASE
    assert item.include_in_changelog is False


def test_breaking_change_remains_major(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = make_commit(
        "feat(testing)!: break compatibility in testing harness\n", repo
    )
    res = parser.parse(commit)

    assert isinstance(res, list)
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.MAJOR


def test_historical_commit_1390608_is_patch(
    parser: ScopedConventionalCommitParser, repo: git.Repo
) -> None:
    commit = repo.commit("1390608")
    res = parser.parse(commit)

    assert isinstance(res, list)
    item = res[0]
    assert isinstance(item, ParsedCommit)
    assert item.bump == LevelBump.PATCH
    assert item.scope == "testing"
    assert item.include_in_changelog is True
