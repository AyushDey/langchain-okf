"""Custom Conventional Commits parser for python-semantic-release.

Enforces scope-aware bump rules:
- `feat(testing)` (and scopes 'test', 'tests') triggers a PATCH bump
  instead of the default MINOR bump.
- Standard `feat` without testing scopes triggers a MINOR bump.
- Standard `fix` and `perf` trigger a PATCH bump.
- All breaking changes (`!` or `BREAKING CHANGE:`) trigger a MAJOR bump
  (or minor when allow_zero_version / major_on_zero=false).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    from semantic_release.commit_parser import (
        ConventionalCommitParser,
        ConventionalCommitParserOptions,
    )
except ImportError:  # pragma: no cover
    from semantic_release.commit_parser.conventional.options import (
        ConventionalCommitParserOptions,
    )
    from semantic_release.commit_parser.conventional.parser import (
        ConventionalCommitParser,
    )

from semantic_release.commit_parser.token import ParsedCommit
from semantic_release.enums import LevelBump

if TYPE_CHECKING:
    from git.objects.commit import Commit
    from semantic_release.commit_parser.token import ParseResult

logger = logging.getLogger(__name__)

# Scopes under `feat` that should trigger a PATCH bump instead of a MINOR bump
PATCH_FEAT_SCOPES: tuple[str, ...] = ("testing", "test", "tests")


class ScopedConventionalCommitParser(ConventionalCommitParser):
    """Conventional commit parser that treats feat(testing) as a PATCH bump."""

    parser_options = ConventionalCommitParserOptions

    def _downgrade_if_testing(self, item: ParseResult) -> ParseResult:
        if (
            isinstance(item, ParsedCommit)
            and item.type == "feat"
            and item.scope in PATCH_FEAT_SCOPES
        ):
            logger.info(
                "Downgrading feat(%s) to PATCH bump (descriptions: %s)",
                item.scope,
                item.descriptions,
            )
            return item._replace(bump=LevelBump.PATCH)
        return item

    def parse(self, commit: Commit) -> ParseResult:
        result = super().parse(commit)

        if isinstance(result, (list, tuple)):
            return type(result)(self._downgrade_if_testing(item) for item in result)

        return self._downgrade_if_testing(result)
