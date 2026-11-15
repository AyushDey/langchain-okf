"""Custom Conventional Commits parser for python-semantic-release.

Enforces scope-aware bump rules:
- `feat(testing)` (and scopes 'test', 'tests') triggers a PATCH bump
  instead of the default MINOR bump, unless it introduces breaking changes.
- Standard `feat` without testing scopes triggers a MINOR bump.
- Standard `fix` and `perf` trigger a PATCH bump.
- Commits of type `ci` / `continuous integration` are excluded from the
  changelog and release descriptions (`include_in_changelog=False`, `NO_RELEASE`).
- All breaking changes (`!` or `BREAKING CHANGE:`) trigger a MAJOR bump
  (or minor when allow_zero_version / major_on_zero=false).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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

logger = logging.getLogger(__name__)

# Scopes under `feat` that should trigger a PATCH bump instead of a MINOR bump
PATCH_FEAT_SCOPES: tuple[str, ...] = ("testing", "test", "tests")
FEAT_TYPES: tuple[str, ...] = ("feat", "features", "feature")
CI_TYPES: tuple[str, ...] = ("ci", "continuous integration")


class ScopedConventionalCommitParser(ConventionalCommitParser):
    """Conventional commit parser that treats feat(testing) as a PATCH bump and omits CI commits."""

    parser_options = ConventionalCommitParserOptions

    def _transform_item(self, item: Any) -> Any:
        if not isinstance(item, ParsedCommit):
            return item

        # 1. Exclude CI commits from changelog and release notes
        if item.type.lower() in CI_TYPES:
            logger.info(
                "Excluding CI commit (%s) from changelog: %s",
                item.scope,
                item.descriptions,
            )
            return item._replace(
                bump=LevelBump.NO_RELEASE,
                include_in_changelog=False,
            )

        # 2. Downgrade feat(testing) to PATCH bump (unless it is a breaking change)
        if (
            item.bump != LevelBump.MAJOR
            and not item.breaking_descriptions
            and item.type.lower() in FEAT_TYPES
            and item.scope
            and item.scope.lower().strip() in PATCH_FEAT_SCOPES
        ):
            logger.info(
                "Downgrading feat(%s) to PATCH bump (descriptions: %s)",
                item.scope,
                item.descriptions,
            )
            return item._replace(bump=LevelBump.PATCH)

        return item

    def parse(self, commit: Commit) -> Any:
        result = super().parse(commit)

        if isinstance(result, list):
            return [self._transform_item(item) for item in result]
        if isinstance(result, tuple):
            return tuple(self._transform_item(item) for item in result)

        return self._transform_item(result)
