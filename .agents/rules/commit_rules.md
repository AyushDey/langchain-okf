# Commit Rules & SemVer Guidelines

Every commit in this repository (and every PR title) must adhere to Conventional Commits 1.0.0.

## Release Impact Matrix

- `feat`: **MINOR** release bump (`0.2.0` -> `0.3.0`)
- `test`: **PATCH** release bump (`0.2.0` -> `0.2.1`)
- `feat(testing)`: **PATCH** release bump (`0.2.0` -> `0.2.1`) (downgraded via `scripts/commit_parser.py`)
- `fix`: **PATCH** release bump (`0.2.0` -> `0.2.1`)
- `perf`: **PATCH** release bump (`0.2.0` -> `0.2.1`)
- `feat!` / `fix!` / `BREAKING CHANGE:`: **MAJOR** release bump
- `docs`, `chore`, `ci`, `refactor`, `style`, `build`: **NO_RELEASE**

## Testing & SemVer Rules
Both `test` commit prefixes (configured in `[tool.semantic_release.commit_parser_options]` `patch_tags = ["fix", "perf", "test"]`) and `feat(testing)` commits are configured to increment the **PATCH** version only (`0.2.0` -> `0.2.1`). Minor version bumps are reserved strictly for library-level features.

## Changelog & Release Notes Exclusion
Continuous integration (`ci`) commits are excluded from generated changelogs and GitHub Release descriptions (`include_in_changelog = False`).
