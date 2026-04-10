# Contributing to Orchid

## Development Setup

```bash
git clone <repo-url>
cd orchid
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The last command installs git hooks that **automatically run ruff (lint + format) and gitlint (commit message check) before every commit**. A commit with lint errors or a non-conventional message will be rejected locally — no need to wait for CI.

## Commit Message Convention

This project uses **[Conventional Commits](https://www.conventionalcommits.org/)** to enable automatic semantic versioning. Every commit message must follow this format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description | Version Bump |
|------|-------------|-------------|
| `feat` | New feature | **minor** (0.X.0) |
| `fix` | Bug fix | **patch** (0.0.X) |
| `perf` | Performance improvement | **patch** (0.0.X) |
| `refactor` | Code refactor (no feature/fix) | none |
| `docs` | Documentation only | none |
| `style` | Formatting, whitespace | none |
| `test` | Adding/updating tests | none |
| `build` | Build system, dependencies | none |
| `ci` | CI/CD configuration | none |
| `chore` | Maintenance tasks | none |

### Breaking Changes

Append `!` after the type or add `BREAKING CHANGE:` in the footer to trigger a **major** version bump:

```
feat!: remove deprecated build_graph kwargs API

BREAKING CHANGE: The old `build_graph(config=..., default_model=..., reader=...)` 
signature has been removed. Use `OrchidRuntime` instead.
```

### Examples

```
feat(rag): add hierarchical scope filtering for multi-tenant queries
fix(agents): handle timeout in MCP tool calls gracefully
perf(qdrant): batch embed documents instead of one-by-one
docs: update README with OrchidRuntime usage
test(strategies): add concurrent execution tests for CallAllStrategy
refactor(graph): extract supervisor prompt building into helper
ci: add GitLab pipeline with semantic-release
chore: bump litellm to 1.62.0
```

### Validation

Commit messages are validated in CI via [gitlint](https://jorisroovers.com/gitlint/). To check locally:

```bash
pip install gitlint
gitlint             # validates last commit
gitlint --commits HEAD~3..HEAD  # validate range
```

## Code Standards

- **Python 3.11+** with `from __future__ import annotations` in every file
- **Ruff** for linting and formatting (line length 120)
- **SOLID principles** enforced across all code -- see AGENTS.md for details
- **Tests required** for all changes

## Architecture Rules

1. `orchid/core/` must have **ZERO external dependencies** (only Python stdlib)
2. No Qdrant imports outside `rag/backends/`
3. No vendor-specific code in this package -- platform integrations belong in consumers
4. Consumer agents use inherited methods (`self.summarise()`, `self.fetch_rag_context()`)
5. New agents should be YAML-configured via `GenericAgent` when possible

## Running Tests

```bash
pytest tests/ -x                    # all tests, stop on first failure
pytest tests/ --cov=orchid          # with coverage
ruff check orchid/                  # lint
ruff format orchid/                 # format
```

## Pull / Merge Requests

1. Create a feature branch from `main`
2. Make your changes with tests
3. Use conventional commit messages
4. Ensure all tests pass and linting is clean
5. Keep MRs focused -- one feature or fix per MR
6. Update AGENTS.md if you change architecture or patterns
