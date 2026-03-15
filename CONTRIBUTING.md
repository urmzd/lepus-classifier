# Contributing

## Prerequisites

- Python 3.8+ (see `requirements.txt` for dependencies)
- [uv](https://github.com/astral-sh/uv) (package manager)
- [just](https://github.com/casey/just) (task runner)
- A `GH_TOKEN` with repo access (for releases)
- A [Weights & Biases](https://wandb.ai) account (for experiment tracking)

## Getting Started

```bash
git clone https://github.com/urmzd/lepus-classifier.git
cd lepus-classifier
just init
```

## Development

```bash
just check    # format, lint, test
just test     # run tests
just fmt      # format code
```

## Commit Convention

We use [Angular Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`

Commits are enforced via [gitit](https://github.com/urmzd/gitit).

## Pull Requests

1. Fork the repository
2. Create a feature branch (`feat/my-feature`)
3. Make changes and commit using conventional commits
4. Open a pull request against `main`

## Code Style

- `ruff` for formatting and linting
- `pytest` for testing
