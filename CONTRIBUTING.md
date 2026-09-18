# Contributing to subspaceknn

Thank you for considering a contribution. The project has one goal: an interpretable nearest-neighbour classifier that behaves exactly like a scikit-learn estimator and explains every prediction it makes. The guidelines below keep that intact as the code grows.

## Ground rules

- Be respectful. The project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- Open an issue before large work so the design can be agreed first. Small fixes can go straight to a pull request.
- Every behavioural claim in the code or the documentation must be backed by a test.

## Development setup

You need Python 3.10 or newer and [uv](https://docs.astral.sh/uv/). There are no other build dependencies.

```sh
git clone https://github.com/DiogoRibeiro7/subspaceknn.git
cd subspaceknn
uv sync --all-extras
```

Optional local hooks that run the same checks as CI: `uvx pre-commit install`.

## Before you open a pull request

Run the same checks CI runs. All of them must pass:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run python examples/iris_explanations.py
uv run --group docs mkdocs build --strict
uv build && uvx twine check dist/*
```

Ruff runs with every rule enabled except the few listed in `pyproject.toml`, and mypy runs in strict mode on the package. Prefer fixing a finding over silencing it; when silencing is the right call, scope the `noqa` or `type: ignore` to the line and say why.

## Coding standards

- **scikit-learn contract first.** `__init__` only stores parameters, validation happens in `fit`, fitted state lives in attributes with a trailing underscore, `predict` has no side effects, `predict_proba` and `predict` agree, and invalid input raises an informative `ValueError`. `tests/test_sklearn_compat.py` runs `check_estimator`; a change that breaks it is a bug.
- **No hidden randomness.** Subspace enumeration, selection and tie-breaking are documented in `docs/method.md`; keep them seed-free and update the note if they change.
- **Explanations are data.** Anything a user needs to understand a prediction belongs in `Explanation` and `SubspaceVote`, not in printed output or plot side effects.
- **Optional dependencies stay optional.** matplotlib is imported lazily; the core package depends only on numpy and scikit-learn.
- **Typed and documented.** Public functions have numpy-style docstrings and complete type annotations.

## Testing standards

- Unit tests for behaviour and validation live in `tests/`; the scikit-learn contract runs against every configuration that changes a code path.
- Numbers that appear in the documentation come from `benchmarks/run_benchmark.py`, and outputs shown in the user guide from running its code. If you change the method, rerun the benchmark and update `docs/benchmark.md`, the README table and the guide together. `tests/test_benchmark.py` is a smaller, offline check of the same comparison.
- Keep tests deterministic: fixed seeds, fixed splitters, no network.

## Documentation

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) from the Markdown files in `docs/`, and the API reference is generated from the docstrings by [mkdocstrings](https://mkdocstrings.github.io/). `uv run --group docs mkdocs serve` previews it with live reload. Every push to `main` publishes it to GitHub Pages.

- Public API changes need a docstring update and, when user-visible, an entry under *Unreleased* in `CHANGELOG.md`.
- Docstrings are rendered as Markdown: use backticks for code and `[name][subspaceknn.Object]` to link to another object, not Sphinx roles.
- Design changes need a corresponding change in `docs/method.md`.
- `scripts/docs_figures.py` regenerates the figures in `docs/assets/`; rerun it when plotting or the default method changes.

## Branches, commits and pull requests

- Branch from `main` using a short prefix: `feat/`, `fix/`, `test/`, `docs/`, `chore/`.
- Write commit subjects in the imperative mood, at most 72 characters.
- Keep pull requests focused and fill in the template, including how the change was validated.

## Licensing of contributions

By submitting a contribution you agree that it is licensed under the project's MIT license, without any additional terms or conditions.
