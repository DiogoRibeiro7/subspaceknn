## Summary

<!-- What does this change and why? Link the issue if there is one. -->

## Validation

<!-- Which tests cover the change? If the method changed, paste the new benchmark numbers. -->

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy` passes
- [ ] `uv run pytest` passes, including the scikit-learn contract checks
- [ ] public functions have docstrings and type annotations
- [ ] `CHANGELOG.md` has an entry under *Unreleased* (when user-visible)
- [ ] `docs/method.md` is updated (when the method changes)
