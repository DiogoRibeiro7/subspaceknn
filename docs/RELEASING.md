# Releasing

Releases are driven by git tags. Pushing a tag of the form `vX.Y.Z` runs the [release workflow](../.github/workflows/release.yml), which:

1. checks that the tag matches the version in `pyproject.toml`;
2. runs the test suite, builds the wheel and sdist, and checks them with twine;
3. publishes to PyPI through trusted publishing, unless that version is already there;
4. creates a GitHub release whose notes are the matching `CHANGELOG.md` section.

## Release checklist

1. Make sure `main` is green in CI.
2. On a branch, bump `version` in `pyproject.toml` and `__version__` in `src/subspaceknn/__init__.py` following semantic versioning.
3. In `CHANGELOG.md`, rename the *Unreleased* section to `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty *Unreleased* section above it, and update the link references at the bottom.
4. Open a pull request titled `Release X.Y.Z` and merge it.
5. Tag the merge commit and push the tag:

   ```sh
   git checkout main
   git pull
   git tag -a vX.Y.Z -m "subspaceknn X.Y.Z"
   git push origin vX.Y.Z
   ```

6. Watch the release workflow. If it fails before publishing, fix the problem, delete the tag locally and remotely, and tag again. If it fails after publishing, do not re-tag: PyPI versions are immutable, so bump to the next patch version instead.

## One-time setup

### Trusted publishing on PyPI

PyPI allows a trusted publisher to be configured before the first release ("pending publisher"). On <https://pypi.org/manage/account/publishing/> add:

| Field | Value |
| --- | --- |
| PyPI project name | `subspaceknn` |
| Owner | `DiogoRibeiro7` |
| Repository name | `subspaceknn` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Then in the GitHub repository settings create an environment named `pypi`. Restricting it to protected tags is recommended so only maintainers can trigger a publish.

## Versioning policy

- Pre-1.0: minor versions may contain breaking API changes and are called out in the changelog; patch versions never do.
- Dropping a Python version or raising the minimum scikit-learn version is at least a minor bump.
- Changes that alter predictions on the benchmark datasets are documented in the changelog with the new numbers.
