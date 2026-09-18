# subspaceknn

[![CI](https://github.com/DiogoRibeiro7/subspaceknn/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/subspaceknn/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/subspaceknn)](https://pypi.org/project/subspaceknn/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/DiogoRibeiro7/subspaceknn)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Interpretable k-nearest-neighbour classification by ensembling kNN models fitted on low-dimensional feature subspaces.

`SubspaceKNNClassifier` fits one k-nearest-neighbour model per small subset of features, ranks those subspaces by cross-validated performance, and lets the best of them vote, each weighted by its score. Because every member of the ensemble lives in a space of one, two or three features, a prediction can be explained by showing the neighbourhoods that produced it, and each subspace can be drawn with its decision regions.

The method generalises the *interpretable kNN* (ikNN) idea of [Brett Kennedy](https://github.com/Brett-Kennedy/ikNN), described in his article [Interpretable kNN (ikNN)](https://towardsdatascience.com/interpretable-knn-iknn-33d38402b8fc), from pairs of features to subspaces of any small size. This package is an independent implementation written from the description of the method. It shares no code, text or results with the original.

## Installation

```sh
pip install subspaceknn            # core: numpy and scikit-learn
pip install "subspaceknn[plot]"    # adds matplotlib for plot_subspaces
```

The package supports Python 3.10 to 3.13 and scikit-learn 1.6 or newer.

## Quick start

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from subspaceknn import SubspaceKNNClassifier

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target_names[iris.target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

clf = SubspaceKNNClassifier(subspace_size=(1, 2), n_subspaces=4).fit(X_train, y_train)
print(clf.score(X_test, y_test))

for subspace, score, weight in zip(clf.subspaces_, clf.subspace_scores_, clf.subspace_weights_):
    print(list(X.columns[list(subspace)]), f"score={score:.3f}", f"weight={weight:.3f}")
```

The estimator follows the scikit-learn contract, so it works inside `Pipeline`, `GridSearchCV` and `cross_val_score`, accepts data frames, and exposes `predict`, `predict_proba` and `score`. Feature scales matter for nearest neighbours, so put a `StandardScaler` in front of it unless the features are already comparable.

### Explaining a prediction

```python
explanation = clf.explain(X_test.iloc[:1])[0]
print(explanation.prediction, explanation.agreement())
for vote in explanation.votes:
    print(vote.feature_names, vote.prediction, f"weight={vote.weight:.3f}")
```

`explain` returns one `Explanation` per sample. It holds the ensemble prediction and probabilities and one `SubspaceVote` per subspace with the features involved, that subspace's own prediction and probabilities, its cross-validated score and its voting weight. `agreement()` is the total weight of the subspaces that voted for the final prediction, and `to_records()` produces rows ready for `pandas.DataFrame.from_records`.

### Drawing the subspaces

```python
from subspaceknn.plotting import plot_subspaces

fig = plot_subspaces(clf, X_train, y_train, sample=X_test.iloc[0].to_numpy())
fig.savefig("subspaces.png")
```

One panel per subspace: a strip plot with decision intervals for one feature, a scatter plot with decision regions for two, a 3-D scatter for three. The highlighted sample is the point being explained.

## How it works

For subspace sizes `d` in `subspace_size`, the estimator enumerates every `d`-subset of the features, fits a `KNeighborsClassifier` on each subset and scores it with stratified cross-validation on the training data (macro-F1 by default). The `n_subspaces` best subsets form the ensemble. For a new sample the class probabilities are the weighted average of the subspace models' probabilities,

```text
p(c | x) = sum_s w_s * p_s(c | x),        w_s = score_s / sum_t score_t,
```

and the prediction is the class with the largest probability. `voting="hard"` replaces `p_s` by the one-hot prediction of each subspace and `weighting="uniform"` replaces `w_s` by equal weights.

The number of subsets grows combinatorially with the number of features, so `max_candidates` caps how many are cross-validated. Above the cap, features are screened by the cross-validated score of their one-dimensional model, and only the best-scoring features are combined, as many as keep the candidate count within the cap. Everything is deterministic: subsets are enumerated in lexicographic order and ties keep that order.

The full description, including how tiny training sets are handled, is in [docs/method.md](docs/method.md).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_neighbors` | `5` | Neighbours used by every subspace model. |
| `subspace_size` | `2` | Size of each subspace, or a sequence of sizes to enumerate together. |
| `n_subspaces` | `5` | Number of best subspaces that vote; `None` uses all candidates. |
| `max_candidates` | `100` | Cap on cross-validated subspaces; triggers feature screening above it. |
| `voting` | `"soft"` | `"soft"` averages probabilities, `"hard"` averages one-hot votes. |
| `weighting` | `"score"` | Weight by cross-validated score, or `"uniform"`. |
| `cv` | `5` | Folds, or any scikit-learn splitter, used to score subspaces. |
| `scoring` | `"f1_macro"` | Any scikit-learn scorer; higher must be better. |
| `knn_weights`, `metric` | `"uniform"`, `"minkowski"` | Passed to the subspace models. |

Fitted attributes include `subspaces_`, `subspace_scores_`, `subspace_weights_`, `estimators_`, the full `candidate_subspaces_` with `candidate_scores_`, the `screened_features_`, and `feature_scores_`, a coarse feature-relevance measure. See the class docstring for the complete list.

## Does interpretability cost accuracy?

Five-fold stratified cross-validated macro-F1 on scikit-learn's toy datasets, features standardised, everything else at its defaults:

| Dataset | kNN | Subspaces of size 2 | Sizes 1, 2 and 3 (8 subspaces) |
| --- | ---: | ---: | ---: |
| iris (4 features) | 0.953 | 0.953 | 0.953 |
| wine (13 features) | 0.960 | 0.945 | 0.967 |
| breast cancer (30 features) | 0.962 | 0.946 | 0.943 |

The ensemble stays within a couple of points of plain kNN while every one of its votes is a picture. The test suite asserts this stays true. Details in [docs/benchmark.md](docs/benchmark.md).

## Limitations

- Features must be numeric and are used as given; encode categorical features and scale everything first.
- Candidate subspaces grow as the binomial coefficient of the feature count; rely on `max_candidates` or a sequence of small sizes for wide data.
- The voting weights are cross-validated scores, not calibrated probabilities. Treat `predict_proba` as a ranking rather than a probability estimate.
- Classification only.

## Development

```sh
git clone https://github.com/DiogoRibeiro7/subspaceknn.git
cd subspaceknn
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

The test suite runs scikit-learn's estimator contract (`check_estimator`) against two configurations, plus behavioural, explanation, plotting and benchmark tests. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License and attribution

MIT, see [LICENSE](LICENSE). The method is due to Brett Kennedy's ikNN; this implementation, its generalisation to arbitrary subspace sizes, and everything in this repository were written independently.
