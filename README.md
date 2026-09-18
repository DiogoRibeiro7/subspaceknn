# subspaceknn

[![CI](https://github.com/DiogoRibeiro7/subspaceknn/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/subspaceknn/actions/workflows/ci.yml)
[![Docs](https://github.com/DiogoRibeiro7/subspaceknn/actions/workflows/docs.yml/badge.svg)](https://diogoribeiro7.github.io/subspaceknn/)
[![PyPI](https://img.shields.io/pypi/v/subspaceknn)](https://pypi.org/project/subspaceknn/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/DiogoRibeiro7/subspaceknn)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/DiogoRibeiro7/subspaceknn/blob/main/LICENSE)

Interpretable k-nearest-neighbour classification by complementary selection of low-dimensional feature subspaces.

**Documentation:** <https://diogoribeiro7.github.io/subspaceknn/>

`SubspaceKNNClassifier` fits a k-nearest-neighbour model on every small subset of features, one, two or three at a time, and builds a small ensemble of them that votes on new samples. Every member lives in a space that can be drawn, so a prediction is explained by a handful of pictures: which subspaces agreed, which dissented, and where the sample sits among its neighbours in each.

What sets the method apart is how the ensemble is chosen. Instead of keeping the subspaces that score best on their own, which tend to be near-copies of each other, **complementary selection** adds subspaces one vote at a time, each time the one that most improves the ensemble's out-of-fold predictions. Exact leave-one-out predictions, computed from a single neighbour query per subspace, make it cheap to consider up to a thousand candidates. On fourteen benchmark datasets this gains two to three points of macro-F1 over ranking subspaces individually, and with subspaces of up to three features it beats plain kNN in the full feature space on average; see [Does interpretability cost accuracy?](#does-interpretability-cost-accuracy).

The idea of an ensemble of drawable kNN models comes from Brett Kennedy's [interpretable kNN (ikNN)](https://towardsdatascience.com/interpretable-knn-iknn-33d38402b8fc), which ranks pairs of features by their individual accuracy. That ranked scheme is still available as `selection="ranked"`. See [References](#references).

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

for subspace, weight in zip(clf.subspaces_, clf.subspace_weights_):
    print(list(X.columns[list(subspace)]), f"weight={weight:.2f}")
```

The estimator follows the scikit-learn contract, so it works inside `Pipeline`, `GridSearchCV` and `cross_val_score`, accepts data frames, and exposes `predict`, `predict_proba` and `score`. Feature scales matter for nearest neighbours, so put a `StandardScaler` in front of it unless the features are already comparable.

### Explaining a prediction

```python
explanation = clf.explain(X_test.iloc[:1])[0]
print(explanation.prediction, explanation.agreement())
for vote in explanation.votes:
    print(vote.feature_names, vote.prediction, f"weight={vote.weight:.2f}")
```

`explain` returns one `Explanation` per sample. It holds the ensemble prediction and probabilities and one `SubspaceVote` per subspace with the features involved, that subspace's own prediction and probabilities, its individual score and its voting weight. `agreement()` is the total weight of the subspaces that voted for the final prediction, and `to_records()` produces rows ready for `pandas.DataFrame.from_records`.

`selection_path_` tells why each subspace is in the ensemble: the out-of-fold loss after each vote, in the order the votes were cast.

### Drawing the subspaces

```python
from subspaceknn.plotting import plot_subspaces

fig = plot_subspaces(clf, X_train, y_train, sample=X_test.iloc[0].to_numpy())
fig.savefig("subspaces.png")
```

One panel per subspace: a strip plot with decision intervals for one feature, a scatter plot with decision regions for two, a 3-D scatter for three. The highlighted sample is the point being explained.

## How it works

1. **Candidates.** Every subset of `subspace_size` features is a candidate, in lexicographic order. Above `max_candidates` subsets, features are first screened by the score of their one-dimensional model, and only the best are combined.
2. **Out-of-fold votes.** For every candidate, one neighbour query that leaves each training sample out of its own neighbours gives its exact leave-one-out class probabilities. Each candidate is also scored on these predictions with `scoring` (macro-F1 by default).
3. **Complementary selection.** Starting from an empty ensemble, each step adds the candidate that minimises the class-balanced Brier score of the averaged out-of-fold probabilities. A subspace may be added again, which increases its weight. Once `n_subspaces` distinct subspaces are in, only those can be added. After `max_votes` steps the best ensemble seen is kept, and each subspace's weight is its share of the votes.
4. **Prediction.** The class probabilities are the weighted average of the chosen subspace models' probabilities,

   ```text
   p(c | x) = sum_S w_S * p_S(c | x),        w_S = votes_S / total votes,
   ```

   and the prediction is the class with the largest probability. `voting="hard"` replaces `p_S` by the one-hot prediction of each subspace, both in selection and in prediction.

With `selection="ranked"` step 3 is replaced by ikNN-style ranking: the `n_subspaces` best candidates by individual score, weighted by that score or uniformly. Selection does not depend on a random seed: candidates are enumerated in a fixed order, leave-one-out needs no random split, and ties go to the candidate enumerated first. As in scikit-learn's own kNN, which of several equidistant points counts as the k-th neighbour is up to the neighbour search, and it can differ between platforms; on data with many repeated values, such as iris, the chosen subspaces can differ too.

The full description, with the reasoning behind each choice, the cost, and the relation to prior work, is in [docs/method.md](https://diogoribeiro7.github.io/subspaceknn/method/).

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `n_neighbors` | `5` | Neighbours used by every subspace model. |
| `subspace_size` | `2` | Size of each subspace, or a sequence of sizes to enumerate together. |
| `n_subspaces` | `5` | Maximum number of distinct subspaces, that is, pictures; `None` removes the limit. |
| `selection` | `"complementary"` | `"complementary"` selects greedily for joint performance; `"ranked"` keeps the best individual scores (ikNN). |
| `max_votes` | `50` | Greedy steps of complementary selection; weights are vote counts over the best number of votes. |
| `balance_classes` | `True` | Weight classes equally in the Brier score that complementary selection minimises. |
| `max_candidates` | `1000` | Cap on candidate subspaces; triggers feature screening above it. |
| `voting` | `"soft"` | `"soft"` averages probabilities, `"hard"` averages one-hot votes. |
| `weighting` | `"score"` | Ranked selection only: weight by individual score, or `"uniform"`. |
| `cv` | `"loo"` | Exact leave-one-out, or folds, or any splitter that partitions the samples. |
| `scoring` | `"f1_macro"` | Any scikit-learn scorer, used for screening, ranking and reporting; higher must be better. |
| `knn_weights`, `metric` | `"uniform"`, `"minkowski"` | Passed to the subspace models. |

Fitted attributes include `subspaces_`, `subspace_weights_`, `subspace_scores_`, `selection_path_`, `estimators_`, the full `candidate_subspaces_` with `candidate_scores_`, the `screened_features_`, and `feature_scores_`, a coarse feature-relevance measure. See the class docstring for the complete list.

## Does interpretability cost accuracy?

Macro-F1 under 5-fold stratified cross-validation repeated three times, features standardised, averaged over fourteen datasets: scikit-learn's iris, wine and breast cancer and eleven OpenML datasets. Ranked (0.1.0) is the ikNN-style ranking in this package's first release; kNN uses all features.

| Setting | kNN | Ranked (0.1.0) | Complementary |
| --- | ---: | ---: | ---: |
| pairs, 5 subspaces | 0.785 | 0.762 | 0.780 |
| pairs, 3 subspaces | 0.785 | 0.757 | 0.777 |
| sizes 1, 2 and 3, 8 subspaces | 0.785 | 0.772 | **0.798** |

With pairs the ensemble stays within a point of plain kNN while every vote is a scatter plot, and three complementary pairs do better than five ranked ones. With subspaces of up to three features it beats plain kNN on average, on seven datasets out of fourteen, and loses on four. Complementary selection improves on ranking in every setting, and with mixed sizes it is at least as good on every dataset, to within half a point. Per-dataset results, an ablation separating the two ingredients, and fit times are in [docs/benchmark.md](https://diogoribeiro7.github.io/subspaceknn/benchmark/); `benchmarks/run_benchmark.py` reproduces them.

## Limitations

- Features must be numeric and are used as given; encode categorical features and scale everything first.
- Candidate subspaces grow as the binomial coefficient of the feature count; rely on `max_candidates` or a sequence of small sizes for wide data. Screening by one-dimensional scores can miss features that only matter in combination.
- Complementary selection stores every candidate's out-of-fold probabilities, `max_candidates * n_samples * n_classes` floats; lower `max_candidates` for large training sets.
- Complementary means complementary under probability averaging: a weak but independent signal can lower the averaged Brier score and be left out. See [docs/method.md](https://diogoribeiro7.github.io/subspaceknn/method/#what-complementary-means-here).
- The voting weights are vote shares, not calibrated probabilities. Treat `predict_proba` as a ranking unless you calibrate it.
- Classification only.

## Development

```sh
git clone https://github.com/DiogoRibeiro7/subspaceknn.git
cd subspaceknn
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run --group docs mkdocs serve    # documentation at http://127.0.0.1:8000
```

The test suite runs scikit-learn's estimator contract (`check_estimator`) against three configurations, plus behavioural, explanation, plotting and benchmark tests. See [CONTRIBUTING.md](https://github.com/DiogoRibeiro7/subspaceknn/blob/main/CONTRIBUTING.md).

## References

- Kennedy, W. B. (2024, May 14). Interpretable kNN (ikNN). *Towards Data Science*. <https://towardsdatascience.com/interpretable-knn-iknn-33d38402b8fc>. Code: <https://github.com/Brett-Kennedy/ikNN>.
- Caruana, R., Niculescu-Mizil, A., Crew, G., and Ksikes, A. (2004). Ensemble selection from libraries of models. In *Proceedings of the Twenty-First International Conference on Machine Learning (ICML 2004)*. <https://doi.org/10.1145/1015330.1015432>
- Bay, S. D. (1998). Combining nearest neighbor classifiers through multiple feature subsets. In *Proceedings of the Fifteenth International Conference on Machine Learning (ICML 1998)*, pp. 37–45.
- Ho, T. K. (1998). The random subspace method for constructing decision forests. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(8), 832–844.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Cover, T. M., and Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.

## License and attribution

MIT, see [LICENSE](https://github.com/DiogoRibeiro7/subspaceknn/blob/main/LICENSE). The ensemble of drawable kNN models follows Brett Kennedy's ikNN (Kennedy, 2024); complementary selection, the leave-one-out scoring and everything in this repository were written independently and share no code with it.
