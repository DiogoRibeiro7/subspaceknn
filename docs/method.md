# The subspace kNN method

This note describes what `SubspaceKNNClassifier` computes, the choices that are not forced by the idea itself, and how the implementation relates to the ikNN method it generalises.

## Idea

A k-nearest-neighbour classifier in the full feature space is accurate but opaque: a neighbourhood in twenty dimensions cannot be drawn. A kNN model on one, two or three features can be drawn completely, decision regions included, but is usually too weak on its own. Subspace kNN takes many such small models, keeps the ones that predict well, and lets them vote. Every vote is a picture, so a prediction can be explained by showing which subspaces agreed, which dissented, and how the sample sits among its neighbours in each.

## Notation

- `X` is the training matrix with `n` samples and `p` features, `y` the labels with classes `c = 1, ..., C`.
- A subspace `S` is a subset of feature indices with `|S| = d`.
- `knn_S` is a `KNeighborsClassifier` fitted on the columns `S` of `X`.
- `score_S` is the cross-validated score of `knn_S` on the training data.

## Algorithm

1. **Enumerate candidates.** For every size `d` in `subspace_size`, list the `d`-subsets of `{1, ..., p}` in lexicographic order. Sizes larger than `p` are skipped; if none fits, fitting fails with an error that names `p`.
2. **Screen features when there are too many candidates.** If the number of subsets exceeds `max_candidates`, score every single feature by cross-validation of its one-dimensional model, rank the features (ties keep index order), and keep the largest number `m` of top features such that the number of subsets of those `m` features, summed over the requested sizes, does not exceed the cap. `m` is never smaller than the largest requested size. Candidates are then enumerated over the kept features only.
3. **Score candidates.** Compute `score_S` for every candidate with stratified k-fold cross-validation and the configured scorer.
4. **Select.** Order candidates by score, descending, ties keeping enumeration order, and take the first `n_subspaces` (all of them if `None`).
5. **Weight.** With `weighting="score"`, `w_S = max(score_S, 0) / sum_T max(score_T, 0)`; if every score is zero the weights are uniform. With `weighting="uniform"`, `w_S = 1 / n_selected`.
6. **Fit.** Refit `knn_S` on the whole training set for each selected `S`.
7. **Predict.** For a sample `x`, `p(c | x) = sum_S w_S * v_S(c | x)`, where `v_S` is the probability vector of `knn_S` under soft voting or the one-hot prediction of `knn_S` under hard voting. The prediction is `argmax_c p(c | x)`; ties resolve to the first class in `classes_`. Rows are renormalised defensively so they always sum to one.

Because `predict` is defined as the argmax of `predict_proba`, the two are consistent by construction, which scikit-learn's contract checks require.

## Small training sets

Cross-validation needs enough samples per class and enough samples per training fold for `n_neighbors`. When `cv` is an integer the implementation uses `min(cv, smallest class count)` folds, and if that leaves fewer than two folds, or a training fold smaller than `n_neighbors`, it scores each subspace on the training data itself instead. This resubstitution score is optimistic and only exists so that the estimator behaves sensibly on toy inputs; it is documented rather than hidden. A user-supplied splitter is passed through unchanged.

## Cost

Fitting cross-validates every candidate: `n_candidates * n_splits` kNN fits, each on `d` columns. With the defaults (pairs, cap of 100) this is at most 500 fits of tiny models. Prediction costs `n_selected` kNN queries in `d` dimensions, which is usually cheaper than one query in `p` dimensions.

## What the explanation contains

`explain(X)` returns one `Explanation` per row: the ensemble prediction and probabilities, the class order, and one `SubspaceVote` per selected subspace with the feature indices and names, `score_S`, `w_S`, the subspace's probability vector and its own prediction, ordered from the heaviest to the lightest weight. `agreement()` is the total weight behind the ensemble prediction. The explanation is computed from the same quantities as `predict_proba`, so it cannot disagree with it.

`feature_scores_` aggregates the selected subspaces' scores per feature (mean over the selected subspaces containing the feature, zero if none). It is a coarse relevance measure, not a Shapley value or a permutation importance.

## Relation to ikNN

Brett Kennedy's ikNN builds one kNN per pair of numeric features, scores pairs by cross-validated macro-F1, predicts with the best pairs weighted by their score, and draws the pairs as scatter plots. Subspace kNN keeps that design and changes the following.

- Subspaces can have any size, and several sizes can be enumerated together. One-dimensional subspaces are drawn as strip plots with decision intervals, three-dimensional ones as 3-D scatter plots.
- The screening rule for wide data is stated exactly (the largest feature count whose subset count fits the cap) rather than a square-root heuristic.
- The estimator satisfies scikit-learn's full estimator contract: no parameter is altered in `__init__`, fitted state lives in trailing-underscore attributes, `predict_proba` exists and agrees with `predict`, prediction has no side effects, and invalid inputs raise informative errors.
- Explanations are data (`Explanation` objects) rather than side effects of `predict`, and plotting returns a figure instead of showing it.
- All ordering is deterministic and documented.

The implementation was written from this description; it does not derive from the original source code.

## Not done on purpose

- Regression. The voting scheme carries over, but the scoring, weighting and explanation semantics would need their own contracts.
- Categorical features. Encode them first; the neighbourhood geometry depends on the encoding and should be an explicit choice.
- Calibration of `predict_proba`. Weights are scores, not likelihoods; a calibration layer such as `CalibratedClassifierCV` can be wrapped around the estimator when calibrated probabilities matter.
