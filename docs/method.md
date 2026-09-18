# Complementary subspace kNN

This note describes the method implemented by `SubspaceKNNClassifier`: what it computes, why, what it costs, and how it relates to the work it builds on.

## Idea

A k-nearest-neighbour classifier in the full feature space is accurate but opaque: a neighbourhood in twenty dimensions cannot be drawn. A kNN model on one, two or three features can be drawn completely, decision regions included, but is usually too weak on its own. An ensemble of such small models keeps the pictures and recovers much of the accuracy, as Brett Kennedy's ikNN showed for pairs of features (Kennedy, 2024).

The question is which small models to keep. Ranking subspaces by their individual score and keeping the best fills the ensemble with near-duplicates: the best pairs of a dataset tend to share its strongest feature, so their votes are right and wrong on the same samples, and every extra picture repeats the first. On five of the fourteen datasets in the [benchmark](benchmark.md#redundancy-of-the-selected-pairs), all five top-ranked pairs contain the same feature. Complementary subspace kNN chooses subspaces for what they add to the ensemble instead. A subspace is admitted when it improves the out-of-fold predictions of the subspaces already chosen, and it is admitted again, gaining weight, when that improves them further.

Two ingredients make this affordable:

- **Exact leave-one-out predictions from one query.** For a kNN model, the leave-one-out prediction of every training sample comes from a single neighbour query that excludes each sample from its own neighbours. Scoring a subspace costs one query instead of one fit and one query per fold, which makes pools of a thousand candidate subspaces cheap.
- **Greedy selection on stored predictions.** The out-of-fold probabilities of all candidates are computed once. Each greedy step then costs one matrix-vector product over the candidates, without refitting anything.

## Notation

- `X` is the training matrix with `n` samples and `p` features, `y` the labels with classes `c = 1, ..., C`, and `Y` the `n x C` one-hot encoding of `y`.
- A subspace `S` is a subset of feature indices with `|S| = d`.
- `knn_S` is a `KNeighborsClassifier` on the columns `S` of `X`.
- `V_S` is the `n x C` matrix of out-of-fold votes of `knn_S`: class probabilities under soft voting, one-hot predictions under hard voting.
- `s_i` is the weight of sample `i`: `n / (C * n_c)` for a sample of class `c` with `n_c` members when `balance_classes=True`, and `1` otherwise. With balanced weights every class contributes the same total weight.
- The Brier score of an `n x C` matrix of probabilities `Q` is `B(Q) = sum_i s_i * |Q_i - Y_i|^2 / sum_i s_i`, where `Q_i` is row `i` (Brier, 1950).

## Algorithm

1. **Enumerate candidates.** For every size `d` in `subspace_size`, list the `d`-subsets of `{1, ..., p}` in lexicographic order. Sizes larger than `p` are skipped; if none fits, fitting fails with an error that names `p`.
2. **Compute out-of-fold votes.** With `cv="loo"` (the default), query the `k` nearest neighbours of every training sample among the other training samples, in the columns `S`. Row `i` of `V_S` holds the class frequencies among the neighbours of sample `i`, weighted by inverse distance when `knn_weights="distance"`. This equals refitting `knn_S` without sample `i` and predicting it, up to how ties between equidistant neighbours are broken. With an integer or a splitter, `V_S` comes from `cross_val_predict` instead, and the splitter must partition the samples.
3. **Score candidates.** `score_S` is the configured scorer evaluated on the out-of-fold predictions in `V_S`. Any scikit-learn scorer works, including those that need probabilities.
4. **Screen features when there are too many candidates.** If the number of subsets exceeds `max_candidates`, score every single feature this way, rank the features (ties keep index order), and keep the largest number `m` of top features such that the number of subsets of those `m` features, summed over the requested sizes, does not exceed the cap. `m` is never smaller than the largest requested size. Candidates are then enumerated over the kept features only.
5. **Select, complementary (default).** Greedy forward selection with replacement (Caruana et al., 2004) on the balanced Brier score. Start from `T_0 = 0`. At step `t = 1, ..., max_votes`, choose

   ```text
   S_t = argmin_S  B((T_{t-1} + V_S) / t),        T_t = T_{t-1} + V_{S_t},
   ```

   where, once `n_subspaces` distinct subspaces have been chosen, `S` ranges over those only. Keep the step `t*` with the lowest loss. The ensemble is the set of subspaces chosen in the first `t*` steps, and the weight of `S` is the number of times it was chosen divided by `t*`. Ties go to the subspace enumerated first.
6. **Select, ranked (ikNN-style).** With `selection="ranked"`, order candidates by `score_S`, descending, ties keeping enumeration order, and take the first `n_subspaces`. With `weighting="score"`, `w_S = max(score_S, 0) / sum_T max(score_T, 0)`, uniform if every score is zero; with `weighting="uniform"`, `w_S = 1 / n_selected`.
7. **Fit.** Refit `knn_S` on the whole training set for every selected `S`.
8. **Predict.** For a sample `x`, `p(c | x) = sum_S w_S * v_S(c | x)`, where `v_S` is the probability vector of `knn_S` under soft voting or its one-hot prediction under hard voting. The prediction is `argmax_c p(c | x)`; ties resolve to the first class in `classes_`. Rows are renormalised defensively so they always sum to one.

Because `predict` is defined as the argmax of `predict_proba`, the two are consistent by construction, which scikit-learn's contract checks require.

## Design choices

**Why the Brier score.** The selection loss has to be a proper scoring rule, so that it rewards honest probabilities, and it has to stay finite and discriminating. The log-loss is infinite whenever a subspace gives the true class probability zero, which a kNN model with five neighbours does routinely. Accuracy and macro-F1 move in coarse steps, so many candidates tie and the greedy search stalls. The Brier score is proper, bounded and smooth in the weights.

**Why balanced.** On imbalanced data the plain Brier score is dominated by the majority class, and the search prefers subspaces that are confident about it. Weighting samples inversely to their class frequency makes the loss agree with macro-averaged metrics, the default `scoring`. In the prototype that preceded this implementation, the balanced loss was the most robust of the five selection objectives tried: the plain and balanced Brier score and log-loss, and macro-F1 itself.

**Why with replacement.** Choosing a subspace again doubles its weight without adding a picture. Weights stay simple, as vote counts, and the search can still move weight towards the subspaces that matter most. The picture budget `n_subspaces` limits the distinct subspaces, not the votes.

**Why the best prefix.** The search runs `max_votes` steps and keeps the best ensemble it passed through, so a late step that makes the ensemble worse is never kept. The loss after each kept vote is exposed as `selection_path_`.

**Why leave-one-out.** Besides being cheaper than k-fold cross-validation for kNN, leave-one-out uses every sample for every candidate and needs no random split, so selection is deterministic and does not depend on a seed.

### What "complementary" means here

The ensemble averages probabilities, so a subspace is complementary when it is right where the subspaces already chosen are unsure or wrong. That is not the same as carrying independent information. Averaging in a weak but independent signal at half the weight pulls confident, correct predictions towards the middle and can raise the Brier score. Complementary selection will then prefer a near-copy of the strong signal, whose slightly different neighbourhoods smooth the coarse kNN probabilities. What it reliably finds is the other case, a signal that is decisive exactly where the chosen ones are blind: the test suite checks that such a signal is chosen even when ranked selection would pass it over for copies of the first.

## Small training sets

Leave-one-out needs at least `n_neighbors + 1` samples. Cross-validation needs enough samples per class and enough samples per training fold for `n_neighbors`: when `cv` is an integer the implementation uses `min(cv, smallest class count)` folds. If the chosen scheme is not possible (leave-one-out with `n_neighbors = n`, or k-fold with fewer than two folds or a training fold smaller than `n_neighbors`), the votes are computed on the training data itself. These resubstitution votes are optimistic and only exist so that the estimator behaves sensibly on toy inputs; the fallback is documented rather than hidden.

## Cost

- **Scoring.** One neighbour query of `n` points per candidate, in `d` dimensions, so the fit time grows with the number of candidates more than with `n`. With the defaults, fits on the fourteen benchmark datasets take between a few hundredths of a second and seven seconds, the slowest being qsar-biodeg with 820 candidate pairs; the benchmark note lists every measured time.
- **Memory.** Complementary selection stores the out-of-fold votes of every candidate: `n_candidates * n * C` floats of eight bytes. A thousand candidates on 5000 samples and three classes take 120 MB. Lower `max_candidates` for large training sets; ranked selection does not store them.
- **Selection.** `max_votes` matrix-vector products with an `n_candidates x (n * C)` matrix. Writing `A = T_{t-1} / t - Y`, the loss of candidate `S` is, up to the constant factor `1 / sum_i s_i`, `sum_i s_i |A_i|^2 + (2 / t) <V_S, s A> + (1 / t^2) sum_i s_i |V_S,i|^2`. The last term is computed once, so each step needs only the inner products, and the candidate ensembles are never formed.
- **Prediction.** `n_selected` kNN queries in `d` dimensions, which is usually cheaper than one query in `p` dimensions.

## What the explanation contains

`explain(X)` returns one `Explanation` per row: the ensemble prediction and probabilities, the class order, and one `SubspaceVote` per selected subspace with the feature indices and names, `score_S`, `w_S`, the subspace's probability vector and its own prediction, ordered from the heaviest to the lightest weight. `agreement()` is the total weight behind the ensemble prediction. The explanation is computed from the same quantities as `predict_proba`, so it cannot disagree with it.

Under complementary selection the weights have a concrete reading: a weight of `0.4` with 25 votes means the subspace was chosen 10 times. `selection_path_` records why each subspace is in the ensemble: the out-of-fold loss after each vote, in the order the votes were cast.

`feature_scores_` aggregates the selected subspaces' scores per feature (mean over the selected subspaces containing the feature, zero if none). It is a coarse relevance measure, not a Shapley value or a permutation importance.

## Relation to prior work

- **Nearest neighbours** (Cover and Hart, 1967) are the base models, unchanged.
- **Ensembles over feature subsets.** The random subspace method trains each member on a random subset of the features (Ho, 1998). Bay's MFS applies the idea to nearest neighbours, combining nearest-neighbour classifiers on random feature subsets by simple voting (Bay, 1998). Those subsets are random and usually too large to draw, and their purpose is accuracy. Here the subsets are small enough to draw, they are enumerated rather than sampled, and they are chosen.
- **ikNN** (Kennedy, 2024) is the direct ancestor. It builds one two-dimensional kNN per pair of features, evaluates the pairs on the training data, and lets the most predictive pairs vote, weighted by their accuracy and, at prediction time, by how pure the neighbourhood is. Version 0.1.0 of this package was an independent implementation of that design, generalised to subspaces of any size; `selection="ranked"` keeps it available for comparison. Complementary selection departs from it by choosing subspaces jointly, for their contribution to the ensemble, rather than one by one for their own score.
- **Ensemble selection** (Caruana et al., 2004) is the selection procedure: forward stepwise selection with replacement from a library of models, optimising a chosen metric on held-out predictions. This package applies it to a library of subspace kNN models, with exact leave-one-out predictions in place of a separate hill-climbing set, the balanced Brier score as the metric, and a limit on distinct members so that the explanation stays a handful of pictures.

The implementation was written from these descriptions; it does not derive from the source code of any of them.

## Not done on purpose

- **Per-sample weights.** ikNN adjusts a subspace's weight by the purity of the neighbourhood around the sample being predicted. The prototype tried a related scheme, weighting each subspace by its out-of-fold accuracy on the sample's neighbours. It improved the Brier score on some datasets but not macro-F1 consistently, so the weights here are global and each explanation reads the same way for every sample.
- **Regression.** The selection carries over with the squared error in place of the Brier score, but the scoring, weighting and explanation semantics need their own contracts.
- **Categorical features.** Encode them first; the neighbourhood geometry depends on the encoding and should be an explicit choice.
- **Calibration of `predict_proba`.** Complementary selection optimises a proper score, which helps, but the probabilities of five-neighbour models are coarse. A calibration layer such as `CalibratedClassifierCV` can be wrapped around the estimator when calibrated probabilities matter.

## References

- Bay, S. D. (1998). Combining nearest neighbor classifiers through multiple feature subsets. In *Proceedings of the Fifteenth International Conference on Machine Learning (ICML 1998)*, pp. 37–45.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Caruana, R., Niculescu-Mizil, A., Crew, G., and Ksikes, A. (2004). Ensemble selection from libraries of models. In *Proceedings of the Twenty-First International Conference on Machine Learning (ICML 2004)*. <https://doi.org/10.1145/1015330.1015432>
- Cover, T. M., and Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27.
- Ho, T. K. (1998). The random subspace method for constructing decision forests. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(8), 832–844.
- Kennedy, W. B. (2024, May 14). Interpretable kNN (ikNN). *Towards Data Science*. <https://towardsdatascience.com/interpretable-knn-iknn-33d38402b8fc>. Code: <https://github.com/Brett-Kennedy/ikNN>.
