# Roadmap to 1.0.0

`subspaceknn` is an interpretable nearest-neighbour classifier: a small ensemble of kNN models on one-, two- and three-feature subspaces, chosen by complementary selection, where every vote can be drawn. Version 1.0.0 is the point where the method, the API and the evidence behind them are stable enough to promise: no breaking changes without a major version, results that do not depend on the operating system, and documented limits on the data sizes it handles.

This roadmap lists what has to happen before then. Milestones are ordered by dependency, not by date. Each ends with an exit criterion that can be checked, and every behavioural claim a milestone adds must be backed by a test, as the [contributing guide](https://github.com/DiogoRibeiro7/subspaceknn/blob/main/CONTRIBUTING.md) requires.

Progress is tracked on GitHub: each milestone below from M2 on is a [GitHub milestone](https://github.com/DiogoRibeiro7/subspaceknn/milestones), and every open item links to its issue.

## Where 0.2.0 stands

What works: complementary selection with exact leave-one-out scoring, the scikit-learn estimator contract on three configurations, structured explanations and plots, a fourteen-dataset benchmark in which the method beats ikNN-style ranking by 1.8 to 2.6 points of mean macro-F1, and a documentation site.

Known gaps, each addressed by a milestone below:

| Gap | Evidence | Milestone |
| --- | --- | --- |
| Selection can differ between operating systems | On iris, macOS chose three subspaces where Linux and Windows chose four: equidistant neighbours are ordered by the platform's C++ library | M2 |
| `selection_path_` docstring says the loss is balanced when `balance_classes=False` | Docstring of `SubspaceKNNClassifier` | M2 |
| Scoring overhead is a large share of small fits | At 2,000 samples, 45% of the fit time is spent in scikit-learn's scorer wrapper | M3 |
| Memory grows with samples times candidates | 367 MB peak at 50,000 samples and 435 candidate pairs, almost all of it stored out-of-fold votes | M3 |
| `n_jobs` does not parallelise across candidates | Only the neighbour query inside each candidate uses it | M3 |
| Feature screening misses interactions | Features are screened by their one-dimensional score, which cannot see a feature that only matters in a pair | M3 |
| Explanations do not show the neighbours | `SubspaceVote` holds probabilities but not which training samples produced them | M4 |
| No missing values, no `sample_weight` | Input validation rejects NaN; `fit` takes no sample weights | M4, M6 |
| `feature_scores_` ignores the weights | It averages individual scores, which says little under complementary selection | M4 |
| Evidence is fourteen datasets, compared only with plain kNN and ikNN-style ranking | `docs/benchmark.md` | M5 |

Baseline for the performance targets, measured on 0.2.0 with the default settings (pairs, at most five subspaces) on `make_classification` data with 30 features, 8 of them informative, and two classes, on a Windows laptop. Times are the best of repeated fits after a warm-up fit:

| Samples | Candidate pairs | Fit time | Peak memory |
| ---: | ---: | ---: | ---: |
| 2,000 | 435 | 2.5 s | 15 MB |
| 10,000 | 435 | 10.9 s | 74 MB |
| 50,000 | 435 | 49.5 s | 367 MB |

## M0 — Foundation (0.1.0, released)

- [x] scikit-learn compatible `SubspaceKNNClassifier` with ikNN-style ranked selection over subspaces of any size;
- [x] `explain` returning `Explanation` and `SubspaceVote` objects, and `plot_subspaces`;
- [x] `check_estimator` in the test suite; CI on three operating systems and four Python versions;
- [x] tag-driven release to PyPI through trusted publishing.

Exit criterion: an installable, tested estimator with a documented method. Met.

## M1 — Our own method (0.2.0, released)

- [x] complementary selection: greedy forward selection with replacement on the class-balanced Brier score of out-of-fold probabilities, under a budget of distinct subspaces;
- [x] exact leave-one-out votes from one neighbour query per subspace, and a default pool of 1000 candidates;
- [x] fourteen-dataset benchmark with an ablation of the two ingredients and a redundancy analysis;
- [x] documentation site with a user guide, the method and benchmark notes and an API reference;
- [x] references, starting with Brett Kennedy's ikNN article, which the method builds on.

Exit criterion: complementary selection is the default and beats ranked selection on the benchmark in every setting tested. Met: +1.8 to +2.6 points of mean macro-F1.

## M2 — Reproducibility (0.3.0)

Goal: the same data and parameters give the same model on every platform, in any row order.

The platform dependence comes from ties at the $k$-th neighbour. The fix is a tie rule that does not depend on the order in which the neighbour search returns points. For a sample whose distances to the other training points, sorted, are $d_{(1)} \le d_{(2)} \le \dots$, let $d^\ast = d_{(k)}$, let $A = \{j : d_j < d^\ast\}$ be the points strictly closer and $B = \{j : d_j = d^\ast\}$ the points tied at the boundary. Every point in $A$ gets a full vote and the points in $B$ share the remaining votes:

$$
w_j = 1 \quad (j \in A), \qquad w_j = \frac{k - |A|}{|B|} \quad (j \in B).
$$

The total weight is still $k$, and the vote no longer depends on which tied point the search happens to return first.

- [x] implement the tie rule for leave-one-out votes, with a radius query at $d^\ast$ to find every tied point ([#11](https://github.com/DiogoRibeiro7/subspaceknn/issues/11));
- [x] apply the same rule at prediction time, so that `predict_proba` and the out-of-fold votes describe the same model; decide whether `estimators_` stay `KNeighborsClassifier` objects or become a thin wrapper ([#12](https://github.com/DiogoRibeiro7/subspaceknn/issues/12));
- [x] extend the rule to `knn_weights="distance"`, where a point at distance zero already takes all the weight ([#13](https://github.com/DiogoRibeiro7/subspaceknn/issues/13));
- [x] break ties between candidates in the greedy search with a tolerance, choosing the first enumerated candidate among those within $10^{-12}$ of the best loss, so that summation order cannot flip a choice ([#14](https://github.com/DiogoRibeiro7/subspaceknn/issues/14));
- [x] fix the `selection_path_` docstring: the loss is the Brier score weighted as `balance_classes` says ([#15](https://github.com/DiogoRibeiro7/subspaceknn/issues/15));
- [x] property-based tests with Hypothesis for the invariants: weights sum to one and are multiples of $1/t^\ast$, the budget holds, `predict` is the argmax of `predict_proba`, and selection is invariant to permuting the rows ([#16](https://github.com/DiogoRibeiro7/subspaceknn/issues/16));
- [x] a CI job that fits the default model on iris, wine, breast cancer and a rounded synthetic dataset on Linux, macOS and Windows and compares `subspaces_`, `subspace_weights_` and `predict_proba` with a committed fixture ([#17](https://github.com/DiogoRibeiro7/subspaceknn/issues/17));
- [ ] a CI job with the lowest supported versions of numpy and scikit-learn (`uv sync --resolution lowest-direct`) ([#18](https://github.com/DiogoRibeiro7/subspaceknn/issues/18));
- [ ] add Python 3.14 to CI and the classifiers ([#19](https://github.com/DiogoRibeiro7/subspaceknn/issues/19)).

The tie rule changes predictions wherever ties occur, so 0.3.0 is a minor release and the changelog gives the new benchmark numbers, as the versioning policy requires.

Exit criterion: the fixture job passes on all three operating systems, selection is invariant to row order, and the benchmark numbers move by less than half a point on every dataset or the change is explained in the changelog.

## M3 — Scale and wide data (0.4.0)

Goal: fits that are fast and small enough for tens of thousands of samples, and a candidate pool that does not discard interacting features.

- [ ] reduce the overhead of the tie rule, about 1.5 times a plain scikit-learn query on the leave-one-out step for data without repeated values, to at most 1.2 times ([#59](https://github.com/DiogoRibeiro7/subspaceknn/issues/59));
- [ ] compute the built-in scores (`f1_macro`, `accuracy`, `balanced_accuracy`) directly from the out-of-fold votes with a confusion matrix, and keep the scorer adapter for everything else ([#20](https://github.com/DiogoRibeiro7/subspaceknn/issues/20));
- [ ] evaluate candidates in parallel over `n_jobs` with threads, since the neighbour queries release the GIL ([#21](https://github.com/DiogoRibeiro7/subspaceknn/issues/21));
- [ ] store votes compactly: `float32` instead of `float64`, and without the last class column, which follows from the others because every vote sums to one; for two classes that is a quarter of the current memory ([#22](https://github.com/DiogoRibeiro7/subspaceknn/issues/22));
- [ ] replace one-dimensional screening for wide data with candidate growth: score all pairs among a larger screened set, then extend the best pairs to triples, so that a feature that only matters in combination can enter ([#24](https://github.com/DiogoRibeiro7/subspaceknn/issues/24));
- [ ] a synthetic interaction benchmark (for example XOR of two features among noise features) in which one-dimensional screening fails and candidate growth succeeds ([#23](https://github.com/DiogoRibeiro7/subspaceknn/issues/23));
- [ ] a performance smoke test in CI with generous bounds, and a scheduled workflow that reruns the full benchmark and fails if a selected subspace changes unexpectedly ([#25](https://github.com/DiogoRibeiro7/subspaceknn/issues/25));
- [ ] document the time and memory model in the method note, with the measured table ([#26](https://github.com/DiogoRibeiro7/subspaceknn/issues/26)).

Targets against the baseline above, on the same machine:

| Case | 0.2.0 | Target |
| --- | ---: | ---: |
| 2,000 samples, one thread | 2.5 s | under 1.5 s |
| 50,000 samples, four threads | 49.5 s | under 20 s |
| 50,000 samples, peak memory | 367 MB | under 100 MB |

Exit criterion: the targets are met; outside the datasets where candidate growth changes the pool, macro-F1 on the benchmark moves by less than a tenth of a point on every dataset (compact storage rounds the votes, which can flip a near-tie); and the interaction benchmark shows candidate growth finding the interacting pair.

## M4 — Explanations and data (0.5.0)

Goal: explanations that show the evidence itself, and data that is not perfectly clean.

- [ ] add the neighbours to each `SubspaceVote`: training-set indices and distances of the samples behind the vote, so that an explanation can show the five flowers that decided it ([#28](https://github.com/DiogoRibeiro7/subspaceknn/issues/28));
- [ ] a feature importance that reflects the vote, the share of weight on the subspaces that contain feature $j$, $I_j = \sum_{S \ni j} w_S$, exposed as `feature_importances_`, with `feature_scores_` deprecated ([#29](https://github.com/DiogoRibeiro7/subspaceknn/issues/29));
- [ ] missing values at prediction time: a subspace that contains a missing feature abstains, the remaining weights are renormalised, and the explanation marks the abstention; decide whether fitting with missing values is in scope ([#30](https://github.com/DiogoRibeiro7/subspaceknn/issues/30));
- [ ] `plot_selection_path` for the loss after each vote, and a single-sample explanation figure combining the vote table with the panels ([#31](https://github.com/DiogoRibeiro7/subspaceknn/issues/31));
- [ ] a documented recipe for categorical features with `ColumnTransformer`, and an example showing how the choice of encoding changes the neighbourhoods ([#32](https://github.com/DiogoRibeiro7/subspaceknn/issues/32));
- [ ] report, but do not use for weighting, each subspace's out-of-fold accuracy among the explained sample's neighbours; the prototype found that using it as a weight did not improve macro-F1 consistently ([#33](https://github.com/DiogoRibeiro7/subspaceknn/issues/33)).

Exit criterion: every new field is tested against an independent computation (for example the neighbours against `KNeighborsClassifier.kneighbors`), and the user guide shows each feature on real data.

## M5 — Evidence (0.6.0)

Goal: claims about accuracy and interpretability that hold up against the natural alternatives and across many datasets.

- [ ] extend the benchmark to the numeric-feature datasets of the OpenML-CC18 suite, pinned by data id and cached ([#34](https://github.com/DiogoRibeiro7/subspaceknn/issues/34));
- [ ] add baselines: kNN with tuned `n_neighbors`, depth-limited decision trees, logistic regression, and a black-box reference such as `HistGradientBoostingClassifier`; include explainable boosting machines when the `interpret` package is available ([#35](https://github.com/DiogoRibeiro7/subspaceknn/issues/35));
- [ ] tune every model, including `n_neighbors` and `subspace_size` here, inside nested cross-validation ([#36](https://github.com/DiogoRibeiro7/subspaceknn/issues/36));
- [ ] compare methods across datasets with the Friedman test and critical-difference diagrams (Demšar, 2006), rather than counting wins ([#37](https://github.com/DiogoRibeiro7/subspaceknn/issues/37));
- [ ] measure interpretability, not only accuracy: pictures per explanation, distinct features, agreement, and the stability of the selected subspaces across bootstrap refits ([#38](https://github.com/DiogoRibeiro7/subspaceknn/issues/38));
- [ ] evaluate the probabilities: Brier score and calibration curves, with and without `CalibratedClassifierCV` ([#39](https://github.com/DiogoRibeiro7/subspaceknn/issues/39));
- [ ] rerun the benchmark on a schedule and publish the results in the documentation ([#40](https://github.com/DiogoRibeiro7/subspaceknn/issues/40)).

Exit criterion: `docs/benchmark.md` reports these comparisons with confidence intervals, and every accuracy claim in the README and the documentation comes from them.

## M6 — API freeze (0.9.0, release candidate)

Goal: a public API that can be promised for the whole 1.x series.

- [ ] settle the decisions listed below ([issues labelled decision](https://github.com/DiogoRibeiro7/subspaceknn/issues?q=label%3Adecision));
- [ ] review every public name: parameters, fitted attributes, `Explanation` and `SubspaceVote` fields, and the plotting functions ([#45](https://github.com/DiogoRibeiro7/subspaceknn/issues/45));
- [ ] warn when a parameter is set that the chosen selection ignores, such as `weighting` under complementary selection ([#46](https://github.com/DiogoRibeiro7/subspaceknn/issues/46));
- [ ] write the compatibility policy: what 1.x keeps stable (the names in `__all__`, parameter names and defaults, fitted attribute names and shapes, explanation fields) and what it does not (exact floating-point outputs across scikit-learn versions, private modules, pickles across versions) ([#47](https://github.com/DiogoRibeiro7/subspaceknn/issues/47));
- [ ] write the deprecation policy: a deprecated name warns for at least one minor release before it is removed, and removals only happen in a major release after 1.0 ([#48](https://github.com/DiogoRibeiro7/subspaceknn/issues/48));
- [ ] versioned documentation, a migration guide from 0.x, an FAQ and a gallery of worked examples on real datasets ([#49](https://github.com/DiogoRibeiro7/subspaceknn/issues/49));
- [ ] decide the documentation toolchain: MkDocs 2.0 drops the plugin system the site uses, so either stay on MkDocs 1.6 with Material, whose team now develops a successor, Zensical, or move to Zensical before the docs are versioned ([#44](https://github.com/DiogoRibeiro7/subspaceknn/issues/44));
- [ ] release 0.9.0 and keep the API unchanged for at least one release cycle while it is used ([#50](https://github.com/DiogoRibeiro7/subspaceknn/issues/50)).

Exit criterion: 0.9.0 has been out for a full cycle without an API change, and every decision below is recorded in the documentation.

## M7 — 1.0.0

- [ ] every exit criterion above is met ([#51](https://github.com/DiogoRibeiro7/subspaceknn/issues/51));
- [ ] Python support follows the published policy: every CPython version that is not end-of-life at release time, which drops 3.10 after its end of life in October 2026 ([#52](https://github.com/DiogoRibeiro7/subspaceknn/issues/52));
- [ ] `Development Status :: 5 - Production/Stable` in the package classifiers ([#53](https://github.com/DiogoRibeiro7/subspaceknn/issues/53));
- [ ] a conda-forge package ([#54](https://github.com/DiogoRibeiro7/subspaceknn/issues/54));
- [ ] `CITATION.cff` and an archived, citable release with a DOI ([#55](https://github.com/DiogoRibeiro7/subspaceknn/issues/55));
- [ ] release notes that summarise everything since 0.2.0 and link the migration guide ([#56](https://github.com/DiogoRibeiro7/subspaceknn/issues/56)).

## Decisions before the freeze

| Decision | Options | Recommendation |
| --- | --- | --- |
| Regression ([#41](https://github.com/DiogoRibeiro7/subspaceknn/issues/41)) | In 1.0, or in 1.x | 1.x. The selection carries over with squared error in place of the Brier score, but scoring, weighting and explanation need their own contracts. Keep the 1.0 names general enough not to block it. |
| `sample_weight` in `fit` ([#42](https://github.com/DiogoRibeiro7/subspaceknn/issues/42)) | Support fully, support in the selection loss only, or leave out | Support it only if neighbour votes can be weighted consistently with the loss; otherwise leave it out and say so. Half support would make the explanation disagree with the selection. |
| Parameters that apply to one selection only ([#43](https://github.com/DiogoRibeiro7/subspaceknn/issues/43)) | Keep the flat parameters and warn, or group them per selection strategy | Keep the flat parameters, which grid search handles well, and warn when an ignored one is set. |
| Missing values during `fit` ([#27](https://github.com/DiogoRibeiro7/subspaceknn/issues/27)) | Prediction only, or fitting too | Prediction only for 1.0. Fitting would need out-of-fold votes on incomplete rows, which changes the selection loss. |
| Tie rule at prediction time ([#10](https://github.com/DiogoRibeiro7/subspaceknn/issues/10)) | Own implementation, or `KNeighborsClassifier` with a correction | Decided: an own implementation, `TieSharingKNeighborsClassifier`, behind `estimators_`; see the method note. |
| Documentation toolchain ([#44](https://github.com/DiogoRibeiro7/subspaceknn/issues/44)) | MkDocs 1.6 with Material, or Zensical | Decide at M6, when versioned docs are needed and Zensical's maturity can be judged. |

## After 1.0

Ideas that are deliberately left out of 1.0: regression; per-sample subspace weights, if a scheme ever beats global weights in the benchmark; other distance metrics per subspace; and subspaces of four or more features drawn as small multiples.

## References

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
- The method's own references are listed in the [method note](https://diogoribeiro7.github.io/subspaceknn/method/#references).
