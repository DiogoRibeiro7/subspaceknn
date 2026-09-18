# Benchmark

Two questions: does complementary selection beat the ikNN-style ranking of subspaces, and how much accuracy does an ensemble of drawable models give up against plain kNN in the full feature space? `benchmarks/run_benchmark.py` produces every number in this note.

## Protocol

- **Datasets.** Fourteen classification datasets, three from scikit-learn and eleven from OpenML, pinned by data id. Only numeric columns are used, which drops the one categorical column of ilpd.

  | Dataset | Source | Samples | Features | Classes |
  | --- | --- | ---: | ---: | ---: |
  | iris | scikit-learn | 150 | 4 | 3 |
  | wine | scikit-learn | 178 | 13 | 3 |
  | breast-cancer | scikit-learn | 569 | 30 | 2 |
  | diabetes | OpenML 37 | 768 | 8 | 2 |
  | banknote | OpenML 1462 | 1372 | 4 | 2 |
  | ionosphere | OpenML 59 | 351 | 34 | 2 |
  | sonar | OpenML 40 | 208 | 60 | 2 |
  | vehicle | OpenML 54 | 846 | 18 | 4 |
  | glass | OpenML 41 | 214 | 9 | 6 |
  | blood-transfusion | OpenML 1464 | 748 | 4 | 2 |
  | ilpd | OpenML 1480 | 583 | 9 | 2 |
  | climate-crashes | OpenML 1467 | 540 | 20 | 2 |
  | segment | OpenML 36 | 2310 | 19 | 7 |
  | qsar-biodeg | OpenML 1494 | 1055 | 41 | 2 |

- **Evaluation.** Macro-F1 under 5-fold stratified cross-validation repeated three times (`RepeatedStratifiedKFold`, `random_state=0`), with a `StandardScaler` fitted inside each training fold.
- **Models.**
  - *kNN*: `KNeighborsClassifier()`, five neighbours on all features.
  - *ranked (0.1.0)*: `SubspaceKNNClassifier(selection="ranked", cv=5, max_candidates=100)`, the ikNN-style ranking with the scoring and candidate cap of the first release.
  - *ranked, larger pool*: `SubspaceKNNClassifier(selection="ranked")`, the same ranking with leave-one-out scoring and up to 1000 candidates.
  - *complementary*: `SubspaceKNNClassifier()`, the defaults.
- **Settings.** Every subspace model is run with pairs and at most five subspaces (the default), pairs and at most three, and subspaces of one, two and three features with at most eight.
- **Fit time.** One fit of the complementary model with default `n_jobs` on the whole standardised dataset, on a Windows laptop. It is there to give the order of magnitude, not a performance claim.

## Summary

Mean macro-F1 over the fourteen datasets:

| Setting | kNN | ranked (0.1.0) | ranked, larger pool | complementary |
| --- | ---: | ---: | ---: | ---: |
| pairs, 5 subspaces | 0.785 | 0.762 | 0.765 | 0.780 |
| pairs, 3 subspaces | 0.785 | 0.757 | 0.763 | 0.777 |
| sizes 1-3, 8 subspaces | 0.785 | 0.772 | 0.783 | **0.798** |

Counting a dataset as won or lost when the difference exceeds half a point:

| Setting | complementary vs ranked (0.1.0) | complementary vs kNN |
| --- | --- | --- |
| pairs, 5 subspaces | +0.018; better on 9, worse on 1 | -0.006; better on 3, worse on 8 |
| pairs, 3 subspaces | +0.020; better on 10, worse on 3 | -0.009; better on 5, worse on 9 |
| sizes 1-3, 8 subspaces | +0.026; better on 11, worse on 0 | +0.012; better on 7, worse on 4 |

## Both ingredients matter

The two middle columns separate the two changes from 0.1.0. Enlarging the candidate pool while still ranking subspaces individually adds 0.3 to 1.1 points on average; choosing the subspaces jointly adds a further 1.4 to 1.5 points in every setting.

The pool matters on wide data. For pairs, the old cap of 100 candidates screened every dataset with more than 14 features down to its 14 best single features; the new cap of 1000 keeps up to 45. On sonar, whose 60 features are cut to 45 instead of 14, ranking alone gains 3.9 points from the larger pool with five pairs.

With pairs, three complementary subspaces (0.777) beat five ranked ones (0.762): fewer pictures and better predictions.

## Per-dataset results

The best score in each row is in bold. The fit column is the time of one complementary fit.

### Pairs, 5 subspaces

| Dataset | n x p | kNN | ranked (0.1.0) | ranked, larger pool | complementary | fit (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iris | 150 x 4 | 0.955 | 0.951 | 0.951 | **0.960** | 0.03 |
| wine | 178 x 13 | **0.962** | 0.947 | 0.950 | 0.951 | 0.18 |
| breast-cancer | 569 x 30 | **0.964** | 0.938 | 0.948 | 0.960 | 1.24 |
| diabetes | 768 x 8 | 0.697 | 0.690 | 0.696 | **0.701** | 0.10 |
| banknote | 1372 x 4 | **0.998** | 0.982 | 0.982 | 0.985 | 0.07 |
| ionosphere | 351 x 34 | 0.817 | 0.902 | 0.901 | **0.903** | 3.02 |
| sonar | 208 x 60 | **0.807** | 0.727 | 0.766 | 0.772 | 2.62 |
| vehicle | 846 x 18 | **0.710** | 0.633 | 0.630 | 0.653 | 1.04 |
| glass | 214 x 9 | 0.542 | 0.547 | 0.549 | **0.596** | 0.15 |
| blood-transfusion | 748 x 4 | **0.637** | 0.595 | 0.586 | 0.598 | 0.05 |
| ilpd | 583 x 9 | **0.597** | 0.574 | 0.564 | 0.565 | 0.33 |
| climate-crashes | 540 x 20 | **0.530** | 0.478 | 0.476 | 0.503 | 1.27 |
| segment | 2310 x 19 | 0.938 | 0.921 | 0.923 | **0.958** | 1.48 |
| qsar-biodeg | 1055 x 41 | **0.842** | 0.780 | 0.783 | 0.809 | 6.90 |
| **mean** | | 0.785 | 0.762 | 0.765 | 0.780 | |

### Pairs, 3 subspaces

| Dataset | n x p | kNN | ranked (0.1.0) | ranked, larger pool | complementary | fit (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iris | 150 x 4 | 0.955 | 0.953 | 0.958 | **0.962** | 0.03 |
| wine | 178 x 13 | **0.962** | 0.942 | 0.944 | 0.944 | 0.48 |
| breast-cancer | 569 x 30 | **0.964** | 0.935 | 0.946 | 0.955 | 1.11 |
| diabetes | 768 x 8 | 0.697 | 0.698 | 0.692 | **0.704** | 0.08 |
| banknote | 1372 x 4 | **0.998** | 0.970 | 0.973 | 0.984 | 0.03 |
| ionosphere | 351 x 34 | 0.817 | **0.904** | 0.895 | 0.891 | 1.99 |
| sonar | 208 x 60 | **0.807** | 0.707 | 0.758 | 0.782 | 3.48 |
| vehicle | 846 x 18 | **0.710** | 0.625 | 0.613 | 0.653 | 0.76 |
| glass | 214 x 9 | 0.542 | 0.510 | 0.523 | **0.589** | 0.11 |
| blood-transfusion | 748 x 4 | **0.637** | 0.599 | 0.583 | 0.594 | 0.03 |
| ilpd | 583 x 9 | **0.597** | 0.572 | 0.585 | 0.559 | 0.13 |
| climate-crashes | 540 x 20 | **0.530** | 0.489 | 0.499 | 0.510 | 0.67 |
| segment | 2310 x 19 | 0.938 | 0.912 | 0.924 | **0.950** | 1.11 |
| qsar-biodeg | 1055 x 41 | **0.842** | 0.779 | 0.783 | 0.797 | 4.58 |
| **mean** | | 0.785 | 0.757 | 0.763 | 0.777 | |

### Sizes 1-3, 8 subspaces

| Dataset | n x p | kNN | ranked (0.1.0) | ranked, larger pool | complementary | fit (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iris | 150 x 4 | **0.955** | 0.949 | 0.953 | 0.955 | 0.06 |
| wine | 178 x 13 | **0.962** | 0.952 | 0.952 | 0.958 | 0.84 |
| breast-cancer | 569 x 30 | 0.964 | 0.941 | 0.964 | **0.969** | 6.21 |
| diabetes | 768 x 8 | 0.697 | 0.704 | **0.709** | 0.704 | 0.61 |
| banknote | 1372 x 4 | **0.998** | 0.996 | 0.996 | 0.997 | 0.11 |
| ionosphere | 351 x 34 | 0.817 | 0.889 | 0.905 | **0.914** | 4.19 |
| sonar | 208 x 60 | **0.807** | 0.730 | 0.766 | 0.776 | 3.21 |
| vehicle | 846 x 18 | 0.710 | 0.650 | 0.691 | **0.732** | 3.55 |
| glass | 214 x 9 | 0.542 | 0.615 | 0.576 | **0.633** | 0.33 |
| blood-transfusion | 748 x 4 | **0.637** | 0.609 | 0.611 | 0.621 | 0.05 |
| ilpd | 583 x 9 | **0.597** | 0.566 | 0.570 | 0.566 | 0.38 |
| climate-crashes | 540 x 20 | 0.530 | 0.484 | 0.489 | **0.554** | 2.74 |
| segment | 2310 x 19 | 0.938 | 0.914 | 0.963 | **0.968** | 12.99 |
| qsar-biodeg | 1055 x 41 | **0.842** | 0.806 | 0.814 | 0.819 | 7.16 |
| **mean** | | 0.785 | 0.772 | 0.783 | 0.798 | |

## Redundancy of the selected pairs

With pairs and at most five subspaces, fitted on each whole standardised dataset: how many distinct features the selected pairs use, and in how many of them the most frequent feature appears.

| Dataset | ranked (0.1.0) | complementary |
| --- | --- | --- |
| iris | 4 features; top one in 3 of 5 pairs | 3 features; top one in 2 of 2 pairs |
| wine | 6 features; top one in 3 of 5 pairs | 7 features; top one in 2 of 5 pairs |
| breast-cancer | 7 features; top one in 3 of 5 pairs | 7 features; top one in 2 of 5 pairs |
| diabetes | 6 features; top one in 5 of 5 pairs | 5 features; top one in 3 of 5 pairs |
| banknote | 4 features; top one in 3 of 5 pairs | 4 features; top one in 3 of 5 pairs |
| ionosphere | 6 features; top one in 5 of 5 pairs | 7 features; top one in 3 of 5 pairs |
| sonar | 6 features; top one in 5 of 5 pairs | 9 features; top one in 2 of 5 pairs |
| vehicle | 6 features; top one in 5 of 5 pairs | 9 features; top one in 2 of 5 pairs |
| glass | 6 features; top one in 3 of 5 pairs | 7 features; top one in 2 of 5 pairs |
| blood-transfusion | 4 features; top one in 3 of 5 pairs | 4 features; top one in 2 of 4 pairs |
| ilpd | 5 features; top one in 2 of 5 pairs | 7 features; top one in 2 of 5 pairs |
| climate-crashes | 7 features; top one in 3 of 5 pairs | 4 features; top one in 3 of 3 pairs |
| segment | 7 features; top one in 4 of 5 pairs | 6 features; top one in 3 of 5 pairs |
| qsar-biodeg | 6 features; top one in 5 of 5 pairs | 8 features; top one in 2 of 5 pairs |

In five datasets out of fourteen, every one of the five ranked pairs shares the same feature, so the ensemble shows the same variable five times. Complementary selection never repeats a feature in more than three of its pairs. It does not maximise diversity for its own sake either: on iris it stops at two subspaces and on climate-crashes at three, because more would not improve the out-of-fold predictions.

## Reading the numbers

- **Where complementary selection helps most.** Glass, sonar, segment, qsar-biodeg, climate-crashes and vehicle gain two to eight points over the 0.1.0 ranking with pairs. The gain comes largely from avoiding near-duplicates, as the redundancy table below shows: on sonar, vehicle and qsar-biodeg all five ranked pairs contain the same feature, while the complementary pairs share their most common feature in at most two of five and use eight or nine distinct features instead of six. On climate-crashes the gain has another source: complementary selection keeps only three pairs, all sharing one feature, and weights them by what they add. On glass, with six unbalanced classes, the balanced loss matters as well: in the prototype that preceded this implementation, it was worth two to three points there over the plain Brier score.
- **Where it does not.** On ilpd no method reaches 0.6 and the differences are within fold noise. With three pairs, ranking is ahead on ionosphere and ilpd by just over a point, and on blood-transfusion by half a point.
- **Where subspaces beat all features.** On ionosphere every subspace ensemble is seven to ten points above plain kNN. Its 34 features probably include noisy ones that spoil distances in the full space, which low-dimensional subspaces simply leave out. Glass benefits too, though less uniformly.
- **Where plain kNN stays ahead.** On sonar, qsar-biodeg and wine the signal seems spread over many features, and five pairs cannot carry all of it; subspaces of up to three features close part of that gap. Banknote has only four features but needs more than two of them at once: pairs fall short, while triples almost match all four (0.997 against 0.998).

## Caveats

Fourteen small to medium datasets, one metric and one protocol are enough to show a consistent direction, not to rank methods in general. Differences below a point on a single dataset are within the variation between cross-validation repeats. All subspace models use five neighbours and no tuning; tuning `n_neighbors` would help plain kNN as well.

## Running it

```sh
uv run python benchmarks/run_benchmark.py          # all fourteen datasets, about twenty minutes
uv run python benchmarks/run_benchmark.py --quick  # scikit-learn's three datasets, one repeat
```

The first run downloads the OpenML datasets, which scikit-learn caches afterwards. The test suite runs a smaller, offline version of the comparison on the three scikit-learn datasets (`tests/test_benchmark.py`) and fails if the default ensemble falls more than three points behind plain kNN, or more than one point behind the 0.1.0 ranking.
