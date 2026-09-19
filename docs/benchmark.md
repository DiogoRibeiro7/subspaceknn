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
    - *ranked (0.1.0)*: `SubspaceKNNClassifier(selection="ranked", cv=5, max_candidates=100)`, the ikNN-style ranking with the scoring and candidate cap of the first release, and the current tie rule.
    - *ranked, larger pool*: `SubspaceKNNClassifier(selection="ranked")`, the same ranking with leave-one-out scoring and up to 1000 candidates.
    - *complementary*: `SubspaceKNNClassifier()`, the defaults.
- **Settings.** Every subspace model is run with pairs and at most five subspaces (the default), pairs and at most three, and subspaces of one, two and three features with at most eight.
- **Fit time.** One fit of the complementary model with default `n_jobs` on the whole standardised dataset, on a Windows laptop. It is there to give the order of magnitude, not a performance claim: on this machine, repeated runs of the same fit differ by up to a factor of two.

## Summary

Mean macro-F1 over the fourteen datasets:

| Setting | kNN | ranked (0.1.0) | ranked, larger pool | complementary |
| --- | ---: | ---: | ---: | ---: |
| pairs, 5 subspaces | 0.785 | 0.760 | 0.764 | 0.777 |
| pairs, 3 subspaces | 0.785 | 0.755 | 0.761 | 0.774 |
| sizes 1-3, 8 subspaces | 0.785 | 0.772 | 0.782 | **0.795** |

Counting a dataset as won or lost when the difference exceeds half a point:

| Setting | complementary vs ranked (0.1.0) | complementary vs kNN |
| --- | --- | --- |
| pairs, 5 subspaces | +0.017; better on 10, worse on 1 | -0.008; better on 3, worse on 8 |
| pairs, 3 subspaces | +0.018; better on 9, worse on 3 | -0.012; better on 3, worse on 9 |
| sizes 1-3, 8 subspaces | +0.024; better on 10, worse on 1 | +0.010; better on 7, worse on 4 |

## Both ingredients matter

The two middle columns separate the two changes from 0.1.0. Enlarging the candidate pool while still ranking subspaces individually adds 0.4 to 1.0 points on average; choosing the subspaces jointly adds a further 1.3 points in every setting.

The pool matters on wide data. For pairs, the old cap of 100 candidates screened every dataset with more than 14 features down to its 14 best single features; the new cap of 1000 keeps up to 45. On sonar, whose 60 features are cut to 45 instead of 14, ranking alone gains 3.8 points from the larger pool with five pairs.

With pairs, three complementary subspaces (0.774) beat five ranked ones (0.760): fewer pictures and better predictions.

## Per-dataset results

The best score in each row is in bold. The fit column is the time of one complementary fit.

### Pairs, 5 subspaces

| Dataset | n x p | kNN | ranked (0.1.0) | ranked, larger pool | complementary | fit (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iris | 150 x 4 | 0.955 | 0.951 | 0.951 | **0.960** | 0.04 |
| wine | 178 x 13 | **0.962** | 0.947 | 0.950 | 0.951 | 0.73 |
| breast-cancer | 569 x 30 | **0.964** | 0.938 | 0.948 | 0.960 | 3.19 |
| diabetes | 768 x 8 | 0.697 | 0.696 | 0.699 | **0.702** | 0.18 |
| banknote | 1372 x 4 | **0.998** | 0.982 | 0.982 | 0.985 | 0.05 |
| ionosphere | 351 x 34 | 0.817 | 0.902 | **0.905** | 0.904 | 5.11 |
| sonar | 208 x 60 | **0.807** | 0.728 | 0.766 | 0.772 | 8.12 |
| vehicle | 846 x 18 | **0.710** | 0.629 | 0.628 | 0.650 | 2.11 |
| glass | 214 x 9 | 0.542 | 0.539 | 0.556 | **0.600** | 0.54 |
| blood-transfusion | 748 x 4 | **0.637** | 0.568 | 0.570 | 0.579 | 0.11 |
| ilpd | 583 x 9 | **0.597** | 0.564 | 0.553 | 0.552 | 0.30 |
| climate-crashes | 540 x 20 | **0.530** | 0.477 | 0.476 | 0.503 | 0.75 |
| segment | 2310 x 19 | 0.938 | 0.926 | 0.929 | **0.959** | 2.56 |
| qsar-biodeg | 1055 x 41 | **0.842** | 0.791 | 0.789 | 0.801 | 5.73 |
| **mean** | | 0.785 | 0.760 | 0.764 | 0.777 | |

### Pairs, 3 subspaces

| Dataset | n x p | kNN | ranked (0.1.0) | ranked, larger pool | complementary | fit (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iris | 150 x 4 | 0.955 | 0.953 | 0.955 | **0.960** | 0.04 |
| wine | 178 x 13 | **0.962** | 0.942 | 0.944 | 0.944 | 0.60 |
| breast-cancer | 569 x 30 | **0.964** | 0.935 | 0.946 | 0.955 | 1.81 |
| diabetes | 768 x 8 | 0.697 | 0.692 | 0.690 | **0.701** | 0.20 |
| banknote | 1372 x 4 | **0.998** | 0.970 | 0.973 | 0.984 | 0.06 |
| ionosphere | 351 x 34 | 0.817 | **0.906** | 0.894 | 0.894 | 6.08 |
| sonar | 208 x 60 | **0.807** | 0.701 | 0.758 | 0.782 | 3.65 |
| vehicle | 846 x 18 | **0.710** | 0.625 | 0.630 | 0.640 | 2.21 |
| glass | 214 x 9 | 0.542 | 0.502 | 0.525 | **0.588** | 0.24 |
| blood-transfusion | 748 x 4 | **0.637** | 0.588 | 0.574 | 0.579 | 0.05 |
| ilpd | 583 x 9 | **0.597** | 0.575 | 0.563 | 0.554 | 0.35 |
| climate-crashes | 540 x 20 | **0.530** | 0.489 | 0.499 | 0.510 | 0.77 |
| segment | 2310 x 19 | 0.938 | 0.917 | 0.922 | **0.951** | 2.09 |
| qsar-biodeg | 1055 x 41 | **0.842** | 0.783 | 0.783 | 0.787 | 7.92 |
| **mean** | | 0.785 | 0.755 | 0.761 | 0.774 | |

### Sizes 1-3, 8 subspaces

| Dataset | n x p | kNN | ranked (0.1.0) | ranked, larger pool | complementary | fit (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| iris | 150 x 4 | **0.955** | 0.949 | 0.949 | 0.955 | 0.17 |
| wine | 178 x 13 | **0.962** | 0.955 | 0.952 | 0.958 | 1.01 |
| breast-cancer | 569 x 30 | 0.964 | 0.941 | 0.964 | **0.969** | 7.63 |
| diabetes | 768 x 8 | 0.697 | 0.705 | 0.705 | **0.710** | 1.26 |
| banknote | 1372 x 4 | **0.998** | 0.996 | 0.996 | 0.997 | 0.25 |
| ionosphere | 351 x 34 | 0.817 | 0.880 | 0.906 | **0.912** | 5.62 |
| sonar | 208 x 60 | **0.807** | 0.729 | 0.760 | 0.773 | 3.74 |
| vehicle | 846 x 18 | 0.710 | 0.650 | 0.694 | **0.728** | 12.12 |
| glass | 214 x 9 | 0.542 | 0.615 | 0.578 | **0.624** | 0.47 |
| blood-transfusion | 748 x 4 | **0.637** | 0.596 | 0.595 | 0.599 | 0.09 |
| ilpd | 583 x 9 | **0.597** | 0.587 | 0.586 | 0.570 | 0.96 |
| climate-crashes | 540 x 20 | 0.530 | 0.484 | 0.489 | **0.554** | 5.45 |
| segment | 2310 x 19 | 0.938 | 0.915 | 0.963 | **0.967** | 29.86 |
| qsar-biodeg | 1055 x 41 | **0.842** | 0.801 | 0.817 | 0.817 | 21.31 |
| **mean** | | 0.785 | 0.772 | 0.782 | 0.795 | |

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
| sonar | 6 features; top one in 3 of 5 pairs | 9 features; top one in 2 of 5 pairs |
| vehicle | 6 features; top one in 4 of 5 pairs | 9 features; top one in 2 of 5 pairs |
| glass | 6 features; top one in 3 of 5 pairs | 6 features; top one in 2 of 5 pairs |
| blood-transfusion | 4 features; top one in 3 of 5 pairs | 4 features; top one in 2 of 3 pairs |
| ilpd | 5 features; top one in 2 of 5 pairs | 7 features; top one in 2 of 5 pairs |
| climate-crashes | 8 features; top one in 2 of 5 pairs | 4 features; top one in 3 of 3 pairs |
| segment | 7 features; top one in 4 of 5 pairs | 5 features; top one in 3 of 5 pairs |
| qsar-biodeg | 6 features; top one in 5 of 5 pairs | 9 features; top one in 2 of 5 pairs |

In three datasets out of fourteen every one of the five ranked pairs shares the same feature, and in two more four of them do, so the ensemble shows the same variable four or five times. Complementary selection never repeats a feature in more than three of its pairs. It does not maximise diversity for its own sake either: on iris it stops at two subspaces, and on blood-transfusion and climate-crashes at three, because more would not improve the out-of-fold predictions.

## Reading the numbers

- **Where complementary selection helps most.** Glass, sonar, segment, climate-crashes and vehicle gain 1.5 to 8.6 points over the 0.1.0 ranking with pairs. The gain comes largely from avoiding near-duplicates, as the redundancy table below shows: on qsar-biodeg all five ranked pairs contain the same feature and on vehicle four do, while the complementary pairs of sonar, vehicle and qsar-biodeg share their most common feature in at most two of five and use nine distinct features instead of six. On climate-crashes the gain has another source: complementary selection keeps only three pairs, all sharing one feature, and weights them by what they add. On glass, with six unbalanced classes, the balanced loss matters as well: in the prototype that preceded this implementation, it was worth two to three points there over the plain Brier score.
- **Where it does not.** On ilpd no method reaches 0.6, and ranking is ahead by 1.2 to 2.1 points in every setting. With three pairs, ranking is also ahead on ionosphere, by 1.2 points, and on blood-transfusion, by 0.9.
- **Where subspaces beat all features.** On ionosphere every subspace ensemble is six to ten points above plain kNN. Its 34 features probably include noisy ones that spoil distances in the full space, which low-dimensional subspaces simply leave out. Glass benefits too, though less uniformly.
- **Effect of the tie rule.** Since 0.3.0, points tied at the distance of the k-th neighbour share its votes instead of the neighbour search picking some of them. Compared with 0.2.0, this lowers mean macro-F1 by 0.3 points in every setting, almost all of it on datasets with many repeated values and imbalanced classes: with five pairs, blood-transfusion moves from 0.598 to 0.579, ilpd from 0.565 to 0.552 and qsar-biodeg from 0.809 to 0.801. A shared vote is the average over every way of breaking the tie, which leans towards the majority class, where an arbitrary pick sometimes landed on the minority. The benefit is a model that is the same on every platform and for any row order.
- **Where plain kNN stays ahead.** On sonar, qsar-biodeg and wine the signal seems spread over many features, and five pairs cannot carry all of it; subspaces of up to three features close part of that gap. Banknote has only four features but needs more than two of them at once: pairs fall short, while triples almost match all four (0.997 against 0.998).

## Caveats

Fourteen small to medium datasets, one metric and one protocol are enough to show a consistent direction, not to rank methods in general. Differences below a point on a single dataset are within the variation between cross-validation repeats. All subspace models use five neighbours and no tuning; tuning `n_neighbors` would help plain kNN as well.

## Running it

```sh
uv run python benchmarks/run_benchmark.py          # all fourteen datasets, about twenty minutes
uv run python benchmarks/run_benchmark.py --quick  # scikit-learn's three datasets, one repeat
```

The first run downloads the OpenML datasets, which scikit-learn caches afterwards. The test suite runs a smaller, offline version of the comparison on the three scikit-learn datasets (`tests/test_benchmark.py`) and fails if the default ensemble falls more than three points behind plain kNN, or more than one point behind the 0.1.0 ranking.
