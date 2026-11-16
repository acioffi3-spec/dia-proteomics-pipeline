"""Missing-value imputation strategies, so they can be graded against known truth.

Each function takes a log2 matrix (features x runs) and returns a filled copy.
None of them is "correct" in general: each encodes an assumption about WHY a
value is missing, which is exactly what Notebook 03 tests.
"""
import numpy as np
import pandas as pd


def downshifted_normal(log2_matrix, shift=1.8, width=0.3, seed=0):
    """Perseus-style imputation: draw from a narrow normal below each run's distribution.

    Assumption: values are missing because they were below the detection limit
    (missing NOT at random). Each run's missing values are drawn from
    Normal(mean - shift*SD, width*SD) of that run's observed values.
    """
    rng = np.random.default_rng(seed)
    out = log2_matrix.copy()
    for run in out.columns:
        observed = out[run].dropna()
        miss = out[run].isna()
        out.loc[miss, run] = rng.normal(observed.mean() - shift * observed.std(), width * observed.std(), miss.sum())
    return out


def run_minimum(log2_matrix):
    """Replace missing values with the smallest observed value in that run.

    Assumption: missing = at the detection limit. Deterministic, so it creates
    many identical values, which can make variances artificially small.
    """
    return log2_matrix.fillna(log2_matrix.min())


def knn(log2_matrix, k=5):
    """K-nearest-neighbour imputation across features.

    Assumption: missing AT RANDOM. A missing value is predicted from features
    with similar profiles across runs. Appropriate for random dropouts,
    biased for values missing because they were low.
    """
    from sklearn.impute import KNNImputer

    filled = KNNImputer(n_neighbors=k).fit_transform(log2_matrix.to_numpy())
    return pd.DataFrame(filled, index=log2_matrix.index, columns=log2_matrix.columns)


STRATEGIES = {
    "none": lambda m: m,
    "down-shifted normal": downshifted_normal,
    "run minimum": run_minimum,
    "KNN": knn,
}
