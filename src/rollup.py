"""Protein roll-up: combine a protein's precursor quantities into one value per run.

All functions take one protein's log2 matrix (its precursors x runs, NaN = not
quantified) and return a 1-D array of log2 protein quantities, one per run.
``rollup`` applies a method to every protein.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components


def top_n_sum(precursors, n=3):
    """Sum of the n most intense precursors (chosen once, by mean intensity across runs).

    Simple, but biased: in a run where one of the top precursors is missing,
    the sum drops even if the protein did not change.
    """
    top = precursors.mean(axis=1).nlargest(n).index
    total = (2 ** precursors.loc[top]).sum(min_count=1)      # NaN only if all n are missing
    return np.log2(total).to_numpy()


def median_polish(precursors, n_iter=10):
    """Tukey median polish: log2 value = overall + precursor effect + run effect.

    The run effect (plus the overall level) is the protein quantity. Medians
    make it robust to a few aberrant precursors.
    """
    resid = precursors.to_numpy(float).copy()
    run_effect = np.zeros(resid.shape[1])
    row_effect = np.zeros(resid.shape[0])
    with warnings.catch_warnings():
        # All-NaN rows/columns are expected (a run without values); silence numpy's warning
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_iter):
            r = np.nan_to_num(np.nanmedian(resid, axis=1)); resid -= r[:, None]; row_effect += r
            c = np.nan_to_num(np.nanmedian(resid, axis=0)); resid -= c[None, :]; run_effect += c
    out = np.median(row_effect) + run_effect
    out[np.all(np.isnan(precursors.to_numpy()), axis=0)] = np.nan
    return out


def maxlfq(precursors, min_ratio_count=2):
    """MaxLFQ protein quantification (Cox et al., Mol Cell Proteomics 2014).

    1. For every pair of runs, the protein ratio is the MEDIAN of the log2
       ratios of precursors quantified in both runs, kept only if at least
       ``min_ratio_count`` precursors are shared (the original default is 2).
    2. Find one value per run whose pairwise differences best match all those
       ratios (least squares).
    3. Put the result on the scale of the summed precursor intensities.

    Why it helps: each pairwise ratio uses only precursors seen in BOTH runs,
    so a precursor missing from one run cannot pull that run's value down
    (the weakness of summing).

    Why ``min_ratio_count`` matters: with 1, a protein seen through a single
    weak precursor in one run is tied to that precursor alone. Notebook 02
    shows this turning "absent in serum" into "equal in serum" for FGG.
    """
    a = precursors.to_numpy(float)
    n_runs = a.shape[1]
    observed = ~np.isnan(a)
    shared = observed.T.astype(int) @ observed.astype(int)      # precursors in common, per pair of runs

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                    # runs pairs with no shared values
        pair_ratio = np.nanmedian(a[:, :, None] - a[:, None, :], axis=0)   # runs x runs
    pair_ratio[shared < min_ratio_count] = np.nan

    weight = (~np.isnan(pair_ratio)).astype(float)
    np.fill_diagonal(weight, 0)
    pair_ratio = np.nan_to_num(pair_ratio)
    degree = weight.sum(axis=1)

    out = np.full(n_runs, np.nan)
    linked = np.where(degree > 0)[0]
    if len(linked) == 0:
        return out

    # Least squares: minimise sum over pairs of (ratio_ij - (x_i - x_j))^2.
    # Its normal equations are L x = b, with L the graph Laplacian of the run pairs.
    laplacian = np.diag(degree) - weight
    b = (weight * pair_ratio).sum(axis=1)
    with np.errstate(divide="ignore"):                                     # runs with no values give log2(0)
        log_sum = np.log2(np.nansum(2 ** a, axis=0))

    # Runs that share no valid ratio with each other form separate groups;
    # each group is solved on its own (their relative scale is unknown).
    n_groups, group = connected_components(weight[np.ix_(linked, linked)], directed=False)
    for k in range(n_groups):
        idx = linked[group == k]
        system = np.vstack([laplacian[np.ix_(idx, idx)], np.ones(len(idx))])   # + constraint: sum = 0
        solution = np.linalg.lstsq(system, np.r_[b[idx], 0.0], rcond=None)[0]
        out[idx] = solution - solution.mean() + log_sum[idx].mean()          # back to intensity scale
    return out


def rollup(log2_matrix, protein_of, method, **kwargs):
    """Apply a roll-up method to every protein.

    Parameters
    ----------
    log2_matrix : precursors x runs (log2)
    protein_of : Series mapping each precursor to its protein group
    method : one of the functions above (top_n_sum, median_polish, maxlfq)

    Returns
    -------
    proteins x runs (log2); proteins with no value in any run are dropped
    """
    protein_of = protein_of.reindex(log2_matrix.index)
    result = {protein: method(log2_matrix.loc[members], **kwargs)
              for protein, members in protein_of.groupby(protein_of).groups.items()}
    out = pd.DataFrame(result, index=log2_matrix.columns).T
    return out.dropna(how="all")
