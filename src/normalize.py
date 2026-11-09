"""Normalization and drift correction for precursor-level DIA data.

Every function takes a log2 matrix (precursors x runs, NaN = not quantified)
and returns a corrected log2 matrix of the same shape. Each method rests on a
different assumption; the docstrings state it, because a normalization is only
valid when its assumption holds for the data in hand.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess


def median_normalize(log2_matrix):
    """Shift each run so that its median equals the overall median.

    Assumption: runs differ only by a constant factor (e.g. how much sample was
    loaded), the same for every precursor.

    The median is taken over precursors quantified in EVERY run, so a run that
    lost its weakest precursors is not mistaken for a brighter one.
    """
    shared = log2_matrix.dropna()
    run_medians = shared.median()
    offset = run_medians - run_medians.median()     # how far each run sits above/below the middle
    return log2_matrix.sub(offset, axis=1)


def rt_lowess_normalize(log2_matrix, rt, frac=0.3, return_curves=False):
    """Retention-time-dependent normalization (one smooth curve per run).

    For each run, compute every precursor's deviation from that precursor's
    median across all runs, fit a LOWESS curve of deviation against retention
    time, and subtract the curve.

    Assumption: in any one run, MOST precursors at a given retention time are
    at their typical level, so a smooth trend in deviations is technical. It
    makes no assumption about HOW signal changes over the sequence (linear,
    stepped or otherwise), because each run is corrected on its own.

    Parameters
    ----------
    rt : Series of median retention time per precursor (same index as the matrix)
    frac : LOWESS span (fraction of points used for each local fit). Small values
        follow the data closely; large values give a smoother curve.
    """
    reference = log2_matrix.median(axis=1)           # each precursor's typical level
    rt = rt.reindex(log2_matrix.index)
    corrected = log2_matrix.copy()
    curves = {}
    for run in log2_matrix.columns:
        deviation = log2_matrix[run] - reference
        ok = deviation.notna() & rt.notna()
        fit = lowess(deviation[ok], rt[ok], frac=frac, return_sorted=True)
        curve = np.interp(rt, fit[:, 0], fit[:, 1])  # the curve evaluated at every precursor's RT
        corrected[run] = log2_matrix[run] - curve
        curves[run] = pd.Series(fit[:, 1], index=fit[:, 0])
    return (corrected, curves) if return_curves else corrected


def qc_linear_correction(log2_matrix, injection_order, fit_runs, min_values=10, alpha=0.01):
    """Per-precursor straight-line drift correction anchored on pooled-QC runs.

    This is the correction described by the data authors (Messner et al. 2020):
    for each precursor with at least ``min_values`` QC measurements, regress
    log2 quantity on injection order; if the slope is significant (p < alpha),
    subtract slope x (injection order - mean QC injection order) from all runs.

    Assumption: drift is linear in time, and the QC drifts exactly as the
    samples do.
    """
    x_all = injection_order.reindex(log2_matrix.columns).to_numpy(float)
    x_qc = injection_order.reindex(fit_runs).to_numpy(float)
    centred = x_all - x_qc.mean()
    corrected = log2_matrix.copy()
    for precursor, values in log2_matrix[fit_runs].iterrows():
        ok = values.notna().to_numpy()
        if ok.sum() < min_values:
            continue
        fit = stats.linregress(x_qc[ok], values.to_numpy()[ok])
        if fit.pvalue < alpha:
            corrected.loc[precursor] = log2_matrix.loc[precursor] - fit.slope * centred
    return corrected


def qc_lowess_correction(log2_matrix, injection_order, fit_runs, rt, frac=0.5,
                         min_values=10, n_rt_bins=10):
    """Per-precursor non-linear drift correction anchored on pooled-QC runs.

    For each precursor with enough QC values, fit a LOWESS curve of log2
    quantity against injection order on the QC runs, evaluate it at every run's
    injection order, and subtract it (centred, so the overall level is kept).

    Precursors without enough QC values (e.g. plasma-only peptides, absent from
    a serum QC pool) borrow the median correction of QC-anchored precursors in
    the same retention-time bin. This uses the finding from Notebook 01 that
    drift depends strongly on retention time.

    Assumption: the QC drifts as the samples do, but not necessarily linearly.

    Returns
    -------
    corrected matrix, number of precursors that used the RT-bin fallback
    """
    runs = log2_matrix.columns
    x_all = injection_order.reindex(runs).to_numpy(float)
    x_qc = injection_order.reindex(fit_runs).to_numpy(float)

    corrections = {}
    for precursor, values in log2_matrix[fit_runs].iterrows():
        ok = values.notna().to_numpy()
        if ok.sum() < min_values:
            continue
        fit = lowess(values.to_numpy()[ok], x_qc[ok], frac=frac, return_sorted=True)
        at_runs = np.interp(x_all, fit[:, 0], fit[:, 1])      # flat beyond the first/last QC
        at_qc = np.interp(x_qc[ok], fit[:, 0], fit[:, 1])
        corrections[precursor] = at_runs - at_qc.mean()        # centred: keeps the precursor's level
    anchored = pd.DataFrame(corrections, index=runs).T

    # Fallback: median correction of anchored precursors with similar retention time
    rt = rt.reindex(log2_matrix.index)
    rt_bin = pd.qcut(rt, n_rt_bins, labels=False)
    bin_curves = anchored.groupby(rt_bin.reindex(anchored.index)).median()
    missing = log2_matrix.index.difference(anchored.index)
    borrowed = pd.DataFrame(bin_curves.loc[rt_bin.reindex(missing)].to_numpy(), index=missing, columns=runs)

    all_corrections = pd.concat([anchored, borrowed]).reindex(log2_matrix.index)
    return log2_matrix - all_corrections, len(missing)


def sample_drift_correction(log2_matrix, injection_order, groups, fit_runs=None, frac=0.5, min_values=30):
    """Per-precursor drift correction anchored on the samples themselves.

    For each precursor: remove each sample group's median (so known biology,
    e.g. plasma vs serum, cannot be mistaken for drift), fit a LOWESS curve of
    the remaining values against injection order, and subtract that curve.

    Assumption: run order is RANDOMIZED with respect to biology, so any smooth
    trend over injection order is technical. Here that holds: plates and
    sample types are interleaved across the sequence (Notebook 01, section 3).
    In a non-randomized design this correction would remove real biology, and
    must not be used.

    Parameters
    ----------
    groups : Series, sample group per run (medians removed per group before fitting)
    fit_runs : runs used to fit the curves (default: all). Passing half the runs
        and correcting the other half allows an honest, out-of-sample score.
    min_values : precursors with fewer observed values among fit_runs are left unchanged

    Returns
    -------
    corrected matrix, number of precursors left unchanged
    """
    runs = log2_matrix.columns
    fit_runs = runs if fit_runs is None else pd.Index(fit_runs)
    groups = groups.reindex(runs)

    # Remove each group's median per precursor, so only within-group variation is left
    centred = log2_matrix.copy()
    for g in groups.unique():
        cols = groups.index[groups == g]
        centred[cols] = log2_matrix[cols].sub(log2_matrix[cols].median(axis=1), axis=0)

    x_fit = injection_order.reindex(fit_runs).to_numpy(float)
    x_all = injection_order.reindex(runs).to_numpy(float)
    corrected = log2_matrix.copy()
    skipped = 0
    for precursor, values in centred[fit_runs].iterrows():
        ok = values.notna().to_numpy()
        if ok.sum() < min_values:
            skipped += 1
            continue
        fit = lowess(values.to_numpy()[ok], x_fit[ok], frac=frac, return_sorted=True)
        corrected.loc[precursor] = log2_matrix.loc[precursor] - np.interp(x_all, fit[:, 0], fit[:, 1])
    return corrected, skipped


def cross_fit(correct, log2_matrix, folds):
    """Out-of-sample version of a correction: each run is corrected by curves fitted on the OTHER fold.

    ``correct(matrix, fit_runs)`` must return a corrected matrix for all runs.
    With two folds, every run's correction is learned without that run, so
    scores computed on the result cannot be flattered by over-fitting.
    """
    out = log2_matrix.copy()
    for k, held_out in enumerate(folds):
        fit_runs = [r for j, fold in enumerate(folds) if j != k for r in fold]
        out[list(held_out)] = correct(log2_matrix, fit_runs)[list(held_out)]
    return out
