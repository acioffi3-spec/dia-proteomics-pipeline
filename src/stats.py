"""Differential abundance testing for two-group comparisons.

Design-driven: every function takes a log2 matrix (features x runs) plus a
design table that says which run belongs to which group. The same code works
for any dataset, e.g. plasma vs serum here, or DMSO vs degrader later, by
changing only the design table.

Conventions
-----------
* log2 fold change = mean(group_b) - mean(group_a), i.e. "b relative to a".
* A feature is tested only if it has at least ``min_per_group`` values in
  EACH group; otherwise its p-value is NaN (it is "not testable", which is
  different from "not significant").
"""
import warnings

import numpy as np
import pandas as pd
from scipy import special, stats


# --------------------------------------------------------------------------
# Group summaries
# --------------------------------------------------------------------------
def _group_stats(log2_matrix, runs):
    """Per-feature count, mean and sample variance (ddof=1) within one group."""
    x = log2_matrix[list(runs)].to_numpy(float)
    n = np.sum(~np.isnan(x), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(np.where(n[:, None] > 0, x, 0.0), axis=1)
        mean[n == 0] = np.nan
        var = np.nansum((x - mean[:, None]) ** 2, axis=1) / (n - 1)
    var[n < 2] = np.nan
    return n, mean, var


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def welch_t(log2_matrix, runs_a, runs_b, min_per_group=2):
    """Welch's t-test per feature (unequal variances, no information shared between features).

    With small groups each feature's variance is estimated from very few
    values, so some variances come out tiny by chance and produce huge t
    statistics. This is the weakness the moderated t-test addresses.
    """
    na, ma, va = _group_stats(log2_matrix, runs_a)
    nb, mb, vb = _group_stats(log2_matrix, runs_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        se2 = va / na + vb / nb
        t = (mb - ma) / np.sqrt(se2)
        # Welch-Satterthwaite degrees of freedom
        df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    testable = (na >= min_per_group) & (nb >= min_per_group)
    return _result(log2_matrix.index, na, nb, mb - ma, np.where(testable, t, np.nan), np.where(testable, p, np.nan))


def _trigamma_inverse(x):
    """Solve trigamma(y) = x for y (Newton iteration, as in Smyth 2004 / limma)."""
    x = np.asarray(x, float)
    y = np.where(x > 1e7, 1 / np.sqrt(x), np.where(x < 1e-6, 1 / x, 0.5 + 1 / x))
    for _ in range(50):
        tri = special.polygamma(1, y)
        step = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + step
        if np.all(np.abs(-step / y) < 1e-8):
            break
    return y


def fit_variance_prior(s2, df):
    """Estimate the prior (d0, s0^2) of feature variances, empirical-Bayes style.

    Method of moments on log variances (Smyth 2004, "fitFDist"): the spread of
    log(s^2) across features, beyond what sampling noise alone explains,
    tells how much true variances differ (small spread -> large d0, i.e.
    variances are similar and can borrow strength heavily).

    Returns
    -------
    d0 : prior degrees of freedom (how many observations the prior is "worth")
    s0_sq : prior (typical) variance
    """
    ok = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0)
    s2, df = np.asarray(s2)[ok], np.asarray(df)[ok]
    z = np.log(s2)
    e = z - special.digamma(df / 2) + np.log(df / 2)        # removes the bias of log(s^2)
    e_mean = e.mean()
    e_var = np.mean((e - e_mean) ** 2) * len(e) / (len(e) - 1) - np.mean(special.polygamma(1, df / 2))
    if e_var > 0:
        d0 = 2 * _trigamma_inverse(e_var)
        s0_sq = np.exp(e_mean + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:                                                   # variances no more spread than noise: share fully
        d0, s0_sq = np.inf, np.exp(e_mean)
    return float(d0), float(s0_sq)


def fit_variance_prior_trend(s2, df, covariate, frac=0.5):
    """Like ``fit_variance_prior``, but the prior variance depends on a covariate (intensity).

    In proteomics, low-abundance proteins are noisier. A single prior pulls
    their variances DOWN towards the typical value, which inflates their t
    statistics and produces false positives. Here the typical log-variance is
    a smooth (LOWESS) function of average log2 intensity, as in limma's
    eBayes(trend=TRUE).

    Returns
    -------
    d0 : prior degrees of freedom (one value)
    s0_sq : prior variance for each feature (array, follows its intensity)
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    s2, df, covariate = (np.asarray(v, float) for v in (s2, df, covariate))
    ok = np.isfinite(s2) & (s2 > 0) & np.isfinite(df) & (df > 0) & np.isfinite(covariate)
    e = np.log(s2[ok]) - special.digamma(df[ok] / 2) + np.log(df[ok] / 2)
    fit = lowess(e, covariate[ok], frac=frac, return_sorted=True)
    e_trend_all = np.interp(covariate, fit[:, 0], fit[:, 1])          # trend at every feature
    resid = e - e_trend_all[ok]
    e_var = np.mean(resid ** 2) * len(resid) / (len(resid) - 1) - np.mean(special.polygamma(1, df[ok] / 2))
    if e_var > 0:
        d0 = float(2 * _trigamma_inverse(e_var))
        s0_sq = np.exp(e_trend_all + special.digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0, s0_sq = np.inf, np.exp(e_trend_all)
    return d0, s0_sq


def moderated_t(log2_matrix, runs_a, runs_b, min_per_group=2, return_prior=False, trend=False):
    """Moderated t-test (limma-style empirical Bayes) per feature.

    1. Pooled variance s^2 per feature (degrees of freedom d = n_a + n_b - 2).
    2. Fit a prior distribution for variances across ALL features.
    3. Shrink each feature's variance towards the prior:
           s_mod^2 = (d0 * s0^2 + d * s^2) / (d0 + d)
       so a variance that is tiny by chance gets pulled up, and t uses d0 + d
       degrees of freedom (more information than the feature alone has).

    This is the same model as limma's two-group test with eBayes(), applied to
    proteomics data as in e.g. the limma/DEqMS/proDA family of methods.
    """
    na, ma, va = _group_stats(log2_matrix, runs_a)
    nb, mb, vb = _group_stats(log2_matrix, runs_b)
    testable = (na >= min_per_group) & (nb >= min_per_group)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (na - 1) + (nb - 1)
        s2 = ((na - 1) * va + (nb - 1) * vb) / d             # pooled variance
    if trend:
        # Prior variance follows average log2 intensity (limma eBayes(trend=TRUE))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN rows are not testable anyway
            avg = np.nanmean(log2_matrix[list(runs_a) + list(runs_b)].to_numpy(float), axis=1)
        d0, s0_trend = fit_variance_prior_trend(s2[testable], d[testable].astype(float), avg[testable])
        s0_sq = np.full_like(s2, np.nan)
        s0_sq[testable] = s0_trend
    else:
        d0, s0_sq = fit_variance_prior(s2[testable], d[testable].astype(float))
    if np.isinf(d0):
        s2_mod, df_total = np.broadcast_to(s0_sq, s2.shape).astype(float), np.full_like(s2, 1e6)
    else:
        s2_mod = (d0 * s0_sq + d * s2) / (d0 + d)
        df_total = d0 + d
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (mb - ma) / np.sqrt(s2_mod * (1 / na + 1 / nb))
    p = 2 * stats.t.sf(np.abs(t), df_total)
    res = _result(log2_matrix.index, na, nb, mb - ma, np.where(testable, t, np.nan), np.where(testable, p, np.nan))
    return (res, (d0, s0_sq)) if return_prior else res


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (q-values). NaN p-values stay NaN and are not counted."""
    p = pd.Series(p, dtype=float)
    ok = p.notna()
    q = pd.Series(np.nan, index=p.index)
    if ok.sum():
        ranked = p[ok].sort_values()
        m = len(ranked)
        adj = ranked * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1].clip(upper=1)   # enforce monotonicity
        q[adj.index] = adj
    return q


def _result(index, na, nb, log2fc, t, p):
    out = pd.DataFrame({"n_a": na, "n_b": nb, "log2fc": log2fc, "t": t, "p": p}, index=index)
    out["q"] = bh_adjust(out["p"])
    return out


# --------------------------------------------------------------------------
# One entry point, driven by a design table
# --------------------------------------------------------------------------
TESTS = {
    "welch": welch_t,
    "moderated": moderated_t,
    "moderated (trend)": lambda m, a, b, min_per_group=2: moderated_t(m, a, b, min_per_group, trend=True),
}


def compare(log2_matrix, design, group_a, group_b, test="moderated", min_per_group=2):
    """Two-group differential abundance, driven by a design table.

    Parameters
    ----------
    design : Series mapping run -> group label (e.g. "DMSO", "compound_1uM")
    group_a, group_b : labels to compare; log2fc = b - a
    test : "moderated" (recommended by Notebook 03) or "welch"

    Returns
    -------
    DataFrame per feature: n_a, n_b, log2fc, t, p, q (BH), and a "status" column:
    "tested", or "absent in a/b" when a feature is quantified in one group but
    not enough in the other (an on/off pattern, reported rather than imputed).
    """
    runs_a = design.index[design == group_a]
    runs_b = design.index[design == group_b]
    res = TESTS[test](log2_matrix, runs_a, runs_b, min_per_group=min_per_group)
    res["status"] = "tested"
    res.loc[(res["n_a"] < min_per_group) & (res["n_b"] >= min_per_group), "status"] = f"absent in {group_a}"
    res.loc[(res["n_b"] < min_per_group) & (res["n_a"] >= min_per_group), "status"] = f"absent in {group_b}"
    res.loc[(res["n_a"] < min_per_group) & (res["n_b"] < min_per_group), "status"] = "too sparse"
    return res


# --------------------------------------------------------------------------
# The recommended analysis (settings chosen in Notebook 03)
# --------------------------------------------------------------------------
RECOMMENDED = {
    "test": "moderated (trend)",   # empirical Bayes, intensity-dependent prior
    "alpha": 0.05,                 # BH-adjusted q-value cutoff
    "min_per_group": 2,            # values needed in EACH group to test without imputation
    "on_off_imputation": "down-shifted normal",   # only for proteins absent from one group
}


def differential(log2_matrix, design, group_a, group_b, alpha=0.05, min_abs_fc=0.0, seed=0):
    """Recommended two-group analysis, calibrated in Notebook 03.

    Two tiers, because the two kinds of protein need different handling:

    1. Proteins quantified in both groups (>= 2 values each): moderated t with an
       intensity-dependent prior, no imputation, BH q-values.
    2. Proteins quantified in only ONE group ("on/off"): their missing values are
       imputed from a down-shifted normal (Perseus-style), then tested the same way.
       Notebook 03 shows this keeps false on/off calls near 5%, whereas simply
       listing every protein absent from one group is wrong about half the time.

    Parameters
    ----------
    design : Series run -> group label; log2fc = group_b - group_a
    min_abs_fc : optional effect-size filter applied on top of q < alpha

    Returns
    -------
    DataFrame per protein: n_a, n_b, log2fc, p, q, tier ("tested" / "on/off" / "too sparse"),
    and call ("up" / "down" / "not significant" / "not testable").
    """
    from .impute import downshifted_normal

    runs = list(design.index)
    m = log2_matrix[runs]
    main = compare(m, design, group_a, group_b, test=RECOMMENDED["test"])
    on_off = main["status"].str.startswith("absent")

    out = main.copy()
    out["tier"] = np.where(on_off, "on/off", np.where(main["status"] == "tested", "tested", "too sparse"))
    if on_off.any():
        # Impute the whole matrix (the imputation uses each run's distribution), then test
        filled = downshifted_normal(m, seed=seed)
        imputed = compare(filled, design, group_a, group_b, test=RECOMMENDED["test"])
        cols = ["log2fc", "t", "p"]
        out.loc[on_off, cols] = imputed.loc[on_off, cols]
        # q-values within each tier, so the on/off tier does not dilute the main one
        out.loc[on_off, "q"] = bh_adjust(out.loc[on_off, "p"])
        out.loc[~on_off, "q"] = bh_adjust(out.loc[~on_off, "p"])

    significant = (out["q"] < alpha) & (out["log2fc"].abs() >= min_abs_fc)
    out["call"] = np.select([out["p"].isna(), significant & (out["log2fc"] > 0), significant & (out["log2fc"] < 0)],
                            ["not testable", "up", "down"], default="not significant")
    return out.drop(columns="status")
