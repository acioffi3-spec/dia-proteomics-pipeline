"""Differential abundance tests (Notebook 03, section 1)."""
import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from src import stats


def _null(rng, n_feat=4000, n=3, d0=4.0, s0_sq=0.05):
    true_var = d0 * s0_sq / rng.chisquare(d0, n_feat)          # scaled inverse chi-square prior
    m = pd.DataFrame(rng.normal(0, np.sqrt(true_var)[:, None], (n_feat, 2 * n)),
                     columns=[f"r{i}" for i in range(2 * n)])
    return m, m.columns[:n], m.columns[n:]


def test_welch_matches_scipy(rng):
    m, a, b = _null(rng, n_feat=200)
    ours = stats.welch_t(m, a, b)
    ref = sps.ttest_ind(m[b], m[a], axis=1, equal_var=False)
    assert np.allclose(ours["p"], ref.pvalue) and np.allclose(ours["t"], ref.statistic)


def test_bh_matches_statsmodels_and_ignores_nan(rng):
    p = pd.Series(rng.uniform(0, 1, 300))
    assert np.allclose(stats.bh_adjust(p), multipletests(p, method="fdr_bh")[1])
    p_nan = pd.concat([p, pd.Series([np.nan])], ignore_index=True)
    q = stats.bh_adjust(p_nan)
    assert q.isna().sum() == 1 and np.allclose(q.dropna(), multipletests(p, method="fdr_bh")[1])


def test_moderated_t_recovers_the_variance_prior(rng):
    m, a, b = _null(rng)
    _, (d0, s0_sq) = stats.moderated_t(m, a, b, return_prior=True)
    assert abs(d0 - 4) < 0.6 and abs(s0_sq - 0.05) < 0.005


def test_moderated_t_null_p_values_are_uniform(rng):
    m, a, b = _null(rng)
    p = stats.moderated_t(m, a, b)["p"]
    assert abs((p < 0.05).mean() - 0.05) < 0.012
    assert sps.kstest(p, "uniform").pvalue > 0.01


def test_trend_prior_fixes_intensity_dependent_noise(rng):
    # Noise grows as intensity falls; the plain moderated t over-calls low-intensity features
    intensity = rng.uniform(8, 14, 4000)
    sd = 0.03 * np.exp(-0.35 * (intensity - 14)) * np.exp(rng.normal(0, 0.3, 4000))
    m = pd.DataFrame(intensity[:, None] + rng.normal(0, 1, (4000, 6)) * sd[:, None],
                     columns=[f"r{i}" for i in range(6)])
    a, b = m.columns[:3], m.columns[3:]
    low = intensity < np.quantile(intensity, 0.25)
    plain = (stats.moderated_t(m, a, b)["p"].to_numpy()[low] < 0.05).mean()
    trend = (stats.moderated_t(m, a, b, trend=True)["p"].to_numpy()[low] < 0.05).mean()
    assert plain > 0.08 and abs(trend - 0.05) < 0.02


def test_compare_labels_features_that_cannot_be_tested():
    m = pd.DataFrame({"a1": [1.0, 1.0, np.nan], "a2": [1.1, 1.2, np.nan],
                      "b1": [2.0, np.nan, 3.0], "b2": [2.1, np.nan, 3.1]}, index=["both", "only_a", "only_b"])
    design = pd.Series(["A", "A", "B", "B"], index=m.columns)
    res = stats.compare(m, design, "A", "B", test="welch")
    assert res.loc["both", "status"] == "tested"
    assert res.loc["only_a", "status"] == "absent in B" and res.loc["only_b", "status"] == "absent in A"
    assert np.isnan(res.loc["only_a", "p"])


def test_differential_finds_a_real_change_and_reports_on_off_separately(rng):
    n_feat = 300
    base = rng.normal(15, 1.5, n_feat)
    m = pd.DataFrame(base[:, None] + rng.normal(0, 0.15, (n_feat, 6)), columns=[f"r{i}" for i in range(6)])
    m.iloc[0, 3:] += 2.0                                      # a 4-fold increase in group B
    m.iloc[1, 3:] = np.nan                                    # gone in B: an on/off protein
    design = pd.Series(["DMSO"] * 3 + ["cmpd"] * 3, index=m.columns)
    res = stats.differential(m, design, "DMSO", "cmpd")
    assert res.iloc[0]["call"] == "up" and res.iloc[0]["tier"] == "tested"
    assert res.iloc[1]["tier"] == "on/off"
    assert (res["call"] != "not significant").sum() <= 5      # few false calls among 298 unchanged
