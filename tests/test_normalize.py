"""Normalization and drift correction (Notebook 02)."""
import numpy as np
import pandas as pd

from src import normalize


def _matrix(rng, n_feat=400, n_runs=12):
    base = rng.normal(15, 2, n_feat)
    m = pd.DataFrame(base[:, None] + rng.normal(0, 0.05, (n_feat, n_runs)),
                     columns=[f"r{i:02d}" for i in range(n_runs)])
    return m, base


def test_median_normalize_removes_constant_run_offsets(rng):
    m, _ = _matrix(rng)
    shifted = m + np.linspace(-1, 1, m.shape[1])           # a different loading per run
    fixed = normalize.median_normalize(shifted)
    assert fixed.median().std() < 1e-9


def test_rt_lowess_removes_retention_time_dependent_offsets(rng):
    m, _ = _matrix(rng)
    rt = pd.Series(rng.uniform(0, 6, len(m)), index=m.index)
    distorted = m.copy()
    distorted["r00"] += 0.5 * rt / 6                       # run r00 loses signal late in the gradient
    fixed = normalize.rt_lowess_normalize(distorted, rt)
    residual = fixed["r00"] - fixed.drop(columns="r00").median(axis=1)
    assert abs(np.corrcoef(residual, rt)[0, 1]) < 0.2
    assert residual.abs().median() < 0.05


def test_qc_linear_correction_removes_linear_drift(rng):
    m, _ = _matrix(rng, n_runs=20)
    injection = pd.Series(np.arange(20) * 10.0, index=m.columns)
    drifted = m - 0.002 * injection.to_numpy()             # every feature loses signal over time
    fixed = normalize.qc_linear_correction(drifted, injection, fit_runs=list(m.columns[::2]))
    slope = np.polyfit(injection, fixed.median(), 1)[0]
    assert abs(slope) < 2e-4


def test_qc_lowess_uses_rt_fallback_for_features_absent_from_qc(rng):
    m, _ = _matrix(rng, n_runs=20)
    injection = pd.Series(np.arange(20) * 10.0, index=m.columns)
    qc_runs = list(m.columns[::2])
    m.loc[m.index[:30], qc_runs] = np.nan                  # 30 features never seen in the QC
    rt = pd.Series(rng.uniform(0, 6, len(m)), index=m.index)
    _, n_fallback = normalize.qc_lowess_correction(m, injection, qc_runs, rt, min_values=5)
    assert n_fallback == 30


def test_sample_drift_correction_keeps_group_differences(rng):
    m, _ = _matrix(rng, n_runs=20)
    injection = pd.Series(rng.permutation(20) * 10.0, index=m.columns)   # randomized run order
    groups = pd.Series(["A", "B"] * 10, index=m.columns)
    m.loc[:, groups == "B"] += 1.0                         # real biology: group B is 2-fold higher
    drifted = m - 0.003 * injection.to_numpy()
    fixed, _ = normalize.sample_drift_correction(drifted, injection, groups, min_values=10)
    diff = fixed.loc[:, groups == "B"].mean(axis=1) - fixed.loc[:, groups == "A"].mean(axis=1)
    assert abs(diff.median() - 1.0) < 0.05                 # biology kept
    a_runs = groups.index[groups == "A"]
    slope = np.polyfit(injection[a_runs], fixed[a_runs].median(), 1)[0]
    assert abs(slope) < 5e-4                               # drift (0.003 per injection) removed


def test_cross_fit_never_uses_a_run_to_correct_itself():
    m = pd.DataFrame(np.zeros((3, 4)), columns=["a", "b", "c", "d"])
    seen = []

    def correct(matrix, fit_runs):
        seen.append(set(fit_runs))
        return matrix + len(fit_runs)

    out = normalize.cross_fit(correct, m, folds=[["a", "b"], ["c", "d"]])
    assert seen == [{"c", "d"}, {"a", "b"}]                # each fold fitted on the OTHER fold only
    assert (out == 2).all().all()
