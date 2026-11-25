"""Protein roll-up (Notebook 02, section 6)."""
import numpy as np
import pandas as pd

from src import rollup


def _protein(rng, n_prec=8, n_runs=12, missing=0.3):
    truth = rng.normal(20, 1, n_runs)
    x = truth[None, :] + rng.normal(0, 2, n_prec)[:, None] + rng.normal(0, 0.05, (n_prec, n_runs))
    x[rng.random(x.shape) < missing] = np.nan
    return pd.DataFrame(x), truth


def _relative_error(est, truth):
    ok = ~np.isnan(est)
    err = (est[ok] - est[ok].mean()) - (truth[ok] - truth[ok].mean())
    return np.sqrt(np.mean(err ** 2))


def test_maxlfq_recovers_relative_levels_despite_missing_values(rng):
    x, truth = _protein(rng)
    assert _relative_error(rollup.maxlfq(x), truth) < 0.1      # noise is 0.05


def test_median_polish_recovers_relative_levels(rng):
    x, truth = _protein(rng)
    assert _relative_error(rollup.median_polish(x), truth) < 0.1


def test_top_n_sum_is_biased_by_missing_values(rng):
    # The documented weakness: a missing top precursor drags the sum down
    x, truth = _protein(rng)
    assert _relative_error(rollup.top_n_sum(x), truth) > 0.3


def test_maxlfq_refuses_a_ratio_built_on_one_shared_precursor():
    # Run 2 shares only ONE precursor with run 1 (the FGG case in Notebook 02)
    x = pd.DataFrame([[20.0, 20.0], [18.0, np.nan], [16.0, np.nan]])
    assert np.isnan(rollup.maxlfq(x, min_ratio_count=2)).all()
    assert not np.isnan(rollup.maxlfq(x, min_ratio_count=1)).any()


def test_rollup_applies_a_method_to_every_protein(rng):
    a, _ = _protein(rng)
    b, _ = _protein(rng)
    m = pd.concat([a, b], ignore_index=True)
    protein_of = pd.Series(["P1"] * len(a) + ["P2"] * len(b), index=m.index)
    out = rollup.rollup(m, protein_of, rollup.maxlfq)
    assert list(out.index) == ["P1", "P2"] and out.shape[1] == m.shape[1]
