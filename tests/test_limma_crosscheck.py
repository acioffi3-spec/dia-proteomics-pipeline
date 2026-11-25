"""Our moderated t-test must reproduce limma (the reference implementation).

limma (Smyth 2004) is the most-cited implementation of the empirical-Bayes
moderated t-test. stats.moderated_t was written from the equations in the
paper, so agreement to ~1e-8 is strong evidence that every step is right:
the variance prior (fitFDist), the shrinkage, the degrees of freedom and,
for trend=True, the intensity-dependent prior (limma-trend).

The reference numbers were produced once by tests/data/make_limma_reference.py
(see there for why they are frozen rather than recomputed every run).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import stats

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def reference():
    x = pd.read_csv(DATA / "limma_input.csv", index_col=0)
    ref = pd.read_csv(DATA / "limma_reference.csv", index_col=0)
    runs_a = [c for c in x.columns if c.startswith("A")]
    runs_b = [c for c in x.columns if c.startswith("B")]
    return x, ref, runs_a, runs_b


@pytest.mark.parametrize("trend", [False, True], ids=["plain", "trend"])
def test_moderated_t_matches_frozen_limma(reference, trend):
    x, ref, runs_a, runs_b = reference
    tag = "trend" if trend else "plain"
    ours, (d0, _) = stats.moderated_t(x, runs_a, runs_b, return_prior=True, trend=trend)
    # Prior degrees of freedom: how much the proteins "borrow strength" from each other
    assert d0 == pytest.approx(ref[f"df_prior_{tag}"].iloc[0], rel=1e-8)
    # Every protein's t statistic and p-value (p on log scale: small p-values matter most)
    np.testing.assert_allclose(ours["t"], ref[f"t_{tag}"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.log10(ours["p"]), np.log10(ref[f"p_{tag}"]), atol=1e-8)


@pytest.mark.parametrize("trend", [False, True], ids=["plain", "trend"])
def test_moderated_t_matches_live_limma(reference, trend):
    """Same comparison against a live limma call, run only where inmoose is installed."""
    limma = pytest.importorskip("inmoose.limma")
    x, _, runs_a, runs_b = reference
    design = np.column_stack([np.ones(len(runs_a) + len(runs_b)),
                              np.r_[np.zeros(len(runs_a)), np.ones(len(runs_b))]])
    fit = limma.eBayes(limma.lmFit(x[runs_a + runs_b].to_numpy(float), design), trend=trend)
    ours = stats.moderated_t(x, runs_a, runs_b, trend=trend)
    np.testing.assert_allclose(ours["t"], np.asarray(fit.t)[:, 1], rtol=1e-8, atol=1e-10)


def test_natural_spline_is_linear_beyond_the_boundary_knots():
    """The trend must not swing wildly outside the observed intensity range."""
    knots = [0.0, 1.0, 2.0, 3.0]
    x = np.array([3.0, 4.0, 5.0, 6.0])                     # all beyond the last knot
    basis = stats.natural_spline_basis(x, knots)
    second_diff = np.diff(basis, n=2, axis=0)              # zero for straight lines
    np.testing.assert_allclose(second_diff, 0, atol=1e-12)
