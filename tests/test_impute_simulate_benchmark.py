"""Imputation, simulation and scoring (Notebook 03)."""
import numpy as np
import pandas as pd

from src import benchmark, impute, simulate


def _with_gaps(rng):
    m = pd.DataFrame(rng.normal(15, 2, (200, 6)), columns=[f"r{i}" for i in range(6)])
    mask = rng.random(m.shape) < 0.1
    return m.mask(mask), mask


def test_every_imputation_fills_gaps_and_keeps_observed_values(rng):
    m, mask = _with_gaps(rng)
    for name, fn in impute.STRATEGIES.items():
        filled = fn(m)
        if name == "none":
            continue
        assert not filled.isna().any().any(), name
        assert np.allclose(filled.to_numpy()[~mask], m.to_numpy()[~mask]), name


def test_downshifted_values_sit_below_each_run(rng):
    m, mask = _with_gaps(rng)
    filled = impute.downshifted_normal(m)
    for j, run in enumerate(m.columns):
        assert filled[run][mask[:, j]].mean() < m[run].mean() - 1.0


def test_random_split_gives_disjoint_groups(rng):
    a, b = simulate.random_split([f"r{i}" for i in range(20)], 5, rng)
    assert len(a) == len(b) == 5 and not set(a) & set(b)


def test_choose_truth_changes_the_requested_fraction(rng):
    truth = simulate.choose_truth([f"P{i}" for i in range(100)], 0.2, np.array([-1, 1]), rng)
    assert (truth != 0).sum() == 20 and set(truth[truth != 0]) <= {-1, 1}


def test_spike_in_shifts_group_b_and_only_loses_values_that_went_down(rng):
    prec = pd.DataFrame(rng.normal(12, 2, (60, 4)), columns=["a1", "a2", "b1", "b2"])
    protein_of = pd.Series([f"P{i // 3}" for i in range(60)], index=prec.index)
    truth = pd.Series(0.0, index=protein_of.unique())
    truth["P0"], truth["P1"] = 2.0, -4.0
    out = simulate.spike_in(prec, protein_of, truth, ["b1", "b2"], rng)
    up = protein_of.index[protein_of == "P0"]
    assert np.allclose(out.loc[up, ["b1", "b2"]], prec.loc[up, ["b1", "b2"]] + 2.0)
    assert out[["a1", "a2"]].equals(prec[["a1", "a2"]])               # group a untouched
    lost = out.isna() & prec.notna()
    assert set(protein_of[lost.any(axis=1)]) <= {"P1"}               # only the decreased protein drops out


def test_biological_noise_is_shared_by_a_proteins_precursors(rng):
    prec = pd.DataFrame(np.zeros((6, 50)))
    protein_of = pd.Series(["P1"] * 3 + ["P2"] * 3, index=prec.index)
    noisy = simulate.add_biological_noise(prec, protein_of, 0.3, rng)
    assert np.allclose(noisy.iloc[0], noisy.iloc[2])                 # same protein, same noise
    assert not np.allclose(noisy.iloc[0], noisy.iloc[3])             # different proteins differ


def test_score_counts_true_and_false_calls_exactly():
    res = pd.DataFrame({
        "p": [0.001, 0.001, 0.001, 0.5, 0.001],
        "q": [0.01, 0.01, 0.01, 0.6, 0.01],
        "log2fc": [1.0, -1.0, 0.5, 0.1, 0.8],
        "truth": [1.0, 1.0, 0.0, -1.0, 1.0],   # call 2 has the WRONG direction; call 3 is a null
        "sim": 0, "n_per_group": 3, "noise_sd": 0.0, "test": "t", "imputation": "none"})
    s = benchmark.score(res).iloc[0]
    assert s["calls / sim"] == 4
    assert np.isclose(s["empirical FDR"], 2 / 4)      # wrong-direction call + null call
    assert np.isclose(s["power"], 2 / 4)              # 2 of 4 truly changed proteins found correctly
