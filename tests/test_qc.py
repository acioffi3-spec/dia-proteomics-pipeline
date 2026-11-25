"""QC metrics (Notebook 01)."""
import numpy as np
import pandas as pd

from src import qc


def test_robust_z_is_not_fooled_by_the_outlier_it_should_flag():
    x = pd.Series([100, 101, 99, 100, 102, 98, 100, 40])
    z_robust = qc.robust_z(x)
    z_plain = (x - x.mean()) / x.std()
    assert z_robust.iloc[-1] < -3          # flagged
    assert z_plain.iloc[-1] > -3           # a plain z-score lets the outlier hide


def test_cv_percent_on_known_values():
    m = pd.DataFrame([[10.0, 12.0, 14.0], [5.0, np.nan, np.nan]])
    cv = qc.cv_percent(m, min_fraction=0.7)
    assert list(cv.index) == [0]           # the sparse feature is not scored
    assert np.isclose(cv.iloc[0], 2 / 12 * 100)


def test_filter_cascade_counts_each_step():
    df = pd.DataFrame({"run": ["r1", "r1", "r2", "r2"], "Precursor.Id": ["a", "b", "a", "b"],
                       "Protein.Group": ["P1", "P2", "P1", "P2"], "Q.Value": [0.001, 0.02, 0.001, 0.001]})
    kept, table = qc.filter_cascade(df, [("q <= 1%", lambda d: d["Q.Value"] <= 0.01)])
    assert len(kept) == 3
    assert table.loc["as delivered", "rows"] == 4 and table.loc["q <= 1%", "rows"] == 3


def test_protein_group_support_is_per_sample_type():
    meta = pd.DataFrame({"sample_type": ["Plasma", "Plasma", "Serum"]}, index=["p1", "p2", "s1"])
    df = pd.DataFrame({"run": ["p1", "p2", "s1"], "Protein.Group": ["FGA"] * 3, "PG.Q.Value": [0.001, 0.03, 0.001]})
    support = qc.protein_group_support(df, meta, q=0.01)
    assert support.loc["FGA", "Plasma"] == 0.5 and support.loc["FGA", "Serum"] == 1.0


def test_median_signal_uses_only_features_seen_in_every_run():
    # Run b lost its weakest feature; a median over "whatever was found" would rise
    m = pd.DataFrame({"a": [2.0 ** 10, 2.0 ** 12, 2.0 ** 4], "b": [2.0 ** 10, 2.0 ** 12, np.nan]})
    medians, n_shared = qc.median_log2_signal(m)
    assert n_shared == 2 and medians["a"] == medians["b"] == 11
