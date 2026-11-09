"""Score a normalized matrix against the objectives set in Notebook 01.

The same scorecard is applied to every candidate method, so the methods are
compared on identical terms.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import qc


def scorecard(log2_matrix, meta, held_out_qc, positive_control=None, min_fraction=0.7):
    """Score one normalized log2 matrix (features x runs).

    Parameters
    ----------
    meta : sample table indexed by run (needs sample_type, injection_order, plate, well_row)
    held_out_qc : pooled-QC runs NOT used to fit any QC-anchored correction.
        Scoring QC precision only on these keeps the comparison honest.
    positive_control : optional Series mapping feature -> True for features of a
        protein known to differ between matrices (e.g. fibrinogen alpha).

    Returns
    -------
    dict of metrics
    """
    out = {}
    groups = {"QC (held out)": list(held_out_qc),
              "Plasma": list(meta.index[meta["sample_type"] == "Plasma"]),
              "Serum": list(meta.index[meta["sample_type"] == "Serum"])}

    # 1. Precision: median CV within each group of technical replicates
    for name, runs in groups.items():
        out[f"CV % {name}"] = qc.cv_percent(2 ** log2_matrix[runs], min_fraction).median()

    # 2. Residual drift: does run-level signal still track injection order?
    shared = log2_matrix.dropna()
    run_level = shared.median()
    for name in ["Plasma", "Serum"]:
        runs = groups[name]
        if run_level[runs].nunique() <= 1:      # medians made identical (e.g. median normalization)
            out[f"drift rho {name}"] = 0.0
        else:
            out[f"drift rho {name}"] = spearmanr(meta.loc[runs, "injection_order"], run_level[runs])[0]

    # 3. Structure: the largest component tied to run order, and the matrix separation
    # (top 5 PCs: a correction can also CREATE small components that track run order)
    scores, explained, _ = qc.pca_scores(log2_matrix, n_components=5)
    order = meta.loc[scores.index, "injection_order"]
    is_plasma = (meta.loc[scores.index, "sample_type"] == "Plasma").astype(float)
    out["max |rho| PC vs injection"] = max(abs(spearmanr(scores[pc], order)[0]) for pc in scores.columns)
    out["|r| PC1 vs matrix"] = abs(np.corrcoef(scores["PC1"], is_plasma)[0, 1])

    # 4. Biology preserved: plasma minus serum for the positive-control features
    if positive_control is not None:
        feats = positive_control[positive_control].index.intersection(log2_matrix.index)
        diff = (log2_matrix.loc[feats, groups["Plasma"]].median(axis=1)
                - log2_matrix.loc[feats, groups["Serum"]].median(axis=1))
        out["fibrinogen-A log2 plasma/serum"] = diff.median()

    # 5. Plate 3 row H: offset of those serum wells vs other serum wells
    plate = pd.to_numeric(meta["plate"], errors="coerce")          # robust to "3", 3 or 3.0
    row_h = meta.index[(plate == 3) & (meta["well_row"] == "H")]
    row_h = [r for r in row_h if r in run_level.index]
    other = [r for r in groups["Serum"] if r not in row_h]
    out["plate-3 row H offset (log2)"] = run_level[row_h].median() - run_level[other].median()
    return out
