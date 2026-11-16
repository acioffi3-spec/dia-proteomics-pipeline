"""Build a known truth into real data, so statistical choices can be graded.

Starting point: runs that are technical replicates of ONE material (the plasma
wells of PXD018874). Any two random groups of them differ by nothing, so:

1. ``random_split``: two groups with no true differences (a null comparison).
2. ``add_biological_noise``: optional extra between-sample variation, so the
   simulation is not limited to the low noise of technical replicates.
3. ``spike_in``: shift a chosen set of proteins by known log2 fold changes in
   one group, at precursor level, and let values that fall below detection
   disappear, as they would in a real experiment.

The proteins are then rolled up and tested exactly as real data would be.
"""
import numpy as np
import pandas as pd


def random_split(runs, n_per_group, rng):
    """Draw two disjoint random groups of n runs each."""
    chosen = rng.choice(np.asarray(runs), size=2 * n_per_group, replace=False)
    return list(chosen[:n_per_group]), list(chosen[n_per_group:])


def choose_truth(proteins, fraction, fold_changes, rng):
    """Assign known log2 fold changes to a random fraction of proteins (the rest stay at 0)."""
    proteins = pd.Index(proteins)
    truth = pd.Series(0.0, index=proteins)
    changed = rng.choice(proteins, size=int(round(fraction * len(proteins))), replace=False)
    truth[changed] = rng.choice(fold_changes, size=len(changed))
    return truth


def add_biological_noise(log2_precursors, protein_of, sd, rng, sd_spread=0.5):
    """Add sample-to-sample variation at PROTEIN level (shared by all precursors of a protein).

    Real biological replicates differ in how much of each protein they
    contain, so every precursor of a protein moves together in a sample.
    Proteins differ in how variable they are: each protein's SD is drawn from
    a log-normal distribution around ``sd`` (spread ``sd_spread`` on the log scale).
    """
    if sd <= 0:
        return log2_precursors
    proteins = protein_of.reindex(log2_precursors.index)
    unique = proteins.unique()
    protein_sd = sd * np.exp(rng.normal(0, sd_spread, len(unique)))
    noise = pd.DataFrame(rng.normal(0, 1, (len(unique), log2_precursors.shape[1])) * protein_sd[:, None],
                         index=unique, columns=log2_precursors.columns)
    return log2_precursors + noise.loc[proteins].to_numpy()


def spike_in(log2_precursors, protein_of, truth, runs_b, rng, floor_quantile=0.05, softness=0.3):
    """Shift precursors of changed proteins in group b by their true log2 fold change.

    Detection: a precursor pushed DOWN becomes less likely to be detected.
    Detection probability is modelled as a logistic curve centred on the run's
    ``floor_quantile`` of observed values (width ``softness`` log2 units). A
    precursor observed at level x and shifted to x' is kept with probability
    P(x') / P(x), i.e. conditional on it having been seen at x.

    Proteins pushed UP do not gain new detections (a conservative simplification).
    """
    out = log2_precursors.copy()
    shift = truth.reindex(protein_of.reindex(out.index)).fillna(0).to_numpy()
    for run in runs_b:
        before = out[run].to_numpy()
        after = before + shift
        floor = np.nanquantile(before, floor_quantile)
        p_before = 1 / (1 + np.exp(-(before - floor) / softness))
        p_after = 1 / (1 + np.exp(-(after - floor) / softness))
        keep_prob = np.clip(p_after / p_before, 0, 1)
        lost = (shift < 0) & ~np.isnan(before) & (rng.random(len(before)) > keep_prob)
        after[lost] = np.nan
        out[run] = after
    return out
