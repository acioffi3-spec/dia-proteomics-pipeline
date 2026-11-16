"""Run differential-abundance simulations with a known truth and score them.

One simulation = draw two groups from technical-replicate runs, optionally add
biological noise, spike known fold changes into a random set of proteins,
roll up to proteins, then impute (or not) and test with each method.
Scoring compares the calls with the truth.
"""
import numpy as np
import pandas as pd

from . import impute, rollup, simulate, stats


def simulate_once(log2_precursors, protein_of, pool_runs, n_per_group, rng, noise_sd=0.0,
                  fraction_changed=0.2, fold_changes=(-2, -1, -0.58, 0.58, 1, 2),
                  tests=("welch", "moderated"), imputations=("none",)):
    """One simulated two-group experiment. Returns one row per protein x test x imputation.

    Parameters
    ----------
    log2_precursors : normalized precursor matrix (precursors x runs, log2)
    protein_of : Series precursor -> protein group
    pool_runs : runs that are replicates of one material (here: plasma wells)
    noise_sd : extra between-sample SD (log2) at protein level; 0 = technical noise only
    fraction_changed, fold_changes : how many proteins change, and by how much (log2)
    """
    runs_a, runs_b = simulate.random_split(pool_runs, n_per_group, rng)
    prec = log2_precursors[runs_a + runs_b].dropna(how="all")

    # The truth is assigned to proteins that can be quantified (>= 2 precursors)
    n_prec = protein_of.reindex(prec.index).value_counts()
    eligible = n_prec.index[n_prec >= 2]
    truth = simulate.choose_truth(eligible, fraction_changed, np.array(fold_changes), rng)

    prec = simulate.add_biological_noise(prec, protein_of, noise_sd, rng)
    if fraction_changed > 0:
        prec = simulate.spike_in(prec, protein_of, truth, runs_b, rng)
    proteins = rollup.rollup(prec, protein_of, rollup.maxlfq, min_ratio_count=2)
    proteins = proteins.loc[proteins.index.intersection(truth.index)]

    design = pd.Series(["a"] * n_per_group + ["b"] * n_per_group, index=runs_a + runs_b)
    rows = []
    for imp in imputations:
        filled = impute.STRATEGIES[imp](proteins) if imp != "none" else proteins
        for test in tests:
            res = stats.compare(filled, design, "a", "b", test=test)
            # Keep the ORIGINAL status too: imputation can make an on/off protein "testable",
            # and those proteins are exactly where imputation choices matter
            res["status_original"] = stats.compare(proteins, design, "a", "b", test="welch")["status"]
            res["truth"] = truth.reindex(res.index).to_numpy()
            res["test"], res["imputation"] = test, imp
            rows.append(res)
    return pd.concat(rows).rename_axis("protein").reset_index()


def run_grid(log2_precursors, protein_of, pool_runs, n_sims, seed=0,
             tests=("welch", "moderated"), imputations=("none",),
             fold_changes=(-2, -1, -0.58, 0.58, 1, 2), **conditions):
    """Repeat ``simulate_once`` over every combination of the given conditions.

    tests, imputations : fixed for every simulation (each simulated dataset is
        analysed with all of them, so they are compared on identical data)
    fold_changes : the set of true log2 fold changes to draw from (fixed)
    conditions : lists for any of n_per_group, noise_sd, fraction_changed, e.g.
        run_grid(..., n_per_group=[3, 5], noise_sd=[0, 0.25])
    """
    rng = np.random.default_rng(seed)
    keys = list(conditions)
    grid = pd.MultiIndex.from_product([conditions[k] for k in keys], names=keys).to_frame(index=False)
    out = []
    for _, cond in grid.iterrows():
        kw = cond.to_dict()
        if "n_per_group" in kw:
            kw["n_per_group"] = int(kw["n_per_group"])
        for sim in range(n_sims):
            res = simulate_once(log2_precursors, protein_of, pool_runs, rng=rng,
                                tests=tests, imputations=imputations, fold_changes=fold_changes, **kw)
            for k, v in cond.items():
                res[k] = v
            res["sim"] = sim
            out.append(res)
    return pd.concat(out, ignore_index=True)


def score(results, alpha=0.05, by=("n_per_group", "noise_sd", "test", "imputation"), min_abs_fc=0.0):
    """Empirical FDR and power at a q-value cutoff, pooled over simulations.

    A call = q < alpha (and |log2fc| >= min_abs_fc, if given).
    Empirical FDR = false calls / all calls, pooled across simulations
    (pooling avoids noisy per-simulation ratios when there are few calls).
    Power = true calls / truly changed proteins that were testable.
    """
    r = results[results["p"].notna()].copy()        # everything this variant could test
    r["called"] = (r["q"] < alpha) & (r["log2fc"].abs() >= min_abs_fc)
    r["changed"] = r["truth"] != 0
    # A call on a changed protein only counts as true if the direction is right
    r["true_call"] = r["called"] & r["changed"] & (np.sign(r["log2fc"]) == np.sign(r["truth"]))
    r["false_call"] = r["called"] & ~r["true_call"]
    g = r.groupby(list(by))
    out = pd.DataFrame({
        "calls / sim": g["called"].sum() / g["sim"].nunique(),
        "empirical FDR": g["false_call"].sum() / g["called"].sum().replace(0, np.nan),
        "power": g["true_call"].sum() / g["changed"].sum(),
    })
    return out


def fc_accuracy(results, by=("n_per_group", "noise_sd", "imputation")):
    """Estimated minus true log2 fold change, per true fold change (testable proteins, one test)."""
    r = results[results["p"].notna() & (results["test"] == results["test"].iloc[0])].copy()
    r["error"] = r["log2fc"] - r["truth"]
    return r.groupby(list(by) + ["truth"])["error"].agg(["median", "std", "count"]).rename(
        columns={"median": "median error", "std": "SD of error", "count": "n"})
