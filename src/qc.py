"""Quality-control metrics for DIA-NN reports.

Conventions used throughout
---------------------------
* ``df``      long report (one row per precursor per run) with a ``run`` column.
* ``meta``    sample table indexed by run (from ``metadata.parse_run_names``).
* ``matrix``  wide table: features (rows) x runs (columns), NaN = not quantified.
* Quantities stay on the linear scale until a function says it takes log2.
"""
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

MAD_TO_SD = 1.4826  # scales the median absolute deviation to an SD for normal data


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def filter_cascade(df, steps):
    """Apply filters one after another and record what each one removes.

    Parameters
    ----------
    steps : list of (name, function)
        Each function takes the current DataFrame and returns a boolean mask of
        rows to KEEP. Order matters: each step sees the output of the previous one.

    Returns
    -------
    filtered DataFrame, summary table (one row per step)
    """
    def snapshot(name, d):
        per_run = d.groupby("run", observed=True)["Precursor.Id"].nunique()
        return {"step": name,
                "rows": len(d),
                "unique precursors": d["Precursor.Id"].nunique(),
                "unique protein groups": d["Protein.Group"].nunique(),
                "median precursors / run": int(per_run.median())}

    records = [snapshot("as delivered", df)]
    for name, keep in steps:
        df = df[keep(df)]
        records.append(snapshot(name, df))
    return df, pd.DataFrame(records).set_index("step")


def protein_group_support(df, meta, q=0.01):
    """Fraction of runs, per sample type, where each protein group has PG.Q.Value <= q.

    Used to build a *global* protein-group filter from run-specific q-values:
    a protein confidently identified in many runs of a matrix is very unlikely
    to be a false identification, even if it misses the cut-off in some runs.
    """
    pg = df.drop_duplicates(["run", "Protein.Group"])[["run", "Protein.Group", "PG.Q.Value"]]
    pg = pg.assign(sample_type=pg["run"].astype(str).map(meta["sample_type"]),
                   confident=pg["PG.Q.Value"] <= q)
    n_runs = meta["sample_type"].value_counts()
    counts = pg.groupby(["Protein.Group", "sample_type"])["confident"].sum().unstack(fill_value=0)
    return counts / n_runs[counts.columns]


# --------------------------------------------------------------------------
# Reshaping
# --------------------------------------------------------------------------
def to_matrix(df, value="Precursor.Quantity", feature="Precursor.Id"):
    """Long report -> wide matrix (features x runs). Missing = NaN.

    For protein-level columns (PG.*), pass feature="Protein.Group"; values are
    identical across a protein's precursor rows, so we take the first.
    """
    sub = df[[feature, "run", value]].drop_duplicates([feature, "run"])
    sub = sub.assign(run=sub["run"].astype(str))   # plain strings: easy column selection
    return sub.pivot(index=feature, columns="run", values=value)


# --------------------------------------------------------------------------
# Per-run metrics
# --------------------------------------------------------------------------
def ids_per_run(df):
    """Number of precursors and protein groups quantified in each run."""
    g = df.groupby("run", observed=True)
    out = pd.DataFrame({"precursors": g["Precursor.Id"].nunique(),
                        "protein_groups": g["Protein.Group"].nunique()})
    out.index = out.index.astype(str)
    return out


def robust_z(x):
    """(x - median) / (1.4826 * MAD): a z-score that a few outliers cannot distort.

    A plain z-score uses the mean and SD, which the outliers themselves inflate,
    so extreme runs can hide. Median and MAD barely move when a few runs fail.
    """
    x = pd.Series(x, dtype=float)
    mad = (x - x.median()).abs().median() * MAD_TO_SD
    return (x - x.median()) / mad


def median_log2_signal(matrix):
    """Median log2 quantity per run, using only features quantified in EVERY run.

    Restricting to shared features matters: if a run loses its weakest
    precursors, its median over 'whatever was found' goes UP, hiding a loss of
    sensitivity. A fixed feature set makes runs comparable.
    """
    shared = matrix.dropna()
    return np.log2(shared).median(), len(shared)


def rt_shift_per_run(df):
    """Retention-time shift of each run relative to the experiment, in seconds.

    For every precursor, RT deviation = RT in this run - median RT across runs.
    The median deviation within a run is its overall shift; the IQR of
    deviations within the run tells you whether the shift is rigid (all
    peptides move together: an LC delay, easy for DIA-NN to calibrate) or
    distorted (gradient shape changed: harder).
    """
    rt = to_matrix(df, value="RT")
    dev = rt.sub(rt.median(axis=1), axis=0) * 60
    return pd.DataFrame({"rt_shift_s": dev.median(),
                         "rt_dev_iqr_s": dev.quantile(0.75) - dev.quantile(0.25)})


def correlation_to_reference(log2_matrix, meta):
    """Pearson r of each run against the median profile of its own sample type."""
    out = {}
    for stype, runs in meta.groupby("sample_type").groups.items():
        sub = log2_matrix[list(runs)].dropna()
        reference = sub.median(axis=1)
        for run in runs:
            out[run] = np.corrcoef(sub[run], reference)[0, 1]
    return pd.Series(out, name="r_to_reference")


# --------------------------------------------------------------------------
# Precision and missingness
# --------------------------------------------------------------------------
def cv_percent(matrix, min_fraction=0.7):
    """Coefficient of variation (SD / mean x 100) per feature, on the LINEAR scale.

    Only features quantified in at least ``min_fraction`` of the runs are used,
    so that a CV is never computed from two lucky values.
    """
    keep = matrix.notna().mean(axis=1) >= min_fraction
    m = matrix[keep]
    return (m.std(axis=1, ddof=1) / m.mean(axis=1) * 100).rename("cv_percent")


def missingness_by_intensity(matrix, n_bins=10):
    """Fraction missing per feature, grouped by the feature's mean log2 intensity.

    If missingness falls steeply as intensity rises, values are mostly missing
    because they were below the detection limit (missing NOT at random, MNAR).
    A flat profile means random misses (MCAR). The two call for different
    imputation strategies, which is why we look before we impute.
    """
    mean_log2 = np.log2(matrix).mean(axis=1)
    frac_missing = matrix.isna().mean(axis=1)
    bins = pd.qcut(mean_log2, n_bins, labels=False) + 1
    return pd.DataFrame({
        "mean_log2_intensity": mean_log2.groupby(bins).median(),
        "fraction_missing": frac_missing.groupby(bins).mean(),
        "n_features": frac_missing.groupby(bins).size(),
    }).rename_axis("intensity_decile")


# --------------------------------------------------------------------------
# Drift and structure
# --------------------------------------------------------------------------
def lowess_trend(x, y, frac=0.5):
    """Smooth trend of y over x (LOWESS). Returns fitted y at each x, in x order."""
    fit = lowess(np.asarray(y, float), np.asarray(x, float), frac=frac, return_sorted=True)
    return fit[:, 0], fit[:, 1]


def per_feature_drift(log2_matrix, injection_order, min_values=30, step_at=None):
    """Straight-line drift of each feature against injection order.

    Parameters
    ----------
    log2_matrix : features x runs (log2), usually the pooled-QC runs only
    injection_order : injection number for each column, same order
    min_values : skip features with fewer observed values than this
    step_at : optional injection number; if given, also test whether adding a
        step (a different intercept after this injection) fits significantly
        better than the straight line alone (nested-model F-test).

    Returns
    -------
    DataFrame with slope (log2 per injection), slope p-value and, if requested,
    the p-value for the step term.
    """
    from scipy import stats

    x_all = np.asarray(injection_order, dtype=float)
    rows = {}
    for feature, values in log2_matrix.iterrows():
        ok = values.notna().to_numpy()
        if ok.sum() < min_values:
            continue
        x, y = x_all[ok], values.to_numpy()[ok]
        fit = stats.linregress(x, y)
        row = {"slope": fit.slope, "p_slope": fit.pvalue}
        if step_at is not None:
            line = np.column_stack([np.ones_like(x), x])
            with_step = np.column_stack([line, (x > step_at).astype(float)])
            rss = [np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2) for X in (line, with_step)]
            f_stat = (rss[0] - rss[1]) / (rss[1] / (len(y) - 3))
            row["p_step"] = stats.f.sf(f_stat, 1, len(y) - 3)
        rows[feature] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def pca_scores(log2_matrix, n_components=5):
    """PCA on runs (rows of the result) using features with no missing values.

    Each feature is centred (mean 0) but not scaled: with log2 data, scaling to
    unit variance would give noisy low-abundance features the same weight as
    well-measured ones.
    """
    from sklearn.decomposition import PCA

    x = log2_matrix.dropna().T                 # runs x features
    x = x - x.mean(axis=0)
    pca = PCA(n_components=n_components).fit(x)
    scores = pd.DataFrame(pca.transform(x), index=x.index,
                          columns=[f"PC{i+1}" for i in range(n_components)])
    return scores, pca.explained_variance_ratio_, x.shape[1]
