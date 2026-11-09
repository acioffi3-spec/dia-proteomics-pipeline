"""Read DIA-NN main reports.

A DIA-NN "main report" is a long table: one row per precursor per run.
Precursor = peptide sequence + modifications + charge (column ``Precursor.Id``).
Protein-level values (``PG.Quantity``, ``PG.Q.Value``...) are repeated on every
precursor row of that protein group in that run.
"""
from pathlib import Path

import pandas as pd

# Only the columns the notebooks use. The full report also carries per-fragment
# strings (Fragment.Quant.Raw etc.) that roughly double memory use; we skip them.
CORE_COLUMNS = [
    # identifiers
    "File.Name", "Protein.Group", "Protein.Ids", "Genes",
    "Precursor.Id", "Stripped.Sequence", "Proteotypic",
    # confidence (q-values = estimated FDR at the stated level)
    "Q.Value",          # precursor, run-specific
    "PG.Q.Value",       # protein group, run-specific
    "Protein.Q.Value",  # protein (individual Protein.Ids), run-specific
    "Lib.Q.Value",      # precursor, across the whole experiment (library level)
    "CScore", "Decoy.CScore",
    # quantity (DIA-NN: "Quantity" = raw, "Normalised" = after its own normalisation)
    "Precursor.Quantity", "Precursor.Normalised",
    "PG.Quantity", "PG.Normalised",
    # chromatography
    "RT", "RT.Start", "RT.Stop", "Predicted.RT", "iRT",
]


def load_diann_report(path, columns=CORE_COLUMNS, cache_dir=None):
    """Load a DIA-NN main report (.tsv) with only the columns we need.

    Parameters
    ----------
    path : str or Path
        The report file.
    columns : list of str
        Columns to read. The function fails loudly if any are missing, because a
        silently missing q-value column would make every downstream filter wrong.
    cache_dir : str or Path, optional
        If given, a parquet copy is written there and re-used on later runs
        (parquet loads ~10x faster than a 350 MB text file).

    Returns
    -------
    pandas.DataFrame
    """
    path = Path(path)
    cache = Path(cache_dir) / f"{path.stem}.parquet" if cache_dir else None

    # Fail early with a readable message instead of a long pandas traceback.
    # Listing what IS in the folder catches the most common mistakes, such as
    # Windows adding a hidden second extension ("report.tsv.tsv").
    if not path.exists():
        folder = path.parent
        found = sorted(p.name for p in folder.iterdir() if p.name != ".gitkeep") if folder.exists() else []
        raise FileNotFoundError(
            f"Data file not found: {path.name}\n"
            f"  looked in: {folder}\n"
            f"  files there: {found or 'none'}\n"
            f"  see data/README.md for where to download it and what to name it."
        )

    # Re-use the cache only if it is newer than the source file.
    if cache is not None and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache)

    header = pd.read_csv(path, sep="\t", nrows=0).columns
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(f"Report is missing expected columns: {missing}")

    df = pd.read_csv(path, sep="\t", usecols=columns)
    df = add_run_column(df)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def add_run_column(df):
    """Add a short ``run`` name: File.Name without folder or file extension.

    Older DIA-NN versions write just the run name; newer ones write the full
    path (e.g. ``D:\\data\\run01.raw``). Stripping both makes joins to the
    sample table work for either.
    """
    names = df["File.Name"].astype(str)
    run = (names.str.replace("\\", "/", regex=False)
                .str.split("/").str[-1]
                .str.replace(r"\.(raw|wiff|wiff2|d|mzML|dia)$", "", regex=True))
    df = df.assign(run=run.astype("category"))
    return df


def describe_report(df):
    """One-row-per-check summary of what a report contains.

    The max of each q-value column tells you which FDR filters were already
    applied before the file reached you; you can tighten them but never loosen them.
    """
    summary = {
        "rows (precursor x run)": len(df),
        "runs": df["run"].nunique(),
        "unique precursors": df["Precursor.Id"].nunique(),
        "unique protein groups": df["Protein.Group"].nunique(),
        "duplicate precursor-run rows": int(df.duplicated(["run", "Precursor.Id"]).sum()),
    }
    for col in ["Q.Value", "PG.Q.Value", "Protein.Q.Value", "Lib.Q.Value"]:
        if col in df:
            summary[f"max {col}"] = round(float(df[col].max()), 4)
    return pd.Series(summary, name="value").to_frame()
