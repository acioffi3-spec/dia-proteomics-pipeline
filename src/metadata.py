"""Build the sample table from run names.

For PXD018874 (Messner et al., Cell Systems 2020) the run names encode the
design, e.g.

    Plasma_1-F4_64_128   ->  sample type, plate 1, well F4, plate position 64, injection 128
    tQC_NA_NA_389        ->  pooled QC (not on a plate), injection 389

The last number is read as injection order. Evidence for that reading is
checked in the notebook (it is unique per run and the pooled QC appears every
~10th value, matching the PRIDE description).
"""
import re

import pandas as pd

RUN_PATTERN = re.compile(
    r"^(?P<sample_type>[A-Za-z]+)_"   # Plasma / Serum / tQC
    r"(?P<plate_well>[^_]+)_"         # 1-F4  or  NA
    r"(?P<plate_position>[^_]+)_"     # 64    or  NA
    r"(?P<injection_order>\d+)$"      # 128
)

# Human-readable labels used in tables and figure legends.
SAMPLE_LABELS = {"Plasma": "Plasma", "Serum": "Serum", "tQC": "Pooled QC"}


def parse_run_names(runs):
    """Parse run names into a tidy sample table (one row per run).

    Raises if any run does not match the expected pattern, so a renamed file
    can never slip through unannotated.
    """
    runs = pd.Series(pd.unique(pd.Series(runs).astype(str)), name="run")
    parts = runs.str.extract(RUN_PATTERN)

    unparsed = runs[parts["sample_type"].isna()]
    if len(unparsed):
        raise ValueError(f"{len(unparsed)} run names did not parse, e.g. {unparsed.iloc[0]!r}")

    plate_well = parts["plate_well"].str.split("-", n=1, expand=True)
    meta = pd.DataFrame({
        "run": runs,
        "sample_type": parts["sample_type"],
        "plate": plate_well[0].replace("NA", pd.NA),
        "well": plate_well[1] if plate_well.shape[1] > 1 else pd.NA,
        "injection_order": parts["injection_order"].astype(int),
    })
    meta["well_row"] = meta["well"].str[0]
    meta["well_col"] = pd.to_numeric(meta["well"].str[1:], errors="coerce").astype("Int64")
    meta["label"] = meta["sample_type"].map(SAMPLE_LABELS).fillna(meta["sample_type"])
    return meta.sort_values("injection_order").set_index("run")


def check_design(meta):
    """Sanity checks on the parsed design. Returns a small table of findings."""
    qc_order = meta.loc[meta["sample_type"] == "tQC", "injection_order"].sort_values()
    gaps = qc_order.diff().dropna().astype(int).value_counts().sort_index()
    checks = {
        "runs": len(meta),
        "injection order unique": meta["injection_order"].is_unique,
        "injection order range": f"{meta['injection_order'].min()}-{meta['injection_order'].max()}",
        "injection numbers absent from this file": int(
            meta["injection_order"].max() - meta["injection_order"].min() + 1 - len(meta)),
        "pooled-QC spacing (injections: count)": ", ".join(f"{k}: {v}" for k, v in gaps.items()),
    }
    return pd.Series(checks, name="value").to_frame()
