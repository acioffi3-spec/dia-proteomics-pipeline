"""Run names -> sample table (Notebook 01, section 3)."""
import pandas as pd
import pytest

from src import metadata


def test_parses_plate_wells_and_pooled_qc():
    meta = metadata.parse_run_names(["Plasma_1-F4_64_128", "Serum_3-H7_91_250", "tQC_NA_NA_389"])
    assert meta.loc["Plasma_1-F4_64_128", "sample_type"] == "Plasma"
    assert meta.loc["Plasma_1-F4_64_128", "plate"] == "1"
    assert meta.loc["Plasma_1-F4_64_128", "well"] == "F4"
    assert meta.loc["Serum_3-H7_91_250", "well_row"] == "H"
    assert meta.loc["Serum_3-H7_91_250", "well_col"] == 7
    assert meta.loc["tQC_NA_NA_389", "label"] == "Pooled QC"
    assert pd.isna(meta.loc["tQC_NA_NA_389", "plate"])


def test_sorted_by_injection_order():
    meta = metadata.parse_run_names(["Serum_1-D2_2_30", "Plasma_1-B2_1_10", "tQC_NA_NA_20"])
    assert list(meta["injection_order"]) == [10, 20, 30]


def test_unparseable_name_fails_loudly():
    # A renamed file must never slip through unannotated
    with pytest.raises(ValueError, match="did not parse"):
        metadata.parse_run_names(["Plasma_1-F4_64_128", "some_other_file"])


def test_check_design_reports_qc_spacing():
    runs = ["tQC_NA_NA_1", "Plasma_1-B2_1_2", "tQC_NA_NA_11", "Serum_1-D2_2_12", "tQC_NA_NA_21"]
    checks = metadata.check_design(metadata.parse_run_names(runs))["value"]
    assert checks["injection order unique"]
    assert checks["pooled-QC spacing (injections: count)"] == "10: 2"
