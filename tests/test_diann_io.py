"""Reading DIA-NN reports (Notebook 01, section 2)."""
import pandas as pd
import pytest

from src import diann_io


def test_run_column_strips_folders_and_extensions():
    df = pd.DataFrame({"File.Name": [r"D:\data\run01.raw", "/mnt/x/run02.d", "run03"]})
    assert list(diann_io.add_run_column(df)["run"].astype(str)) == ["run01", "run02", "run03"]


def test_missing_file_gives_readable_error(tmp_path):
    # The error lists what IS in the folder, which catches Windows' hidden ".tsv.tsv"
    (tmp_path / "report.tsv.tsv").write_text("x")
    with pytest.raises(FileNotFoundError, match=r"report\.tsv\.tsv"):
        diann_io.load_diann_report(tmp_path / "report.tsv")


def test_missing_columns_fail_loudly(tmp_path):
    path = tmp_path / "report.tsv"
    pd.DataFrame({"File.Name": ["a"], "Precursor.Id": ["PEPTIDE2"]}).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        diann_io.load_diann_report(path)


def test_loads_selected_columns_and_caches(tmp_path):
    cols = ["File.Name", "Protein.Group", "Precursor.Id", "Q.Value"]
    path = tmp_path / "report.tsv"
    pd.DataFrame({"File.Name": ["r1", "r2"], "Protein.Group": ["P1", "P1"], "Precursor.Id": ["A2", "A2"],
                  "Q.Value": [0.001, 0.002], "Unused": [1, 2]}).to_csv(path, sep="\t", index=False)
    df = diann_io.load_diann_report(path, columns=cols, cache_dir=tmp_path / "cache")
    assert "Unused" not in df.columns and "run" in df.columns
    assert (tmp_path / "cache" / "report.parquet").exists()
    summary = diann_io.describe_report(df)["value"]
    assert summary["runs"] == 2 and summary["duplicate precursor-run rows"] == 0
