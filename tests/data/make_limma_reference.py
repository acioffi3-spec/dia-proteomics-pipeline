"""Regenerate the frozen limma reference used by tests/test_limma_crosscheck.py.

Run once (needs `pip install inmoose`, a Python port of Bioconductor limma
whose own test suite is checked against R):

    python tests/data/make_limma_reference.py

Why freeze the output instead of calling limma in every test run? inmoose
ships no pre-built wheels, so installing it means compiling C code - fragile
in CI and on Windows. The reference values are what matter; the script that
made them is kept here so anyone can regenerate and inspect them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from inmoose.limma import eBayes, lmFit

HERE = Path(__file__).parent
rng = np.random.default_rng(2024)

# Synthetic proteomics-like data: 300 proteins, 4 vs 4 runs.
# Noise depends on intensity (low-abundance proteins are noisier), so the
# trend version of the prior has something real to fit.
n, n_a, n_b = 300, 4, 4
mean = rng.uniform(12, 24, n)
sd = 0.08 + 0.6 * np.exp(-(mean - 12) / 3) * rng.gamma(4, 0.25, n)     # heteroscedastic, varies by protein
x = mean[:, None] + sd[:, None] * rng.standard_normal((n, n_a + n_b))
x[:30, n_a:] += 1.0                                                        # 30 true changes
x[rng.random(x.shape) < 0.08] = np.nan                                    # ~8% missing values
x = pd.DataFrame(x, index=[f"P{i:03d}" for i in range(n)],
                 columns=[f"A{i}" for i in range(n_a)] + [f"B{i}" for i in range(n_b)])
# limma needs >= 2 values per group to estimate the group difference
keep = (x.iloc[:, :n_a].notna().sum(axis=1) >= 2) & (x.iloc[:, n_a:].notna().sum(axis=1) >= 2)
x = x[keep].round(8)          # round BEFORE fitting so the saved input reproduces the reference exactly

design = np.column_stack([np.ones(n_a + n_b), np.r_[np.zeros(n_a), np.ones(n_b)]])
ref = pd.DataFrame(index=x.index)
for trend in (False, True):
    fit = eBayes(lmFit(x.to_numpy(float), design), trend=trend)
    tag = "trend" if trend else "plain"
    ref[f"t_{tag}"] = np.asarray(fit.t)[:, 1]
    ref[f"p_{tag}"] = np.asarray(fit.p_value)[:, 1]
    ref[f"df_prior_{tag}"] = float(np.atleast_1d(fit.df_prior)[0])

x.to_csv(HERE / "limma_input.csv", float_format="%.8f")
ref.to_csv(HERE / "limma_reference.csv", float_format="%.12g")
print(f"wrote {len(x)} proteins")
