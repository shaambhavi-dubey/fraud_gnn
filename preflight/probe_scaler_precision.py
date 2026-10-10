"""Read-only probe. Run from the repo root (venv active):

    python preflight\\probe_scaler_precision.py

Features only. No labels, no models, no outcome files, nothing written to disk.
Question: is the 0.1-0.2% standard-deviation gap seen by diagnose_inductive_numerics.py caused by float32
accumulation inside fit_scaler (x.mean(0), x.std(0) on a float32 array)?
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraudshift import config as C  # noqa: E402
from fraudshift.data import build_raw_splits, fit_scaler  # noqa: E402

_, raw = build_raw_splits(C.PRIMARY_FS)
x = raw["train"].x                      # float32, all train-period nodes, unstandardized
print("numpy", np.__version__, "| dtype", x.dtype, "| shape", x.shape, "| C-contiguous", x.flags["C_CONTIGUOUS"])

a = x.astype(np.float64)
m_ref, s_ref = a.mean(0), a.std(0)
m_ax0, s_ax0 = x.mean(0), x.std(0)                          # what fit_scaler does
m_f64, s_f64 = x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)   # same float32 data, float64 accumulator
m_col = np.array([x[:, j].mean() for j in range(x.shape[1])])             # one contiguous column at a time
s_col = np.array([x[:, j].std() for j in range(x.shape[1])])

rel = lambda s: float(np.max(np.abs(s - s_ref) / s_ref))  # noqa: E731
print("\nmax relative error of the standard deviation versus the float64 reference:")
print(f"  fit_scaler way, x.std(0) in float32      : {rel(s_ax0):.2e}")
print(f"  x.std(0, dtype=float64)                  : {rel(s_f64):.2e}")
print(f"  column by column, float32 (1-D)          : {rel(s_col):.2e}")
print("max absolute error of the mean:")
print(f"  x.mean(0) in float32                     : {float(np.max(np.abs(m_ax0 - m_ref))):.2e}")
print(f"  x.mean(0, dtype=float64)                 : {float(np.max(np.abs(m_f64 - m_ref))):.2e}")

print("\ncolumns the diagnostic flagged (160, 63, 57, 4): sd float32 / sd float64-accumulator / sd reference")
for j in (160, 63, 57, 4):
    print(f"  col {j:3d}: {s_ax0[j]:.7f}  {s_f64[j]:.7f}  {s_ref[j]:.7f}   max |value| {np.abs(a[:, j]).max():.1f}")

mu, sd = fit_scaler(x.copy())
print("\nfit_scaler output identical to the float32 x.mean(0)/x.std(0) above:",
      bool(np.array_equal(mu, m_ax0) and np.array_equal(sd, np.where(s_ax0 == 0, 1.0, s_ax0))))