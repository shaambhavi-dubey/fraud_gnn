"""Read-only diagnostic for the one failing preflight-v2 check. Run from the repo root:

    python preflight\\diagnose_inductive_numerics.py

Features only: no labels-versus-predictions anything, no model, no outcome file, nothing written to disk.
For seeds 0-9 it compares three standardizations of the retained train-period nodes:
  new  = the corrected path (raw -> fit scaler once on retained nodes, float32), as in inductive_split.py
  v1   = the protocol-v1 route (standardize on all train nodes, then re-standardize retained, float32)
  ref  = float64 reference computed directly from the raw retained rows
and prints the feature columns where they disagree.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraudshift import config as C  # noqa: E402
from fraudshift.data import build_raw_splits, fit_scaler  # noqa: E402
from fraudshift.inductive_split import make_inductive_split  # noqa: E402

_, raw = build_raw_splits(C.PRIMARY_FS)
tr, va = raw["train"], raw["val"]
mu_all, sd_all = fit_scaler(tr.x)
x_all = ((tr.x - mu_all) / sd_all).astype(np.float32)

worst = {}            # column -> dict of worst values across seeds
overall_new_v1 = overall_new_ref = overall_v1_ref = 0.0
for seed in C.SEEDS:
    sp = make_inductive_split(tr, va, seed)
    mask = np.zeros(tr.n, dtype=bool)
    mask[sp.held] = True
    new = sp.g_train.x
    xs = x_all[~mask]
    m1, s1 = xs.mean(0), xs.std(0)
    s1[s1 == 0] = 1.0
    v1 = ((xs - m1) / s1).astype(np.float32)
    r = tr.x[~mask].astype(np.float64)
    m64, s64 = r.mean(0), r.std(0)
    s64_safe = np.where(s64 == 0, 1.0, s64)
    ref = (r - m64) / s64_safe
    d_nv = np.abs(new - v1).max(0)
    d_nr = np.abs(new - ref).max(0)
    d_vr = np.abs(v1 - ref).max(0)
    overall_new_v1 = max(overall_new_v1, float(d_nv.max()))
    overall_new_ref = max(overall_new_ref, float(d_nr.max()))
    overall_v1_ref = max(overall_v1_ref, float(d_vr.max()))
    for j in np.flatnonzero(d_nv > 1e-3):
        w = worst.setdefault(int(j), dict(new_v1=0, new_ref=0, v1_ref=0))
        w["new_v1"] = max(w["new_v1"], float(d_nv[j]))
        w["new_ref"] = max(w["new_ref"], float(d_nr[j]))
        w["v1_ref"] = max(w["v1_ref"], float(d_vr[j]))
        w["sd32"] = float(sp.scaler[1][j])
        w["sd64"] = float(s64[j])
        w["mean64"] = float(m64[j])
        w["n_unique"] = int(len(np.unique(tr.x[~mask][:, j])))

print(f"feature set {C.PRIMARY_FS}; seeds {C.SEEDS[0]}-{C.SEEDS[-1]}")
print(f"max |new - v1|  over all columns/seeds: {overall_new_v1:.3e}")
print(f"max |new - ref| over all columns/seeds: {overall_new_ref:.3e}   (ref = float64 from raw retained rows)")
print(f"max |v1  - ref| over all columns/seeds: {overall_v1_ref:.3e}")
print(f"columns where |new - v1| > 1e-3: {len(worst)}")
print("col  new-v1     new-ref    v1-ref     sd(float32)  sd(float64)  mean(float64)  n_unique")
for j, w in sorted(worst.items(), key=lambda kv: -kv[1]["new_v1"])[:15]:
    print(f"{j:3d}  {w['new_v1']:.2e}  {w['new_ref']:.2e}  {w['v1_ref']:.2e}  {w['sd32']:.3e}   "
          f"{w['sd64']:.3e}   {w['mean64']:.3e}   {w['n_unique']}")