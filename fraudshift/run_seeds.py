"""Step 2: train every (model, feature set, seed); save per-node TEST scores; build per_seed_metrics.csv."""
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .data import build_splits
from .metrics import choose_threshold, compute
from .models import fit_and_score


def train_all():
    C.SCORES_DIR.mkdir(parents=True, exist_ok=True)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fs in C.FEATURE_SETS:
        _, g = build_splits(fs)
        gt, gv, gte = g["train"], g["val"], g["test"]
        lab = gte.y >= 0
        indeg = gte.in_degree()
        lv = gv.y >= 0
        for model in C.MODELS:
            for seed in C.SEEDS:
                out = C.SCORES_DIR / f"{model}_{fs}_seed{seed}.npz"
                if out.exists():
                    continue
                t0 = time.time()
                sv, (ste,) = fit_and_score(model, seed, gt, gv, [gte])
                thr = choose_threshold(gv.y[lv], sv[lv])         # validation only
                vm = compute(gv.y[lv], sv[lv], thr)
                np.savez(out, model=model, fs=fs, seed=seed, thr=thr, val_aucpr=vm["aucpr"],
                         y=gte.y[lab], score=ste[lab], time=gte.time[lab], indeg=indeg[lab],
                         node=gte.node_idx[lab],seconds=time.time() - t0)
                print(f"{model:9s} {fs:8s} seed={seed} val_aucpr={vm['aucpr']:.4f} "
                      f"thr={thr:.3f} ({time.time()-t0:.0f}s)", flush=True)


def collect_metrics():
    rows = []
    for f in sorted(C.SCORES_DIR.glob("*.npz")):
        d = np.load(f)
        m = compute(d["y"], d["score"], float(d["thr"]))
        rows.append(dict(seed=int(d["seed"]), model=str(d["model"]), feature_set=str(d["fs"]),
                         split="test", threshold=float(d["thr"]), val_aucpr=float(d["val_aucpr"]), **m))
    df = pd.DataFrame(rows).sort_values(["feature_set", "model", "seed"])
    df.to_csv(C.RAW_DIR / "per_seed_metrics.csv", index=False)
    print(df.groupby(["feature_set", "model"])[["aucpr", "f1", "recall", "fpr_licit"]].mean().round(4))


if __name__ == "__main__":
    train_all()
    collect_metrics()
