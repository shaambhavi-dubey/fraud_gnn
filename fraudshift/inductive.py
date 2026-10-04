"""Step 5: inductive new-node test (primary feature set).

For each seed: hold out 20% of LABELED train-period nodes (stratified).
  training graph : held-out nodes and ALL their edges removed; held-out labels never used
  scaler         : fit once on the retained nodes only, starting from UNSTANDARDIZED features
  inference graph: held-out nodes added back with edges ONLY to retained train-period nodes
                   (held-out <-> held-out edges dropped; no val/test edges exist here)
Split construction lives in inductive_split.py (no model code); see amendments/AMENDMENT_1.md.
"""
import pandas as pd

from . import config as C
from .data import build_raw_splits
from .inductive_split import make_inductive_split, stratified_holdout  # noqa: F401
from .metrics import choose_threshold, compute
from .models import fit_and_score


def main():
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    _, raw = build_raw_splits(C.PRIMARY_FS)
    raw_tr, raw_val = raw["train"], raw["val"]
    lv = raw_val.y >= 0
    rows = []
    for seed in C.SEEDS:
        sp = make_inductive_split(raw_tr, raw_val, seed)
        for model in C.MODELS:
            sv, (s_inf,) = fit_and_score(model, seed, sp.g_train, sp.g_val, [sp.g_infer])
            thr = choose_threshold(raw_val.y[lv], sv[lv])       # validation only
            r = compute(raw_tr.y[sp.held], s_inf[sp.held], thr)
            rows.append(dict(seed=seed, model=model, feature_set=C.PRIMARY_FS,
                             n_heldout=len(sp.held), threshold=thr, **r))
            print(f"inductive {model:9s} seed={seed} aucpr={r['aucpr']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.RAW_DIR / "inductive_metrics.csv", index=False)
    print("wrote inductive_metrics.csv")


if __name__ == "__main__":
    main()
