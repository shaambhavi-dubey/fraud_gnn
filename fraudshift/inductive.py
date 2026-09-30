"""Step 5: inductive new-node test (primary feature set).

For each seed: hold out 20% of LABELED train-period nodes (stratified).
  training graph : held-out nodes and ALL their edges removed; held-out labels never used
  inference graph: held-out nodes added back with edges ONLY to retained train-period nodes
                   (held-out <-> held-out edges dropped; no val/test edges exist here)
"""
import numpy as np
import pandas as pd

from . import config as C
from .data import Graph, build_splits
from .metrics import choose_threshold, compute
from .models import fit_and_score


def stratified_holdout(rng, y, frac):
    held = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        held.append(rng.choice(idx, int(round(frac * len(idx))), replace=False))
    return np.concatenate(held)


def main():
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    _, g = build_splits(C.PRIMARY_FS)
    gt, gv = g["train"], g["val"]
    lv = gv.y >= 0
    rows = []
    for seed in C.SEEDS:
        rng = np.random.default_rng(1000 + seed)          # same hold-out for every model
        held = stratified_holdout(rng, gt.y, C.INDUCTIVE_HOLDOUT)
        held_mask = np.zeros(gt.n, dtype=bool)
        held_mask[held] = True

        g_tr = gt.subset(~held_mask)
        # leakage checks
        assert not np.isin(g_tr.node_idx, gt.node_idx[held]).any(), "held-out node in training graph"
        assert g_tr.n == gt.n - len(held)
        assert g_tr.edge_index.max(initial=-1) < g_tr.n

                # re-standardize using RETAINED nodes only, so held-out features never influence preprocessing
        m = g_tr.x.mean(0)
        s = g_tr.x.std(0)
        s[s == 0] = 1.0
        g_tr.x = ((g_tr.x - m) / s).astype(np.float32)
        gv_s = Graph(((gv.x - m) / s).astype(np.float32), gv.y, gv.edge_index, gv.time, gv.node_idx)

        ei = gt.edge_index
        keep = ~(held_mask[ei[0]] & held_mask[ei[1]])      # drop held<->held edges
        g_inf = Graph(((gt.x - m) / s).astype(np.float32), gt.y, ei[:, keep], gt.time, gt.node_idx)

        for model in C.MODELS:
            sv, (s_inf,) = fit_and_score(model, seed, g_tr, gv_s, [g_inf])
            thr = choose_threshold(gv.y[lv], sv[lv])       # validation only
            m = compute(gt.y[held], s_inf[held], thr)
            rows.append(dict(seed=seed, model=model, feature_set=C.PRIMARY_FS,
                             n_heldout=len(held), threshold=thr, **m))
            print(f"inductive {model:9s} seed={seed} aucpr={m['aucpr']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.RAW_DIR / "inductive_metrics.csv", index=False)
    print("wrote inductive_metrics.csv")


if __name__ == "__main__":
    main()
