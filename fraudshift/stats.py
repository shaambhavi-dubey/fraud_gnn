"""Step 6: paired seed tests (Holm corrected) + block bootstrap over test time steps."""
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score

from . import config as C
from .io import iter_scores


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tests():
    df = pd.read_csv(C.RAW_DIR / "per_seed_metrics.csv")
    rows = []
    for fs in df.feature_set.unique():
        for metric in ("aucpr", "f1"):
            wide = df[df.feature_set == fs].pivot(index="seed", columns="model", values=metric)
            part = []
            for a, b in C.CONTRASTS:
                d = (wide[a] - wide[b]).to_numpy()
                sd = d.std(ddof=1)
                try:
                    pw = float(sps.wilcoxon(d).pvalue)
                except ValueError:
                    pw = float("nan")
                part.append(dict(feature_set=fs, metric=metric, contrast=f"{a} - {b}", n_seeds=len(d),
                                 mean_diff=float(d.mean()), sd_diff=float(sd),
                                 cohens_dz=float(d.mean() / sd) if sd > 0 else float("nan"),
                                 p_ttest=float(sps.ttest_rel(wide[a], wide[b]).pvalue), p_wilcoxon=pw))
            adj = holm([r["p_ttest"] for r in part])
            for r, q in zip(part, adj):
                r["p_ttest_holm"] = float(q)
            rows += part
    out = pd.DataFrame(rows)
    out.to_csv(C.RAW_DIR / "paired_tests.csv", index=False)
    return out


def block_bootstrap():
    rng = np.random.default_rng(2024)
    S = {(e["model"], e["seed"]): e for e in iter_scores(C.PRIMARY_FS)}
    first = next(iter(S.values()))
    y, t = first["y"], first["t"]
    for e in S.values():
        assert np.array_equal(e["y"], y), "test nodes are not aligned across runs"
    steps = np.unique(t)
    by_t = {k: np.flatnonzero(t == k) for k in steps}
    rows = []
    for a, b in C.CONTRASTS:
        diffs = []
        for _ in range(C.N_BOOT):
            ts = rng.choice(steps, len(steps), replace=True)                     # resample time steps
            idx = np.concatenate([rng.choice(by_t[k], len(by_t[k]), replace=True) for k in ts])  # nodes in block
            if y[idx].sum() == 0:
                continue
            d = [average_precision_score(y[idx], S[(a, s)]["s"][idx]) -
                 average_precision_score(y[idx], S[(b, s)]["s"][idx]) for s in C.SEEDS]
            diffs.append(np.mean(d))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append(dict(contrast=f"{a} - {b}", metric="aucpr", n_boot=len(diffs),
                         mean_diff=float(np.mean(diffs)), ci_low=float(lo), ci_high=float(hi),
                         ci_excludes_zero=bool(lo > 0 or hi < 0)))
        print("bootstrap", a, b, round(lo, 4), round(hi, 4), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(C.RAW_DIR / "bootstrap_intervals.csv", index=False)
    return out


def inductive_tests():
    path = C.RAW_DIR / "inductive_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    wide = df.pivot(index="seed", columns="model", values="aucpr")
    rows = []
    for a, b in C.CONTRASTS:
        d = (wide[a] - wide[b]).to_numpy()
        rows.append(dict(contrast=f"{a} - {b}", metric="inductive_aucpr", mean_diff=float(d.mean()),
                         p_ttest=float(sps.ttest_rel(wide[a], wide[b]).pvalue)))
    adj = holm([r["p_ttest"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_ttest_holm"] = float(q)
    pd.DataFrame(rows).to_csv(C.RAW_DIR / "inductive_paired_tests.csv", index=False)


def verdict(pt, bs):
    a, b = C.PRIMARY_CONTRAST
    name = f"{a} - {b}"
    p = pt[(pt.feature_set == C.PRIMARY_FS) & (pt.metric == "aucpr") & (pt.contrast == name)].iloc[0]
    q = bs[bs.contrast == name].iloc[0]
    real = p.p_ttest_holm < 0.05 and q.ci_excludes_zero
    print(f"\nPRIMARY ({name}): mean AUC-PR diff={p.mean_diff:.4f}, Holm p={p.p_ttest_holm:.4f}, "
          f"bootstrap CI=[{q.ci_low:.4f}, {q.ci_high:.4f}] -> "
          f"{'advantage supported by both tests' if real else 'NOT supported by both tests (inconclusive or null)'}")


if __name__ == "__main__":
    pt = paired_tests()
    bs = block_bootstrap()
    inductive_tests()
    verdict(pt, bs)
