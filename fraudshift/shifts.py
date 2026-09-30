"""Step 4: population shifts on the TEST set. Model and threshold are never changed."""
import numpy as np
import pandas as pd

from . import config as C
from .data import assign_group
from .io import iter_scores
from .metrics import compute


def resample_activity(rng, y, grp, share, n_total):
    """Set low-degree share to `share` inside each class; overall illicit prevalence unchanged."""
    n_pos = int(round(y.mean() * n_total))
    parts = []
    for cls, n_cls in ((1, n_pos), (0, n_total - n_pos)):
        ci = np.flatnonzero(y == cls)
        g = grp[ci]
        low, mid, high = ci[g == 0], ci[g == 1], ci[g == 2]
        n_low = int(round(share * n_cls))
        rest = n_cls - n_low
        n_mid = int(round(rest * len(mid) / (len(mid) + len(high)))) if len(mid) + len(high) else 0
        for pool, k in ((low, n_low), (mid, n_mid), (high, rest - n_mid)):
            if k > 0:
                if len(pool) == 0:
                    return None
                parts.append(rng.choice(pool, k, replace=True))
    return np.concatenate(parts)


def resample_prevalence(rng, y, mult, n_total):
    """Change illicit share to mult x original (capped at 50%); within-class mix unchanged."""
    prev = min(float(y.mean()) * mult, 0.5)
    n_pos = int(round(prev * n_total))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, n_pos, replace=True),
                           rng.choice(neg, n_total - n_pos, replace=True)])


def mean_metrics(y, s, thr, idxs):
    return pd.DataFrame([compute(y[i], s[i], thr) for i in idxs]).mean().to_dict()


def main():
    rng = np.random.default_rng(12345)
    rows = []
    for e in iter_scores(C.PRIMARY_FS):
        y, s, t, thr = e["y"], e["s"], e["t"], e["thr"]
        grp = assign_group(e["indeg"])
        base = dict(seed=e["seed"], model=e["model"], feature_set=e["fs"])

        # 1. natural temporal shift
        for name, (a, b) in C.TEMPORAL_WINDOWS.items():
            m = (t >= a) & (t <= b)
            r = compute(y[m], s[m], thr)
            rows.append(dict(**base, shift_type="temporal", level=name, level_num=np.nan,
                             underpowered=r["n_illicit"] < C.MIN_ILLICIT, **r))

        # 2. activity-mix shift (prevalence held fixed)
        rows.append(dict(**base, shift_type="activity", level="original",
                         level_num=float((grp == 0).mean()), underpowered=False, **compute(y, s, thr)))
        for share in C.ACTIVITY_LEVELS:
            idxs = []
            for _ in range(C.N_RESAMPLES):
                idx = resample_activity(rng, y, grp, share, len(y))
                if idx is None:
                    break
                idxs.append(idx)
            if idxs:
                rows.append(dict(**base, shift_type="activity", level=f"{share:.2f}", level_num=share,
                                 underpowered=False, **mean_metrics(y, s, thr, idxs)))

        # 3. prevalence shift (activity mix held fixed)
        for mult in C.PREVALENCE_MULTS:
            idxs = [resample_prevalence(rng, y, mult, len(y)) for _ in range(C.N_RESAMPLES)]
            rows.append(dict(**base, shift_type="prevalence", level=f"{mult}x", level_num=mult,
                             underpowered=False, **mean_metrics(y, s, thr, idxs)))
        print("shifts done:", e["model"], e["seed"], flush=True)
    pd.DataFrame(rows).to_csv(C.RAW_DIR / "shift_metrics.csv", index=False)
    print("wrote shift_metrics.csv")


if __name__ == "__main__":
    main()
