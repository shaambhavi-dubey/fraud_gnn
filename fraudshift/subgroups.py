"""Step 3: metrics per observed-activity (degree) group, threshold fixed from validation."""
import pandas as pd

from . import config as C
from .data import GROUP_NAMES, assign_group
from .io import iter_scores
from .metrics import compute


def main():
    rows = []
    for e in iter_scores():
        grp = assign_group(e["indeg"])
        for k, name in GROUP_NAMES.items():
            m = grp == k
            r = compute(e["y"][m], e["s"][m], e["thr"])
            rows.append(dict(seed=e["seed"], model=e["model"], feature_set=e["fs"], group=name,
                             underpowered=r["n_illicit"] < C.MIN_ILLICIT, **r))
    pd.DataFrame(rows).to_csv(C.RAW_DIR / "per_group_metrics.csv", index=False)
    print("wrote per_group_metrics.csv")


if __name__ == "__main__":
    main()
