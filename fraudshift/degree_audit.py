"""Degree audit on TRAIN-period labeled nodes only. No model outcomes involved."""
import numpy as np
import pandas as pd

from . import config as C
from .data import build_splits


def main():
    _, g = build_splits(C.PRIMARY_FS)
    tr = g["train"]
    src, dst = tr.edge_index
    indeg = np.bincount(dst, minlength=tr.n)
    outdeg = np.bincount(src, minlength=tr.n)
    lab = tr.y >= 0
    for name, d in (("in-degree", indeg), ("out-degree", outdeg), ("total degree", indeg + outdeg)):
        print(f"\n{name}: labeled TRAIN nodes by value (6 means 6 or more)")
        table = pd.crosstab(np.minimum(d[lab], 6), tr.y[lab])
        table.columns = ["licit", "illicit"]
        print(table.to_string())


if __name__ == "__main__":
    main()