"""Load Elliptic and build the temporal split graphs.

Raw file layout (no headers in the features file):
  elliptic_txs_features.csv : col0 = txId, col1 = time step, cols 2..166 = 165 features
                              (first 93 = local, last 72 = aggregated one-hop neighbours)
  elliptic_txs_classes.csv  : txId, class   ('1' illicit, '2' licit, 'unknown')
  elliptic_txs_edgelist.csv : txId1, txId2  (directed money flow)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Graph:
    x: np.ndarray            # N x F float32
    y: np.ndarray            # 1 illicit, 0 licit, -1 unknown
    edge_index: np.ndarray   # 2 x E, indices LOCAL to this graph
    time: np.ndarray         # time step of every node
    node_idx: np.ndarray     # row position of every node in the full table (global id)

    @property
    def n(self):
        return len(self.y)

    def in_degree(self):
        """In-degree inside this graph: number of parent transactions, i.e. edges whose
        destination is the node. Out-degree is NOT included (protocol section 9)."""
        return np.bincount(self.edge_index[1], minlength=self.n)

    def subset(self, mask):
        """Keep nodes where mask is True and ONLY edges with both endpoints kept."""
        nodes = np.flatnonzero(mask)
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        s, d = self.edge_index
        keep = mask[s] & mask[d]
        ei = np.stack([remap[s[keep]], remap[d[keep]]])
        return Graph(self.x[nodes], self.y[nodes], ei, self.time[nodes], self.node_idx[nodes])


def load_full(data_dir=None):
    """Whole dataset as one Graph (unstandardized, 165 features)."""
    data_dir = data_dir or C.DATA_DIR
    feats = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    cls = pd.read_csv(data_dir / "elliptic_txs_classes.csv", dtype={"txId": np.int64, "class": str})
    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")

    tx = feats[0].to_numpy(dtype=np.int64)
    time = feats[1].to_numpy(dtype=np.int64)
    x = feats.iloc[:, 2:].to_numpy(dtype=np.float32)
    assert x.shape[1] == 165, f"expected 165 feature columns, got {x.shape[1]}"

    lab = cls.set_index("txId")["class"].reindex(tx).to_numpy()
    y = np.full(len(tx), -1, dtype=np.int64)
    y[lab == "1"] = 1
    y[lab == "2"] = 0

    index = pd.Index(tx)
    src = index.get_indexer(edges.iloc[:, 0].to_numpy())
    dst = index.get_indexer(edges.iloc[:, 1].to_numpy())
    keep = (src >= 0) & (dst >= 0)
    ei = np.stack([src[keep], dst[keep]]).astype(np.int64)
    return Graph(x, y, ei, time, np.arange(len(tx)))


def build_raw_splits(fs, data_dir=None):
    """Feature-set slice split by time, features UNSTANDARDIZED. No scaler is fit here."""
    full = load_full(data_dir)
    full.x = full.x[:, : C.FEATURE_SETS[fs]]
    splits = {
        "train": full.subset(np.isin(full.time, C.TRAIN_STEPS)),
        "val": full.subset(np.isin(full.time, C.VAL_STEPS)),
        "test": full.subset(np.isin(full.time, C.TEST_STEPS)),
    }
    return full, splits


def fit_scaler(x):
    """Per-feature mean/std of the rows passed in (features only, no labels).

    Accumulated in float64. Summing ~1e5 float32 rows column-wise in float32 is off by ~1e-3
    (measured on Elliptic: sd relative error 8e-4, mean 6e-4); see amendments/AMENDMENT_1.md, correction 5.
    """
    mu = x.mean(0, dtype=np.float64)
    sd = x.std(0, dtype=np.float64)
    sd[sd == 0] = 1.0
    return mu, sd


def apply_scaler(x, scaler):
    mu, sd = scaler
    return ((x - mu) / sd).astype(np.float32)


def build_splits(fs, data_dir=None):
    """Standardize using TRAIN-period nodes only, slice the feature set, split by time."""
    full, raw = build_raw_splits(fs, data_dir)
    scaler = fit_scaler(raw["train"].x)
    full.x = apply_scaler(full.x, scaler)
    splits = {}
    for name, g in raw.items():
        g.x = apply_scaler(g.x, scaler)
        splits[name] = g
    return full, splits


def assign_group(indeg):
    """Fixed bins from config: 0 = in-degree 0, 1 = in-degree 1, 2 = in-degree 2 or more."""
    return np.where(indeg <= C.GROUP_CUTS[0], 0, np.where(indeg <= C.GROUP_CUTS[1], 1, 2))


GROUP_NAMES = C.GROUP_LABELS