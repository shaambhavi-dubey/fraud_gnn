"""Strict inductive split construction (protocol section 12). No model code is imported here.

Order of operations (pre-run correction, see amendments/AMENDMENT_1.md):
  1. start from UNSTANDARDIZED train-period and validation features
  2. draw the stratified hold-out of labeled train-period nodes
  3. remove held-out nodes and ALL their incident edges  -> training graph
  4. fit the scaler ONCE on the retained nodes only
  5. apply that scaler to the training graph, the validation graph and the inference graph
"""
from dataclasses import dataclass

import numpy as np

from . import config as C
from .data import Graph, apply_scaler, fit_scaler


def stratified_holdout(rng, y, frac):
    held = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        held.append(rng.choice(idx, int(round(frac * len(idx))), replace=False))
    return np.concatenate(held)


@dataclass
class InductiveSplit:
    held: np.ndarray        # local indices (into the raw train graph) of held-out nodes
    scaler: tuple           # (mean, std) fit on retained nodes only
    g_train: Graph          # held-out nodes and all incident edges removed, scaled
    g_val: Graph            # validation graph, scaled with the retained-node scaler
    g_infer: Graph          # train-period graph, held<->held edges dropped, scaled


def make_inductive_split(raw_train, raw_val, seed):
    rng = np.random.default_rng(1000 + seed)          # same hold-out for every model
    held = stratified_holdout(rng, raw_train.y, C.INDUCTIVE_HOLDOUT)
    held_mask = np.zeros(raw_train.n, dtype=bool)
    held_mask[held] = True

    g_tr = raw_train.subset(~held_mask)                # raw (unscaled) retained graph
    # leakage checks
    assert (raw_train.y[held] >= 0).all(), "unlabeled node in hold-out"
    assert not np.isin(g_tr.node_idx, raw_train.node_idx[held]).any(), "held-out node in training graph"
    assert g_tr.n == raw_train.n - len(held)
    assert g_tr.edge_index.max(initial=-1) < g_tr.n
    s, d = raw_train.edge_index
    assert g_tr.edge_index.shape[1] == raw_train.edge_index.shape[1] - int((held_mask[s] | held_mask[d]).sum()), \
        "an edge incident to a held-out node survived in the training graph"

    scaler = fit_scaler(g_tr.x)                        # RETAINED nodes only, fit once
    g_tr.x = apply_scaler(g_tr.x, scaler)
    g_val = Graph(apply_scaler(raw_val.x, scaler), raw_val.y, raw_val.edge_index,
                  raw_val.time, raw_val.node_idx)

    keep = ~(held_mask[s] & held_mask[d])              # drop held<->held edges only
    g_inf = Graph(apply_scaler(raw_train.x, scaler), raw_train.y, raw_train.edge_index[:, keep],
                  raw_train.time, raw_train.node_idx)
    return InductiveSplit(held, scaler, g_tr, g_val, g_inf)
