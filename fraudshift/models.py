import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from . import config as C


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _edges(ei, undirected):
    t = torch.as_tensor(ei, dtype=torch.long)
    if undirected:
        from torch_geometric.utils import to_undirected
        t = to_undirected(t)
    return t


class GNN(nn.Module):
    def __init__(self, kind, in_dim, hidden, layers, dropout):
        super().__init__()
        from torch_geometric.nn import GCNConv, SAGEConv
        dims = [in_dim] + [hidden] * layers
        if kind == "gcn":
            self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(layers)])
        else:
            self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1], aggr="mean") for i in range(layers)])
        self.out = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, ei):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, ei)), self.dropout, self.training)
        return self.out(x)


def fit_gnn(kind, seed, g_train, g_val, g_evals):
    """Train on g_train, early-stop on val AUC-PR, return (val_scores, [scores per eval graph])."""
    set_seed(seed)
    cfg = C.GNN
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tens(g):
        return torch.as_tensor(g.x).to(dev), _edges(g.edge_index, C.UNDIRECTED).to(dev)

    xt, et = tens(g_train)
    xv, ev = tens(g_val)
    y_tr = torch.as_tensor(g_train.y).to(dev)
    lab = y_tr >= 0
    n_pos = int((y_tr[lab] == 1).sum())
    n_neg = int((y_tr[lab] == 0).sum())
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=dev)  # from TRAIN prevalence only
    val_lab = g_val.y >= 0

    model = GNN(kind, xt.shape[1], cfg["hidden"], cfg["layers"], cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, bad, best_state = -1.0, 0, None
    for _ in range(cfg["max_epochs"]):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(xt, et)[lab], y_tr[lab], weight=weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(xv, ev), 1)[:, 1].cpu().numpy()
        ap = average_precision_score(g_val.y[val_lab], pv[val_lab])
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()

    def score(g):
        x, e = tens(g)
        with torch.no_grad():
            return torch.softmax(model(x, e), 1)[:, 1].cpu().numpy()

    return score(g_val), [score(g) for g in g_evals]


def fit_xgb(seed, g_train, g_val, g_evals):
    from xgboost import XGBClassifier
    tr, va = g_train.y >= 0, g_val.y >= 0
    n_pos = int((g_train.y[tr] == 1).sum())
    n_neg = int((g_train.y[tr] == 0).sum())
    clf = XGBClassifier(
        **C.XGB, scale_pos_weight=n_neg / max(n_pos, 1), eval_metric="aucpr",
        tree_method="hist", random_state=seed, n_jobs=-1,
    )
    clf.fit(g_train.x[tr], g_train.y[tr], eval_set=[(g_val.x[va], g_val.y[va])], verbose=False)

    def score(g):
        return clf.predict_proba(g.x)[:, 1]

    return score(g_val), [score(g) for g in g_evals]


def fit_and_score(model, seed, g_train, g_val, g_evals):
    if model == "xgboost":
        return fit_xgb(seed, g_train, g_val, g_evals)
    return fit_gnn(model, seed, g_train, g_val, g_evals)
