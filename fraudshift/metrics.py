import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def choose_threshold(y, s):
    """Threshold maximizing F1 for the illicit class. Call on VALIDATION data only."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def compute(y, s, thr):
    """All metrics at a FIXED threshold. y in {0,1}."""
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn
    nan = float("nan")
    prec = tp / (tp + fp) if tp + fp else nan
    rec = tp / n_pos if n_pos else nan
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else nan
    return dict(
        n=len(y), n_illicit=n_pos,
        aucpr=float(average_precision_score(y, s)) if n_pos else nan,
        precision=prec, recall=rec, f1=f1,
        fpr_licit=fp / n_neg if n_neg else nan,   # FP / (FP + TN): licit cases only
        fnr=fn / n_pos if n_pos else nan,
    )
