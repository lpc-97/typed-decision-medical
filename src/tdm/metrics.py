"""Per-criterion concept metrics (shared by training, baselines and the G0 comparison)."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from tdm.derm7pt import NAMES


def macro_auc(y, p):
    """Binary: AUC of P(class 1); multiclass: macro one-vs-rest over classes present in y."""
    p = np.asarray(p, float)
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    present = np.unique(y)
    return float(np.mean([roc_auc_score((y == k).astype(int), p[:, k]) for k in present]))


def ece(y, p, bins=10):
    p = np.asarray(p, float); conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def concept_report(Y, P):
    """Y: {crit: int array}, P: {crit: [N,K] probs} -> {crit: {auc, bacc, ece}, 'mean_auc', 'mean_bacc', 'mean_ece'}"""
    r = {}
    for c in NAMES:
        y = np.asarray(Y[c]); p = np.asarray(P[c])
        r[c] = {"auc": macro_auc(y, p), "bacc": float(balanced_accuracy_score(y, p.argmax(1))), "ece": ece(y, p)}
    for k in ("auc", "bacc", "ece"):
        r["mean_" + k] = float(np.mean([r[c][k] for c in NAMES]))
    return r
