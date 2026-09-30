"""Ablation comparisons on valid (same protocol as g0_compare: last epoch, paired bootstrap over images, B=2000).
  TCB (Kev init) vs A1 (no Kev init)      -> does Kev's Jev-style decision training matter?
  TCB (Kev init) vs A2 (Qwen-vision probe) -> is the gain more than the Qwen visual features?
Seeds paired by index. A difference counts only if all 3 seeds' 95% CIs exclude 0."""
import json, sys
import numpy as np
from scipy.stats import rankdata

from tdm.derm7pt import NAMES

import os
from tdm.paths import RUNS
R = os.path.join(RUNS, "g0")


def auc_bin(y, s):
    pos = y == 1; n1 = pos.sum(); n0 = len(y) - n1
    r = rankdata(s); return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro_auc(y, p):
    if p.shape[1] == 2: return auc_bin(y, p[:, 1])
    return np.mean([auc_bin((y == k).astype(int), p[:, k]) for k in np.unique(y)])


def mean_auc(Y, P, idx):
    v = []
    for c in NAMES:
        y = Y[c][idx]
        if len(np.unique(y)) < len(np.unique(Y[c])): return None
        v.append(macro_auc(y, P[c][idx]))
    return float(np.mean(v))


def load(f):
    z = np.load(f); return {c: z[c + "_y"] for c in NAMES}, {c: z[c] for c in NAMES}


def boot(Y, Pa, Pb, seed, B=2000):
    rng = np.random.default_rng(seed); n = len(Y[NAMES[0]]); d = []
    while len(d) < B:
        i = rng.integers(0, n, n); a, b = mean_auc(Y, Pa, i), mean_auc(Y, Pb, i)
        if a is not None and b is not None: d.append(a - b)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


if __name__ == "__main__":
    full = None; out = {}
    for name, other in (("A1_no_kev_init", "a1_base_s{s}/valid_probs_epoch6.npz"), ("A2_qwen_vision_probe", "a2_qwenvis/valid_probs_seed{s}.npz")):
        rows = []
        for s in (1, 2, 3):
            Y, Pt = load(f"{R}/tcb_s{s}/valid_probs_epoch6.npz"); Yo, Po = load(f"{R}/{other.format(s=s)}")
            assert all((Y[c] == Yo[c]).all() for c in NAMES)
            idx = np.arange(len(Y[NAMES[0]]))
            d, lo, hi = boot(Y, Pt, Po, seed=s)
            rows.append({"seed": s, "TCB": mean_auc(Y, Pt, idx), name: mean_auc(Y, Po, idx), "dAUC": [d, lo, hi],
                         "per_crit": {c: [macro_auc(Y[c], Pt[c]), macro_auc(Y[c], Po[c])] for c in NAMES}})
            print(f"{name:22s} seed {s}: TCB {rows[-1]['TCB']:.4f}  other {rows[-1][name]:.4f}  dAUC {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
        allpos = all(r["dAUC"][1] > 0 for r in rows)
        print(f"  -> {'TCB better on all seeds (CI>0)' if allpos else 'NOT significant on all seeds'}")
        for c in NAMES:
            print(f"     {c:22s} TCB {np.mean([r['per_crit'][c][0] for r in rows]):.3f}  other {np.mean([r['per_crit'][c][1] for r in rows]):.3f}")
        out[name] = {"rows": rows, "tcb_better_all_seeds": allpos}
    json.dump(out, open(f"{R}/a_compare.json", "w"), indent=2)
    print("A_COMPARE_DONE")
