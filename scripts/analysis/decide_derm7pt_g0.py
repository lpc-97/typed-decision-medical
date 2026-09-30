"""G0 decision (pre-registered 2026-09-27, written BEFORE any fine-tuned result was seen).

Compared on Derm7pt VALID (n=203; test never read):
  TCB  = Kev-0.8B + vision, fine-tuned on train, LAST epoch (6; fixed in advance, no selection on valid)
  PROBE = frozen BiomedCLIP + per-criterion logistic regression, C chosen by 5-fold CV inside train (biomedclip_v2)
  ZS   = TCB at epoch 0 (Kev-0.8B zero-shot + vision, no derm training)
Metric: mean over the 7 criteria of macro one-vs-rest AUC.
Test: per seed, paired bootstrap over valid images (B=2000, same resample for both arms) of dAUC = TCB - PROBE.
PASS  iff  all 3 seeds have 95% CI of dAUC strictly > 0.
STOP (pause and reassess the plan) iff any seed's CI includes 0 or mean dAUC <= 0.
"""
import json, sys
import numpy as np

from tdm.derm7pt import NAMES
from tdm.metrics import macro_auc, concept_report

import os
from tdm.paths import RUNS
R = os.path.join(RUNS, "g0")
LAST = 6


def load(path):
    z = np.load(path)
    return {c: z[c + "_y"] for c in NAMES}, {c: z[c] for c in NAMES}


def mean_auc(Y, P, idx):
    vals = []
    for c in NAMES:
        y = Y[c][idx]
        if len(np.unique(y)) < len(np.unique(Y[c])):   # a class vanished in this resample -> skip the resample
            return None
        vals.append(macro_auc(y, P[c][idx]))
    return float(np.mean(vals))


def paired_boot(Y, Pa, Pb, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(Y[NAMES[0]]); d = []
    while len(d) < B:
        idx = rng.integers(0, n, n)
        a, b = mean_auc(Y, Pa, idx), mean_auc(Y, Pb, idx)
        if a is not None and b is not None: d.append(a - b)
    d = np.array(d); return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


out = {"rule": __doc__, "seeds": {}}
passes = []
for s in (1, 2, 3):
    Y, Pt = load(f"{R}/tcb_s{s}/valid_probs_epoch{LAST}.npz")
    Yb, Pb = load(f"{R}/biomedclip_v2/valid_probs_seed{s}.npz")
    Y0, P0 = load(f"{R}/tcb_s{s}/valid_probs_epoch0.npz")
    for c in NAMES:
        assert (Y[c] == Yb[c]).all() and (Y[c] == Y0[c]).all(), "label order mismatch between arms"
    rt, rb, r0 = concept_report(Y, Pt), concept_report(Y, Pb), concept_report(Y, P0)
    full = np.arange(len(Y[NAMES[0]]))
    d, lo, hi = paired_boot(Y, Pt, Pb, seed=s)
    dz, loz, hiz = paired_boot(Y, Pt, P0, seed=100 + s)
    ok = lo > 0
    passes.append(ok)
    out["seeds"][s] = {"TCB": rt, "PROBE": rb, "ZS": r0, "dAUC_vs_probe": [d, lo, hi], "dAUC_vs_zeroshot": [dz, loz, hiz], "pass": ok}
    print(f"seed {s}: meanAUC TCB {rt['mean_auc']:.4f} | PROBE {rb['mean_auc']:.4f} | ZS {r0['mean_auc']:.4f}  "
          f"dAUC(TCB-PROBE) {d:+.4f} [{lo:+.4f},{hi:+.4f}]  dAUC(TCB-ZS) {dz:+.4f} [{loz:+.4f},{hiz:+.4f}]  -> {'PASS' if ok else 'FAIL'}")
print("\nper-criterion AUC (mean over seeds)  TCB / PROBE / ZS   and ECE TCB / PROBE")
for c in NAMES:
    f = lambda k, m: np.mean([out["seeds"][s][k][c][m] for s in (1, 2, 3)])
    print(f"  {c:22s} {f('TCB','auc'):.3f} / {f('PROBE','auc'):.3f} / {f('ZS','auc'):.3f}    ece {f('TCB','ece'):.3f} / {f('PROBE','ece'):.3f}")
ms = lambda k: [out["seeds"][s][k]["mean_auc"] for s in (1, 2, 3)]
print(f"\nmean AUC over seeds: TCB {np.mean(ms('TCB')):.4f}±{np.std(ms('TCB'), ddof=1):.4f}  PROBE {np.mean(ms('PROBE')):.4f}±{np.std(ms('PROBE'), ddof=1):.4f}  ZS {np.mean(ms('ZS')):.4f}")
verdict = "PASS" if all(passes) else "STOP"
out["verdict"] = verdict
json.dump(out, open(f"{R}/g0_decision.json", "w"), indent=2)
print("G0 VERDICT:", verdict, f"({R}/g0_decision.json)")
