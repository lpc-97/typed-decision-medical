"""A3: calibration after temperature scaling, cross-fitted on valid (never fit and scored on the same images).
For each method: 20 random 50/50 splits of valid; fit ONE global temperature (Kev-style, min NLL over all 7 criteria) on
half A, score ECE/NLL/Brier on half B, and vice versa; report mean over the 40 scored halves. Same protocol for every arm.
Temperature scaling never changes argmax, so AUC is unchanged (ranking within a criterion is preserved)."""
import glob, json, sys
import numpy as np
from scipy.optimize import minimize_scalar

from tdm.derm7pt import NAMES
from tdm.metrics import ece

import os
from tdm.paths import RUNS
R = os.path.join(RUNS, "g0")
ARMS = {
    "TCB (Kev init)": [f"{R}/tcb_s{s}/valid_probs_epoch6.npz" for s in (1, 2, 3)],
    "A1 (no Kev init)": [f"{R}/a1_base_s{s}/valid_probs_epoch6.npz" for s in (1, 2, 3)],
    "zero-shot": [f"{R}/tcb_s1/valid_probs_epoch0.npz"],
    "BiomedCLIP probe": [f"{R}/biomedclip_v2/valid_probs_seed{s}.npz" for s in (1, 2, 3)],
    "Qwen-vision probe (A2)": [f"{R}/a2_qwenvis/valid_probs_seed{s}.npz" for s in (1, 2, 3)],
}


def scaled(P, T):
    z = np.log(np.clip(P, 1e-12, 1)) / T
    z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def nll(Y, P, idx, T):
    return -np.mean([np.log(np.clip(scaled(P[c][idx], T)[np.arange(len(idx)), Y[c][idx]], 1e-12, 1)).mean() for c in NAMES])


def scores(Y, P, idx, T):
    e = np.mean([ece(Y[c][idx], scaled(P[c][idx], T)) for c in NAMES])
    b = np.mean([((scaled(P[c][idx], T) - np.eye(P[c].shape[1])[Y[c][idx]]) ** 2).sum(1).mean() for c in NAMES])
    return e, nll(Y, P, idx, T), b


out = {}
for arm, files in ARMS.items():
    files = [f for f in files if glob.glob(f)]
    if not files:
        print(f"{arm:26s} (missing)"); continue
    per_file = []
    for f in files:
        z = np.load(f); Y = {c: z[c + "_y"] for c in NAMES}; P = {c: z[c] for c in NAMES}
        n = len(Y[NAMES[0]]); rng = np.random.default_rng(0); raw, cal, Ts = [], [], []
        for _ in range(20):
            perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
            for fit, ev in ((A, B), (B, A)):
                T = minimize_scalar(lambda t: nll(Y, P, fit, t), bounds=(0.05, 20), method="bounded").x
                raw.append(scores(Y, P, ev, 1.0)); cal.append(scores(Y, P, ev, T)); Ts.append(T)
        per_file.append((np.mean(raw, 0), np.mean(cal, 0), np.mean(Ts)))
    raw = np.mean([p[0] for p in per_file], 0); cal = np.mean([p[1] for p in per_file], 0)
    sd = np.std([p[1][0] for p in per_file], ddof=1) if len(per_file) > 1 else float("nan")
    T = np.mean([p[2] for p in per_file])
    out[arm] = {"n_runs": len(per_file), "raw": dict(zip(["ece", "nll", "brier"], raw.tolist())),
                "temp_scaled": dict(zip(["ece", "nll", "brier"], cal.tolist())), "ece_scaled_sd_over_seeds": sd, "T": T}
    print(f"{arm:26s} runs={len(per_file)}  raw ECE {raw[0]:.4f} NLL {raw[1]:.4f} Brier {raw[2]:.4f} | "
          f"T={T:.2f} -> ECE {cal[0]:.4f}±{sd:.4f} NLL {cal[1]:.4f} Brier {cal[2]:.4f}")
json.dump(out, open(f"{R}/a3_calibration.json", "w"), indent=2)
print("A3_DONE")
