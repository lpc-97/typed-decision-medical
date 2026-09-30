"""Option-name test, fine-tuned models, 3 seeds x {Kev init, base init}, last-epoch weights (Derm7pt validation, 5 Choice criteria).
Prespecified (2026-09-29, before results): per seed, paired bootstrap (B=2000, same image resample for all four quantities)
of the difference-in-differences DiD = [AUC_kev(C3)-AUC_kev(C0)] - [AUC_base(C3)-AUC_base(C0)].
"Kev init reduces name-following" iff all 3 seeds' 95% CI of DiD > 0. Also per-arm swap drop per seed and flips/100."""
import json, sys
import numpy as np
from tdm.derm_stats import auc_bin
from tdm.derm7pt import CRITERIA
import os
from tdm.paths import RUNS
CH = [c for c in CRITERIA if CRITERIA[c][0] == "choice"]
R = os.path.join(RUNS, "optname3")


def load(d):
    z = np.load(f"{R}/{d}/probs.npz"); Y = {c: z[c + "_y"] for c in CH}
    return Y, {cond: {c: z[f"{cond}__{c}"] for c in CH} for cond in ("C0_orig", "C3_swap")}


def mauc(Y, P, idx):
    v = []
    for c in CH:
        y = Y[c][idx]
        if len(np.unique(y)) < len(np.unique(Y[c])): return None
        v.append(np.mean([auc_bin((y == k).astype(int), P[c][idx][:, k]) for k in np.unique(y)]))
    return float(np.mean(v))


def ci(v): return [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


out = {"rule": __doc__, "seeds": {}}
helps = []
for s in (1, 2, 3):
    Yk, Pk = load(f"ft_kev_s{s}"); Yb, Pb = load(f"ft_base_s{s}")
    assert all((Yk[c] == Yb[c]).all() for c in CH)
    n = len(Yk[CH[0]]); rng = np.random.default_rng(100 + s); dk, db, did = [], [], []
    while len(did) < 2000:
        i = rng.integers(0, n, n)
        a = [mauc(Yk, Pk["C3_swap"], i), mauc(Yk, Pk["C0_orig"], i), mauc(Yb, Pb["C3_swap"], i), mauc(Yb, Pb["C0_orig"], i)]
        if None in a: continue
        dk.append(a[0] - a[1]); db.append(a[2] - a[3]); did.append(dk[-1] - db[-1])
    full = np.arange(n)
    flips = lambda P: float(np.mean([np.mean(P["C3_swap"][c].argmax(1) != P["C0_orig"][c].argmax(1)) for c in CH]) * 100)
    row = {"kev_C0": mauc(Yk, Pk["C0_orig"], full), "base_C0": mauc(Yb, Pb["C0_orig"], full),
           "kev_swap_dAUC": ci(dk), "base_swap_dAUC": ci(db), "DiD": ci(did), "kev_flips": flips(Pk), "base_flips": flips(Pb)}
    out["seeds"][s] = row; helps.append(row["DiD"][1] > 0)
    f = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]"
    print(f"seed {s}: Kev swap {f(row['kev_swap_dAUC'])} flips {row['kev_flips']:.1f} | base swap {f(row['base_swap_dAUC'])} flips {row['base_flips']:.1f} | DiD {f(row['DiD'])}")
out["kev_reduces_name_following"] = all(helps)
json.dump(out, open(f"{R}/optname3_decision.json", "w"), indent=2)
print("VERDICT: Kev init reduces name-following" if all(helps) else "VERDICT: no consistent difference (rule not met)")
