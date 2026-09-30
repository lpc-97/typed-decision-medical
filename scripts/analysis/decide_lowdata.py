"""Low-data decision (pre-registered 2026-09-28, BEFORE any low-data result was seen).
Arms: Kev-0.8B warm start (kev) vs untrained base + fresh LoRA/head (base); same subset per seed; 300 optimizer steps;
valid (n=203) evaluated once at the end. Fractions 0.10 / 0.25 / 0.50 of official train.
Per fraction and seed: paired bootstrap over valid images (B=2000) of dAUC = kev - base (mean of 7 criterion macro-AUCs).
"Kev helps at fraction f"  iff  all 3 seeds' 95% CI > 0.
Claim "Jev-style pretraining helps low-data concept learning" survives iff Kev helps at >= 1 fraction; else DROP it."""
import json, sys
import numpy as np

from tdm.derm_stats import boot, load, mean_auc
from tdm.derm7pt import NAMES

import os
from tdm.paths import RUNS
R = os.path.join(RUNS, "lowdata")
out = {"rule": __doc__, "fractions": {}}
survive = False
for f in ("0.1", "0.25", "0.5"):
    rows = []
    for s in (1, 2, 3):
        Y, Pk = load(f"{R}/f{f}_kev_s{s}/valid_probs_final.npz"); Yb, Pb = load(f"{R}/f{f}_base_s{s}/valid_probs_final.npz")
        assert all((Y[c] == Yb[c]).all() for c in NAMES)
        idx = np.arange(len(Y[NAMES[0]]))
        d, lo, hi = boot(Y, Pk, Pb, seed=s)
        rows.append({"seed": s, "kev": mean_auc(Y, Pk, idx), "base": mean_auc(Y, Pb, idx), "dAUC": [d, lo, hi]})
        print(f"frac {f:>4s} seed {s}: kev {rows[-1]['kev']:.4f}  base {rows[-1]['base']:.4f}  dAUC {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    helps = all(r["dAUC"][1] > 0 for r in rows)
    k = [r["kev"] for r in rows]; b = [r["base"] for r in rows]
    print(f"  frac {f}: kev {np.mean(k):.4f}±{np.std(k, ddof=1):.4f}  base {np.mean(b):.4f}±{np.std(b, ddof=1):.4f}  -> {'KEV HELPS' if helps else 'no significant help'}")
    out["fractions"][f] = {"rows": rows, "kev_helps": helps}
    survive |= helps
out["claim_survives"] = survive
json.dump(out, open(f"{R}/lowdata_decision.json", "w"), indent=2)
print("LOWDATA VERDICT:", "claim SURVIVES" if survive else "DROP the Jev-pretraining claim")
