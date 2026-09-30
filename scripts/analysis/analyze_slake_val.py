"""SLAKE official-val analysis (primary protocol per user decision). Image-clustered bootstrap (B=2000).
Same pre-registered comparisons as vqa_analyze.py; plus seen/unseen-image breakdown of fine-tuned models."""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from tdm.vqa_stats import cluster_boot, load, summ
import os
from tdm.paths import RUNS
Z, F = os.path.join(RUNS, "vqa_zs_official"), os.path.join(RUNS, "vqa_ft_official")
out = {"zs": {}, "zs_diff": {}, "ft": []}
arms = {n: load(f"{Z}/{n}_slake_val/preds.npz") for n in ("kev08b", "lp08b", "kev4b", "lp4b", "reflex4b")}
print("SLAKE official val — zero-shot")
for n, (p, y, img) in arms.items():
    s = summ(p, y); _, lo, hi = cluster_boot(y, img, p); out["zs"][n] = {**s, "auc_ci": [lo, hi]}
    print(f"  {n:9s} acc {s['acc']:.3f} AUC {s['auc']:.3f} [{lo:.3f},{hi:.3f}] ECE {s['ece']:.3f} Brier {s['brier']:.3f}")
for a_, b_ in (("reflex4b", "lp4b"), ("kev08b", "lp08b"), ("kev4b", "lp4b")):
    pa, y, img = arms[a_]; pb = arms[b_][0]; d = cluster_boot(y, img, pa, pb, seed=1); out["zs_diff"][f"{a_}-{b_}"] = d
    print(f"  dAUC {a_} - {b_}: {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
print("SLAKE official val — fine-tuned (official train), last epoch; seen = image also in official train")
for s in (1, 2, 3):
    pk, y, img = load(f"{F}/slake_kev_s{s}/eval_preds.npz"); pb, _, _ = load(f"{F}/slake_base_s{s}/eval_preds.npz")
    seen = np.load(f"{F}/slake_kev_s{s}/eval_preds.npz", allow_pickle=True)["seen"]
    d = cluster_boot(y, img, pk, pb, seed=10 + s)
    row = {"seed": s, "kev": summ(pk, y), "base": summ(pb, y), "dAUC": d}
    for nm, mk in (("seen", seen), ("unseen", ~seen)):
        if mk.sum() == 0: row[nm] = {"n": 0}; continue
        row[nm] = {"n": int(mk.sum()), "kev_acc": float(((pk[mk] > .5) == y[mk]).mean()), "base_acc": float(((pb[mk] > .5) == y[mk]).mean()),
                   "kev_auc": float(roc_auc_score(y[mk], pk[mk])), "base_auc": float(roc_auc_score(y[mk], pb[mk]))}
    out["ft"].append(row)
    print(f"  seed {s}: kev acc {row['kev']['acc']:.3f} AUC {row['kev']['auc']:.3f} ECE {row['kev']['ece']:.3f} | base acc {row['base']['acc']:.3f} AUC {row['base']['auc']:.3f} ECE {row['base']['ece']:.3f} | dAUC {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
    print(f"          images seen in yes/no train: {row['seen']['n']} of {len(y)} questions")
out["kev_helps"] = all(r["dAUC"][1] > 0 for r in out["ft"])
print("  -> Kev init", "HELPS (all 3 CI>0)" if out["kev_helps"] else "no significant help")
json.dump(out, open(os.path.join(RUNS, "vqa_slake_official_val_analysis.json"), "w"), indent=2)
