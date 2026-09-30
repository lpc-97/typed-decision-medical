"""Ablation A2: frozen Qwen3.5-0.8B-Base vision tower (the same one TCB uses), mean-pooled merged image tokens,
+ per-criterion logistic regression — exactly the BiomedCLIP-probe protocol (C by 5-fold CV inside train, valid once).
Separates 'Qwen visual features' from 'Kev decision model reading them'."""
import json, os, sys
import numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tdm.derm7pt import NAMES, load_split
from tdm.metrics import concept_report
from tdm.vision_kev import VisionKev

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
EV = sys.argv[2] if len(sys.argv) > 2 else "valid"
vk = VisionKev.released()

def embed(df):
    X = []
    for p in df["img"]:
        pv, g = vk.image_inputs(Image.open(p))
        X.append(vk.image_features(pv, g).float().mean(0).cpu().numpy())
    return np.stack(X)

tr, va = load_split("train"), load_split(EV)
Xtr, Xva = embed(tr), embed(va)
print("features", Xtr.shape, Xva.shape, flush=True)
res = {}
for seed in (1, 2, 3):
    P, Y, Cs = {}, {}, {}
    for c in NAMES:
        y = tr[c + "_y"].values
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                          {"logisticregression__C": [1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0]},
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc_ovr" if len(set(y)) > 2 else "roc_auc")
        gs.fit(Xtr, y)
        P[c] = gs.predict_proba(Xva); Y[c] = va[c + "_y"].values; Cs[c] = gs.best_params_["logisticregression__C"]
    r = concept_report(Y, P); res[seed] = {"report": r, "C": Cs}
    np.savez(f"{out}/valid_probs_seed{seed}.npz", **{c: P[c] for c in NAMES}, **{c + "_y": Y[c] for c in NAMES})
    print(f"seed {seed} valid mean AUC {r['mean_auc']:.4f} bacc {r['mean_bacc']:.4f} ece {r['mean_ece']:.4f} C={Cs}", flush=True)
json.dump(res, open(f"{out}/qwen_vision_probe.json", "w"), indent=2)
print("A2_DONE")
