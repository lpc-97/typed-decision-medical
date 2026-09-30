"""G0 baseline: frozen BiomedCLIP image embedding + per-criterion logistic regression, trained on the same 413 Derm7pt
train images. C is chosen by 5-fold CV inside train only (never on valid); valid is used once for the report. Test unread.
Runs in the server's system python (open_clip 3.3)."""
import json, os, sys
import numpy as np, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tdm.derm7pt import NAMES, load_split
from tdm.metrics import concept_report
import open_clip

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
EV = sys.argv[2] if len(sys.argv) > 2 else "valid"   # C still chosen by CV inside train only
model, _, pre = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
model = model.cuda().eval()

def embed(df):
    X = []
    with torch.no_grad():
        for p in df["img"]:
            x = pre(Image.open(p).convert("RGB")).unsqueeze(0).cuda()
            X.append(torch.nn.functional.normalize(model.encode_image(x), dim=-1)[0].float().cpu().numpy())
    return np.stack(X)

tr, va = load_split("train"), load_split(EV)
Xtr, Xva = embed(tr), embed(va)
print("features", Xtr.shape, Xva.shape, "cos(train0,train1)=%.3f" % float(Xtr[0] @ Xtr[1]))
res = {}
for seed in (1, 2, 3):
    P, Y, Cs = {}, {}, {}
    for c in NAMES:
        y = tr[c + "_y"].values
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                          {"logisticregression__C": [1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0]}, cv=cv, scoring="roc_auc_ovr" if len(set(y)) > 2 else "roc_auc")
        gs.fit(Xtr, y)
        P[c] = gs.predict_proba(Xva); Y[c] = va[c + "_y"].values; Cs[c] = gs.best_params_["logisticregression__C"]
        if P[c].shape[1] < len(set(va[c + "_y"])) or P[c].shape[1] != len(np.unique(y)):
            raise SystemExit(f"{c}: class mismatch")
    r = concept_report(Y, P); res[seed] = {"report": r, "C": Cs}
    np.savez(f"{out}/valid_probs_seed{seed}.npz", **{c: P[c] for c in NAMES}, **{c + "_y": Y[c] for c in NAMES})
    print(f"seed {seed} valid mean AUC {r['mean_auc']:.4f} bacc {r['mean_bacc']:.4f} ece {r['mean_ece']:.4f}  C={Cs}", flush=True)
json.dump(res, open(f"{out}/biomedclip_probe.json", "w"), indent=2)
print("BASELINE_DONE")
