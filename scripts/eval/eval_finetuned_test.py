"""One-time TEST evaluation of saved fine-tuned weights (last epoch, pre-registered). No training, no selection.
Usage: eval_ft_test.py derm7pt <init kev|base> <weights.pt> <out>
       eval_ft_test.py <vqarad|slake> <init> <weights.pt> <out>     (official test split)"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

from tdm.vision_kev import VisionKev, load_trainable

task, init, wpath, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
tok, m, meta, _ = load_trainable(warm=(init == "kev"), checkpointing=False)
sd = torch.load(wpath, map_location="cpu")
names = [n for n, p in m.named_parameters() if p.requires_grad]
missing = [n for n in names if n not in sd]; assert not missing, missing[:3]
with torch.no_grad():
    for n, p in m.named_parameters():
        if n in sd: p.copy_(sd[n])
m.eval(); vk = VisionKev(tok, m, meta)
print("loaded", wpath, len(names), "tensors", flush=True)

with torch.no_grad():
    if task == "derm7pt":
        from tdm.derm7pt import NAMES, load_split, record
        from tdm.metrics import concept_report
        te = load_split("test"); P = {c: [] for c in NAMES}
        for _, r in te.iterrows():
            pv, g = vk.image_inputs(Image.open(r["img"])); f = vk.image_features(pv, g)
            out_ = vk.logits(record({c: int(r[c + "_y"]) for c in NAMES}), f, g)
            for c, (z, _) in zip(NAMES, out_): P[c].append(F.softmax(z, -1).cpu().numpy())
        Y = {c: te[c + "_y"].values for c in NAMES}; P = {c: np.stack(P[c]) for c in NAMES}
        rep = concept_report(Y, P)
        np.savez(f"{out}/test_probs.npz", **{c: P[c] for c in NAMES}, **{c + "_y": Y[c] for c in NAMES})
        print(f"TEST derm7pt {init} mean AUC {rep['mean_auc']:.4f} bacc {rep['mean_bacc']:.4f} ece {rep['mean_ece']:.4f}")
    else:
        from tdm.vqa import image, load, record
        df = load(task); tr = set(df[df.official == "train"].img_md5); te = df[df.official == "test"].reset_index(drop=True)
        p, cache = [], {}
        for _, r in te.iterrows():
            if r.img_md5 not in cache:
                pv, g = vk.image_inputs(image(r)); cache[r.img_md5] = (vk.image_features(pv, g), g)
            f, g = cache[r.img_md5]; (z, _), = vk.logits(record(r.question), f, g); p.append(float(F.softmax(z, -1)[1]))
        p = np.array(p); y = te["y"].values; seen = te.img_md5.isin(tr).values
        from sklearn.metrics import roc_auc_score
        rep = {"n": int(len(y)), "acc": float(((p > .5) == y).mean()), "auc": float(roc_auc_score(y, p)), "n_seen_img": int(seen.sum())}
        np.savez(f"{out}/test_preds.npz", p_yes=p, y=y, img=te.img_md5.values, qid=te.qid.values, seen=seen)
        print(f"TEST {task} {init} acc {rep['acc']:.4f} AUC {rep['auc']:.4f} seen-image questions {rep['n_seen_img']}/{rep['n']}")
json.dump(rep, open(f"{out}/test_report.json", "w"), indent=2)
