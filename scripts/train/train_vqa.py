"""Fine-tune Kev-0.8B + vision (init kev) or the untrained base (init base) on VQA-RAD / SLAKE yes/no.
Pre-registered recipe (2026-09-29): 3 epochs, lr 5e-5 OneCycle, batch 8 questions, CE on the Noul distribution,
option-order reversal on 30% of questions (kev perm_frac analogue), flips/rot90 NOT used (anatomy has orientation).
--split resplit : train = image-disjoint train, eval = image-disjoint val   (supplementary; not used in the paper)
--split official: train = official train, eval = official val (SLAKE); VQA-RAD has no official val -> no eval here
(USER DECISION 2026-09-29: official splits are PRIMARY; image-disjoint resplit is supplementary only)
Final trainable weights are always saved for the one-time official-test pass.
Last epoch evaluated; no selection. Test never read here."""
import argparse, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F

from tdm.vision_kev import VisionKev, load_trainable
from tdm.vqa import image, load, record

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True); ap.add_argument("--init", choices=["kev", "base"], required=True)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--split", default="official", choices=["official", "resplit"])
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--rev_frac", type=float, default=0.3)
ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

df = load(a.ds)
tr = df[df[a.split] == "train"].reset_index(drop=True)
ev = df[df[a.split] == "val"].reset_index(drop=True)
tok, m, meta, src = load_trainable(warm=(a.init == "kev"))
vk = VisionKev(tok, m, meta)
trainable = [p for p in m.parameters() if p.requires_grad]
print("== CONFIG", json.dumps({**vars(a), "n_train": len(tr), "n_eval": len(ev), "train_imgs": tr.img_md5.nunique(),
      "eval_imgs": ev.img_md5.nunique(), "eval_imgs_seen_in_train": int(ev.img_md5.isin(set(tr.img_md5)).sum()),
      "init": src["init_from"], "trainable_M": sum(p.numel() for p in trainable) / 1e6}), flush=True)

cache = {}
def feats(r):
    if r.img_md5 not in cache:
        pv, g = vk.image_inputs(image(r)); cache[r.img_md5] = (vk.image_features(pv, g), g)
    return cache[r.img_md5]

def loss_of(r):
    rev = random.random() < a.rev_frac
    rec = record(r.question, int(r.y), reverse=rev)
    f, g = feats(r)
    (z, _), = vk.logits(rec, f, g)
    return F.cross_entropy(z[None], torch.tensor([rec["questions"][0]["label"]], device="cuda"))

@torch.no_grad()
def evaluate():
    m.eval(); p = []
    for _, r in ev.iterrows():
        f, g = feats(r); (z, _), = vk.logits(record(r.question), f, g); p.append(float(torch.softmax(z, -1)[1]))
    return np.array(p)

opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
m.train(); l0 = loss_of(tr.iloc[0]); l0.backward()
nz = sum(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if p.requires_grad and "lora" in n)
print(f"== GRAD CHECK loss {l0.item():.4f} lora tensors with nonzero grad {nz} | head {all(p.grad is not None for p in m.head.parameters())} | visual {sum(p.grad is not None for p in vk.visual.parameters())}", flush=True)
opt.zero_grad(set_to_none=True)
for ep in range(1, a.epochs + 1):
    m.train(); order = list(range(len(tr))); random.shuffle(order); tot = 0.0; t0 = time.time()
    for b in range(0, len(order), a.batch):
        opt.zero_grad(set_to_none=True); chunk = order[b:b + a.batch]
        for i in chunk:
            l = loss_of(tr.iloc[i]) / len(chunk); l.backward(); tot += l.item() * len(chunk)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
    print(f"epoch {ep} loss {tot / len(tr):.4f} ({time.time() - t0:.0f}s)", flush=True)
torch.save({n: q.detach().cpu() for n, q in m.named_parameters() if q.requires_grad}, f"{a.out}/final_trainable.pt")
if len(ev) == 0:
    json.dump({"args": vars(a), "note": "no eval split (VQA-RAD official has train/test only); weights saved"}, open(f"{a.out}/report.json", "w"), indent=2)
    print("VQA_TRAIN_DONE (no eval split; weights saved)"); sys.exit(0)
p = evaluate(); y = ev["y"].values
from sklearn.metrics import roc_auc_score
seen = ev.img_md5.isin(set(tr.img_md5)).values
rep = {"args": vars(a), "n": int(len(y)), "acc": float(((p > .5) == y).mean()), "auc": float(roc_auc_score(y, p)),
       "brier": float(((p - y) ** 2).mean())}
if seen.any() and (~seen).any():
    for nm, mk in (("seen_img", seen), ("unseen_img", ~seen)):
        rep[nm] = {"n": int(mk.sum()), "acc": float(((p[mk] > .5) == y[mk]).mean()), "auc": float(roc_auc_score(y[mk], p[mk]))}
np.savez(f"{a.out}/eval_preds.npz", p_yes=p, y=y, img=ev.img_md5.values, qid=ev.qid.values, seen=seen)
json.dump(rep, open(f"{a.out}/report.json", "w"), indent=2)
print("VQA_TRAIN_DONE", json.dumps({k: v for k, v in rep.items() if k != "args"}))
