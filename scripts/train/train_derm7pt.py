"""G0: fine-tune Kev-0.8B + vision (Typed Concept Bottleneck) on Derm7pt train; select on valid. Test is NOT read.

Recipe from kev/train.py (delta mode, CE on the option distribution, option permutation of Choice questions with K>=3 on
perm_frac of records); image augmentation = dermoscopy-invariant flips / 90-degree rotations.
"""
import argparse, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

from tdm.derm7pt import CRITERIA, NAMES, load_split, record
from tdm.metrics import concept_report
from tdm.vision_kev import VisionKev, load_trainable

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--perm_frac", type=float, default=0.3)
ap.add_argument("--out", required=True)
ap.add_argument("--frac", type=float, default=1.0, help="low-data: fraction of official train, subset fixed by --seed (same subset for both --init arms)")
ap.add_argument("--steps", type=int, default=0, help="low-data: fixed number of optimizer steps (overrides --epochs); valid evaluated once at the end")
ap.add_argument("--init", choices=["kev", "base"], default="kev", help="kev = warm start from Kev-0.8B; base = ablation A1, untrained base + fresh LoRA/head")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda"

tok, m, meta, src = load_trainable(warm=(a.init == "kev"))
vk = VisionKev(tok, m, meta, device=dev)
train, valid = load_split("train"), load_split("valid")
if a.frac < 1.0:
    train = train.sample(frac=a.frac, random_state=1000 + a.seed).reset_index(drop=True)   # arm-independent subset
print("== CONFIG ECHO", json.dumps({**vars(a), "base": meta.base, "base_revision": meta.base_revision, "init_from": src["init_from"],
      "adapter_tensors": src["tensors"], "n_train": len(train), "n_valid": len(valid)}), flush=True)
trainable = [p for p in m.parameters() if p.requires_grad]
print(f"trainable params {sum(p.numel() for p in trainable)/1e6:.2f}M | visual trainable {sum(p.numel() for p in vk.visual.parameters() if p.requires_grad)}", flush=True)
for c in NAMES:
    print(f"  {c:22s} train dist {np.bincount(train[c+'_y'], minlength=len(CRITERIA[c][2])).tolist()}  valid {np.bincount(valid[c+'_y'], minlength=len(CRITERIA[c][2])).tolist()}")

feat_cache = {}
def feats(path, aug):
    im = Image.open(path).convert("RGB")
    if aug:
        if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5: im = im.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        if k: im = im.rotate(90 * k, expand=True)
        pv, grid = vk.image_inputs(im)
        return vk.image_features(pv, grid), grid
    if path not in feat_cache:
        pv, grid = vk.image_inputs(im); feat_cache[path] = (vk.image_features(pv, grid), grid)
    return feat_cache[path]

def labels_of(row):
    return {c: int(row[c + "_y"]) for c in NAMES}

@torch.no_grad()
def evaluate(df):
    m.eval(); P = {c: [] for c in NAMES}; Y = {c: [] for c in NAMES}
    for _, row in df.iterrows():
        f, g = feats(row["img"], False)
        out = vk.logits(record(labels_of(row)), f, g)
        for c, (z, _) in zip(NAMES, out):
            P[c].append(F.softmax(z, -1).cpu().numpy()); Y[c].append(int(row[c + "_y"]))
    return {c: np.array(Y[c]) for c in NAMES}, {c: np.stack(P[c]) for c in NAMES}

def step_loss(row, check=False):
    lab = labels_of(row); order = {}
    if random.random() < a.perm_frac:
        for c in NAMES:
            K = len(CRITERIA[c][2])
            if CRITERIA[c][0] == "choice" and K >= 3:
                p = list(range(K)); random.shuffle(p); order[c] = p
    rec = record(lab, order)
    f, g = feats(row["img"], True)
    out = vk.logits(rec, f, g)
    return sum(F.cross_entropy(z[None], torch.tensor([q["label"]], device=dev)) for (z, _), q in zip(out, rec["questions"])) / len(out)

opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

# RULE-2 one-batch check: gradients reach LoRA + head, not the frozen tower
m.train(); loss = step_loss(train.iloc[0]); loss.backward()
lora_g = [(n, p.grad) for n, p in m.named_parameters() if p.requires_grad and "lora" in n]
head_g = [p.grad for p in m.head.parameters()]
print(f"== GRAD CHECK loss={loss.item():.4f} lora tensors with grad {sum(g is not None and g.abs().sum() > 0 for _, g in lora_g)}/{len(lora_g)}"
      f" | head grads nonzero {all(g is not None and g.abs().sum() > 0 for g in head_g)} | visual grads {sum(p.grad is not None for p in vk.visual.parameters())}", flush=True)
opt.zero_grad(set_to_none=True)

if a.steps:
    t0 = time.time(); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    pool = []; step = 0; tot = 0.0
    m.train()
    while step < a.steps:
        if len(pool) < a.batch: extra = list(range(len(train))); random.shuffle(extra); pool += extra
        chunk, pool = pool[:a.batch], pool[a.batch:]
        opt.zero_grad(set_to_none=True)
        for i in chunk:
            l = step_loss(train.iloc[i]) / len(chunk); l.backward(); tot += l.item()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); step += 1
        if step % 50 == 0: print(f"step {step} mean loss {tot/50:.4f} ({time.time()-t0:.0f}s)", flush=True); tot = 0.0
    Yv, Pv = evaluate(valid); r = concept_report(Yv, Pv)
    np.savez(f"{a.out}/valid_probs_final.npz", **{c: Pv[c] for c in NAMES}, **{c + "_y": Yv[c] for c in NAMES})
    json.dump({"args": vars(a), "n_train_used": len(train), "valid": r, "init_source": src}, open(f"{a.out}/lowdata_log.json", "w"), indent=2)
    print(f"LOWDATA_DONE frac {a.frac} n={len(train)} init {a.init} seed {a.seed} valid mean AUC {r['mean_auc']:.4f} ece {r['mean_ece']:.4f}", flush=True)
    sys.exit(0)

log = []
Yv, Pv = evaluate(valid); r0 = concept_report(Yv, Pv)
print(f"epoch 0 (zero-shot, warm start) valid mean AUC {r0['mean_auc']:.4f} bacc {r0['mean_bacc']:.4f} ece {r0['mean_ece']:.4f}", flush=True)
log.append({"epoch": 0, "valid": r0}); best = (r0["mean_auc"], 0)
np.savez(f"{a.out}/valid_probs_epoch0.npz", **{c: Pv[c] for c in NAMES}, **{c + "_y": Yv[c] for c in NAMES})
for ep in range(1, a.epochs + 1):
    m.train(); order = list(range(len(train))); random.shuffle(order); t0 = time.time(); tot = 0.0
    for b in range(0, len(order), a.batch):
        opt.zero_grad(set_to_none=True)
        chunk = order[b:b + a.batch]
        for i in chunk:
            l = step_loss(train.iloc[i]) / len(chunk); l.backward(); tot += l.item() * len(chunk)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
    Yv, Pv = evaluate(valid); r = concept_report(Yv, Pv)
    print(f"epoch {ep} train loss {tot/len(train):.4f} ({time.time()-t0:.0f}s) valid mean AUC {r['mean_auc']:.4f} bacc {r['mean_bacc']:.4f} ece {r['mean_ece']:.4f}", flush=True)
    log.append({"epoch": ep, "train_loss": tot / len(train), "valid": r})
    # every epoch saved: the G0 decision uses the pre-fixed LAST epoch (no selection on valid)
    np.savez(f"{a.out}/valid_probs_epoch{ep}.npz", **{c: Pv[c] for c in NAMES}, **{c + "_y": Yv[c] for c in NAMES})
    if r["mean_auc"] > best[0]:
        best = (r["mean_auc"], ep)
        np.savez(f"{a.out}/valid_probs_best.npz", **{c: Pv[c] for c in NAMES}, **{c + "_y": Yv[c] for c in NAMES})
        torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}, f"{a.out}/best_trainable.pt")
torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}, f"{a.out}/final_trainable.pt")   # last epoch (the pre-registered one)
json.dump({"args": vars(a), "log": log, "best_epoch": best[1], "best_valid_mean_auc": best[0], "init_source": src}, open(f"{a.out}/g0_log.json", "w"), indent=2)
print("G0_TRAIN_DONE best epoch", best[1], "valid mean AUC", round(best[0], 4))
