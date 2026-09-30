"""Option-name invariance test (audit checklist item T3, cf. arXiv 2609.26758), pre-registered 2026-09-29 before running.
5 Choice criteria of Derm7pt (the two Noul ones have no definitions to decouple). Ground truth always follows the DEFINITION.
Conditions (per option k with name n_k and definition d_k):
  C0 orig      "n_k: d_k"
  C1 neutral   "A/B/C: d_k"
  C2 random    "<random 3 letters>: d_k"
  C3 swap      "n_{(k+1)%K}: d_k"        names cyclically displaced, definitions fixed  (the key test)
  C4 defonly   "d_k"
  C5 reversed  C0 with option order reversed (T2 order check)
Verdict per model: FOLLOWS NAME iff paired-bootstrap dAUC(C3-C0) CI < 0 AND the C3 drop exceeds the C1 drop.
Also reported: argmax flip rate vs C0 (decisions per 100), and for C3 the share of decisions whose argmax option carries
the NAME of C0's predicted class ("name-following rate").
Valid split only (n=203); no training."""
import json, os, random, sys, time
import numpy as np, torch

from tdm.derm7pt import CRITERIA, STATE, load_split
from tdm.derm_stats import auc_bin, boot

CH = [c for c in CRITERIA if CRITERIA[c][0] == "choice"]
CONDS = ["C0_orig", "C1_neutral", "C2_random", "C3_swap", "C4_defonly", "C5_reversed"]
rng = random.Random(0)
RAND = {c: ["".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(3)) for _ in CRITERIA[c][2]] for c in CH}


def options(c, cond):
    """-> (option texts in presentation order, map presentation index -> definition index)"""
    opts = CRITERIA[c][2]; K = len(opts)
    names = [o[0] for o in opts]; defs = [o[1].split(": ", 1)[1] for o in opts]
    order = list(range(K))
    if cond == "C0_orig": txt = [f"{names[k]}: {defs[k]}" for k in range(K)]
    elif cond == "C1_neutral": txt = [f"{'ABC'[k]}: {defs[k]}" for k in range(K)]
    elif cond == "C2_random": txt = [f"{RAND[c][k]}: {defs[k]}" for k in range(K)]
    elif cond == "C3_swap": txt = [f"{names[(k + 1) % K]}: {defs[k]}" for k in range(K)]
    elif cond == "C4_defonly": txt = list(defs)
    elif cond == "C5_reversed":
        order = order[::-1]; txt = [f"{names[k]}: {defs[k]}" for k in order]
    return txt, order


def macro(y, p):
    return float(np.mean([auc_bin((y == k).astype(int), p[:, k]) for k in np.unique(y)]))


which, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
va = load_split("valid")
Y = {c: va[c + "_y"].values for c in CH}
P = {cond: {c: [] for c in CH} for cond in CONDS}
t0 = time.time()

if which == "reflex":
    from reflex import Engine, SystemOneRequest
    from reflex.serving import engine_kwargs, load_stable
    eng = Engine.load(**engine_kwargs(load_stable()))
    def run(img_path, cond):
        qs, maps = {}, {}
        for c in CH:
            txt, order = options(c, cond)
            keys = [f"opt{i}" for i in range(len(txt))]   # reflex needs criteria keys; key = the option name part
            crit = {}
            for i, t in enumerate(txt):
                name, _, d = t.partition(": ") if ": " in t else (t, "", "")
                crit[name if d else f"option {i + 1}"] = d if d else t
            qs[c] = {"type": "choice", "instructions": CRITERIA[c][1], "criteria": crit}; maps[c] = (list(crit), order)
        r = eng.answer(SystemOneRequest(state={"image": {"type": "image", "source": img_path}, "description": STATE}, questions=qs))
        res = {}
        for c in CH:
            ks, order = maps[c]; pr = r.answers[c].probabilities
            p = np.zeros(len(ks))
            for i, k in enumerate(ks): p[order[i]] = pr[k]
            res[c] = p / p.sum()
        return res
else:
    from PIL import Image
    from tdm.vision_kev import VisionKev, load_trainable
    if which.startswith("ft:"):   # ft:<init kev|base>:<path to best_trainable.pt>
        _, init, path = which.split(":", 2)
        tok, m, meta, _ = load_trainable(warm=(init == "kev"), checkpointing=False)
        sd = torch.load(path, map_location="cpu"); missing = [n for n, p in m.named_parameters() if p.requires_grad and n not in sd]
        assert not missing, missing[:3]
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in sd: p.copy_(sd[n])
        m.eval(); vk = VisionKev(tok, m, meta)
    else:
        vk = VisionKev.released(which)
    feats = {}
    def run(img_path, cond):
        if img_path not in feats:
            pv, g = vk.image_inputs(Image.open(img_path)); feats[img_path] = (vk.image_features(pv, g), g)
        f, g = feats[img_path]
        rec = {"state": STATE, "questions": []}; orders = []
        for c in CH:
            txt, order = options(c, cond)
            rec["questions"].append({"instr": CRITERIA[c][1], "options": txt, "label": 0}); orders.append(order)
        with torch.no_grad():
            out_ = vk.logits(rec, f, g)
        res = {}
        for c, (z, _), order in zip(CH, out_, orders):
            pr = torch.softmax(z, -1).cpu().numpy(); p = np.zeros(len(pr))
            for i, k in enumerate(order): p[k] = pr[i]
            res[c] = p
        return res

for i, path in enumerate(va["img"]):
    for cond in CONDS:
        r = run(path, cond)
        for c in CH: P[cond][c].append(r[c])
    if i % 50 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
P = {cond: {c: np.array(P[cond][c]) for c in CH} for cond in CONDS}
np.savez(f"{out}/probs.npz", **{f"{cond}__{c}": P[cond][c] for cond in CONDS for c in CH}, **{c + "_y": Y[c] for c in CH})

# --- analysis ---
Yd = {c: Y[c] for c in CH}
def mauc(Pc, idx=None):
    idx = np.arange(len(Y[CH[0]])) if idx is None else idx
    return float(np.mean([macro(Y[c][idx], Pc[c][idx]) for c in CH]))
import tdm.derm_stats as a_compare
a_compare.NAMES = CH   # bootstrap over the 5 choice criteria
res = {"model": which, "conditions": {}}
base = P["C0_orig"]
for cond in CONDS:
    Pc = P[cond]
    flips = np.mean([np.mean(Pc[c].argmax(1) != base[c].argmax(1)) for c in CH]) * 100
    row = {"mean_auc": mauc(Pc), "flips_per_100_vs_C0": float(flips),
           "per_crit_auc": {c: macro(Y[c], Pc[c]) for c in CH}}
    if cond != "C0_orig":
        d, lo, hi = boot(Yd, Pc, base, seed=7)
        row["dAUC_vs_C0"] = [d, lo, hi]
    if cond == "C3_swap":   # name-following: argmax lands on the option carrying the NAME of C0's predicted class
        nf = []
        for c in CH:
            K = len(CRITERIA[c][2]); pred0 = base[c].argmax(1); pred3 = Pc[c].argmax(1)
            # in C3, option k carries name (k+1)%K -> the option named like class j is k=(j-1)%K
            nf.append(np.mean(pred3 == (pred0 - 1) % K))
        row["name_following_rate"] = float(np.mean(nf))
    res["conditions"][cond] = row
    extra = f" dAUC {row['dAUC_vs_C0'][0]:+.3f} [{row['dAUC_vs_C0'][1]:+.3f},{row['dAUC_vs_C0'][2]:+.3f}]" if "dAUC_vs_C0" in row else ""
    nfs = f" name-following {row['name_following_rate']:.1%}" if "name_following_rate" in row else ""
    print(f"{cond:12s} meanAUC {row['mean_auc']:.3f}  flips/100 {row['flips_per_100_vs_C0']:5.1f}{extra}{nfs}")
c3, c1 = res["conditions"]["C3_swap"], res["conditions"]["C1_neutral"]
follows = c3["dAUC_vs_C0"][2] < 0 and c3["dAUC_vs_C0"][0] < c1["dAUC_vs_C0"][0]
res["verdict"] = "FOLLOWS OPTION NAME" if follows else "definition-faithful (no significant name effect)"
json.dump(res, open(f"{out}/option_name_report.json", "w"), indent=2)
print("VERDICT", which, "->", res["verdict"], f"({time.time() - t0:.0f}s)")
