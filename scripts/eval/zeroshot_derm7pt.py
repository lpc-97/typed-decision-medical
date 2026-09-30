"""Zero-shot: open-source Jev-style models used AS-IS on Derm7pt valid (n=203), no derm training.
  reflex   : kshetrajna12/reflex@231f896 (MIT), its recommended serving/stable.json config (frozen Qwen3.5-4B, 2 option orders)
  kev-4b   : jaredpalmer/kev-4b + the base's own vision tower via tdm.vision_kev.VisionKev (our smallest-diff adapter)
Same questions/options as training (tdm.derm7pt.CRITERIA); metrics = tdm.metrics.concept_report."""
import json, os, sys, time
import numpy as np

from tdm.derm7pt import CRITERIA, NAMES, STATE, load_split, record
from tdm.metrics import concept_report

which, out = sys.argv[1], sys.argv[2]
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "valid"
os.makedirs(out, exist_ok=True)
va = load_split(SPLIT)
P = {c: [] for c in NAMES}; Y = {c: va[c + "_y"].values for c in NAMES}
t0 = time.time()

if which == "reflex":
    from reflex import Engine, SystemOneRequest
    from reflex.serving import engine_kwargs, load_stable
    kw = engine_kwargs(load_stable()); print("reflex config", kw, flush=True)
    eng = Engine.load(**kw)
    qs = {}
    for c in NAMES:
        qtype, instr, opts = CRITERIA[c]
        if qtype == "noul":
            qs[c] = {"type": "noul", "instructions": instr}
        else:
            qs[c] = {"type": "choice", "instructions": instr, "criteria": {name: text.split(": ", 1)[1] for name, text, _, _ in opts}}
    for i, p in enumerate(va["img"]):
        r = eng.answer(SystemOneRequest(state={"image": {"type": "image", "source": p}, "description": STATE}, questions=qs))
        for c in NAMES:
            a = r.answers[c]
            if CRITERIA[c][0] == "noul":
                P[c].append([1 - a.noul, a.noul])
            else:
                pr = a.probabilities; P[c].append([pr[name] for name, *_ in CRITERIA[c][2]])
        if i % 50 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
else:
    from PIL import Image
    import torch
    from tdm.vision_kev import VisionKev
    vk = VisionKev.released(which)   # e.g. jaredpalmer/kev-4b, served temperature
    print("run", which, "base", vk.meta.base, "T", vk.m.head.temperature, flush=True)
    for i, p in enumerate(va["img"]):
        ps = vk.probs(record(), Image.open(p))
        for c, pr in zip(NAMES, ps): P[c].append(pr.numpy())
        if i % 50 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)

P = {c: np.array(P[c], float) for c in NAMES}
r = concept_report(Y, P)
np.savez(f"{out}/valid_probs.npz", **{c: P[c] for c in NAMES}, **{c + "_y": Y[c] for c in NAMES})
json.dump(r, open(f"{out}/report.json", "w"), indent=2)
for c in NAMES: print(f"  {c:22s} AUC {r[c]['auc']:.3f} bacc {r[c]['bacc']:.3f} ece {r[c]['ece']:.3f}")
print(f"ZS {which}: mean AUC {r['mean_auc']:.4f} bacc {r['mean_bacc']:.4f} ece {r['mean_ece']:.4f} ({time.time()-t0:.0f}s)")
