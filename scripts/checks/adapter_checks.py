"""S1 RULE-2 checks for tdm.vision_kev.VisionKev.
 (1) config echo; (2) text parity vs original Kev path on decision-v7 development (50 records);
 (3) image path: per-image decide states vary (cosine < 0.95), real vs blank image differ; image tokens count; vision frozen.
"""
import json, random, sys
import numpy as np, torch
from PIL import Image
from tdm.vision_kev import VisionKev
from kev.api import SystemOneRequest, to_record

import os
from tdm.paths import DERM7PT_IMAGES, DERM7PT_META, KEV_REPO
torch.manual_seed(0); random.seed(0)
vk = VisionKev.released()   # constructor changed after S1; released() = the S1 configuration
print("== CONFIG ECHO")
print("base", vk.meta.base, vk.meta.base_revision, "temperature", vk.m.head.temperature, "dtype", vk.m.dtype, "hybrid", vk.m.hybrid)
print("lm params", sum(p.numel() for p in vk.m.lm.parameters()) / 1e6, "M | visual params", sum(p.numel() for p in vk.visual.parameters()) / 1e6,
      "M | visual trainable", sum(p.numel() for p in vk.visual.parameters() if p.requires_grad))

# (2) text parity
lines = open(os.path.join(KEV_REPO, "evals/v7/decision-v7/development.jsonl")).read().splitlines()
recs = [json.loads(l) for l in random.sample(lines, 50)]
maxd, flips, nq = 0.0, 0, 0
for r in recs:
    rec, _ = to_record(SystemOneRequest(state=r["state"], questions={k: {kk: vv for kk, vv in q.items() if kk != "label"} for k, q in r["questions"].items()}))
    p_ref = vk.m.probs(vk.m.encode(vk.tok, rec))
    p_new = vk.probs(rec)
    for a, b in zip(p_ref, p_new):
        maxd = max(maxd, float((a - b).abs().max())); flips += int(a.argmax() != b.argmax()); nq += 1
print(f"== TEXT PARITY  questions={nq} max|dp|={maxd:.2e} argmax_flips={flips}")

# (3) image path on Derm7pt
meta = __import__("pandas").read_csv(os.path.join(DERM7PT_META, "meta.csv"))
rec = {"state": "Dermoscopic image of a skin lesion.",
       "questions": [{"instr": "Is a blue-whitish veil present in this lesion?", "options": ["no", "yes"], "label": 0},
                     {"instr": "Streaks in this lesion are:", "options": ["absent", "regular", "irregular"], "label": 0}]}
cases = meta.sample(20, random_state=0)["case_num"].tolist()
decs, pys = [], []
for cn in cases:
    im = Image.open(os.path.join(DERM7PT_IMAGES, f"{cn:04d}.png"))
    ps, ds = vk.probs(rec, im, return_decide=True)
    decs.append(torch.stack(ds)); pys.append(ps[0][1].item())
pv, grid = vk.image_inputs(Image.open(os.path.join(DERM7PT_IMAGES, f"{cases[0]:04d}.png")))
enc, n = vk.encode(rec, grid)
print("== IMAGE  grid", grid.tolist(), "image tokens", n, "state len", enc["seg"].count(0))
D = torch.stack([d[0] for d in decs]); Dn = torch.nn.functional.normalize(D, dim=-1); C = Dn @ Dn.T
off = C[~torch.eye(len(C), dtype=bool)]
print(f"decide-state cosine across 20 images: mean={off.mean():.4f} max={off.max():.4f}  (want < 0.95)")
Dc = torch.nn.functional.normalize(D - D.mean(0, keepdim=True), dim=-1); Cc = (Dc @ Dc.T)[~torch.eye(len(D), dtype=bool)]
print(f"  centered decide-state cosine: mean={Cc.mean():.4f} max={Cc.max():.4f}")
with torch.no_grad():
    Q = vk.m.head.q(D.to(vk.device)).cpu()
Qn = torch.nn.functional.normalize(Q, dim=-1); Cq = (Qn @ Qn.T)[~torch.eye(len(Q), dtype=bool)]
print(f"  pointer-query q(h) cosine: mean={Cq.mean():.4f} max={Cq.max():.4f} | top massive dims |h|:", D.abs().mean(0).topk(3).values.numpy().round(1).tolist(), "median", float(D.abs().mean(0).median()))
Dm = D.clone(); big = D.abs().mean(0).topk(3).indices; Dm[:, big] = 0
Dmn = torch.nn.functional.normalize(Dm, dim=-1); Cm = (Dmn @ Dmn.T)[~torch.eye(len(D), dtype=bool)]
print(f"  cosine without 3 massive dims: mean={Cm.mean():.4f} max={Cm.max():.4f}")
print("P(veil=yes) across images: min %.3f max %.3f std %.3f" % (min(pys), max(pys), float(np.std(pys))))
blank = Image.new("RGB", (448, 448), (128, 128, 128))
p_blank = vk.probs(rec, blank); p_text = vk.probs(rec)
print("blank-image P:", [p.numpy().round(3).tolist() for p in p_blank], "| text-only P:", [p.numpy().round(3).tolist() for p in p_text])
print("S1_CHECKS_DONE")
