"""Is the 6e-4 text gap ours, or Kev's own row-form vs prefix-cache gap on CUDA (fla TF32 rounding)?"""
import json, random, sys, torch, torch.nn.functional as F
from tdm.vision_kev import VisionKev
from kev.api import SystemOneRequest, to_record
import os
from tdm.paths import KEV_REPO
random.seed(0); vk = VisionKev.released()
lines = open(os.path.join(KEV_REPO, "evals/v7/decision-v7/development.jsonl")).read().splitlines()
recs = [json.loads(l) for l in random.sample(lines, 50)]
d_kev, d_ours_row, n = 0.0, 0.0, 0
for r in recs:
    rec, _ = to_record(SystemOneRequest(state=r["state"], questions={k: {kk: vv for kk, vv in q.items() if kk != "label"} for k, q in r["questions"].items()}))
    enc = vk.m.encode(vk.tok, rec)
    p_prefix = vk.m.probs(enc)                                   # Kev serving path: state once + rows from cache
    with torch.no_grad(): p_row = [F.softmax(z, -1).cpu() for z in vk.m.forward(enc)]   # Kev row form: state recomputed per row
    p_ours = vk.probs(rec)
    for a, b, c in zip(p_prefix, p_row, p_ours):
        d_kev = max(d_kev, float((a - b).abs().max())); d_ours_row = max(d_ours_row, float((b - c).abs().max())); n += 1
print(f"questions={n}  Kev prefix-vs-row max|dp|={d_kev:.2e}   ours-vs-Kev-row max|dp|={d_ours_row:.2e}")
