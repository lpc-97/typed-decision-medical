"""B1 baseline on Derm7pt (audit 2609.32160): plain instruct Qwen3.5 (same family as reflex/Kev), chat template, the 7
criteria asked one at a time with lettered options (identical wording to tdm.derm7pt.CRITERIA), next-token P over the
option letters. Noul criteria use their 'no'/'yes' options as letters A/B like the others (uniform format).
Usage: derm_labelprob.py <hf model id> <split valid|test> <out>"""
import json, os, sys, time
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from tdm.derm7pt import CRITERIA, NAMES, STATE, load_split
from tdm.metrics import concept_report
from kev.checkpoint import resolve_run

mid, split, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
path = resolve_run(mid); proc = AutoProcessor.from_pretrained(path)
model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).cuda().eval(); tok = proc.tokenizer
LET = "ABCDEFG"
LID = [tok.encode(l, add_special_tokens=False)[0] for l in LET]
df = load_split(split); P = {c: [] for c in NAMES}; t0 = time.time()
for i, r in df.iterrows():
    img = Image.open(r["img"]).convert("RGB").resize((448, 448))
    for c in NAMES:
        _, instr, opts = CRITERIA[c]
        txt = f"{STATE}\n{instr}\n" + "\n".join(f"{LET[k]}. {o[1]}" for k, o in enumerate(opts)) + "\nAnswer with the letter of the correct option."
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
        with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
        P[c].append(torch.softmax(lg[LID[:len(opts)]], 0).cpu().numpy())
    if i % 50 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
Y = {c: df[c + "_y"].values for c in NAMES}; P = {c: np.stack(P[c]) for c in NAMES}
rep = concept_report(Y, P)
np.savez(f"{out}/{split}_probs.npz", **{c: P[c] for c in NAMES}, **{c + "_y": Y[c] for c in NAMES})
json.dump(rep, open(f"{out}/{split}_report.json", "w"), indent=2)
print(f"LABELPROB {mid} {split}: mean AUC {rep['mean_auc']:.4f} bacc {rep['mean_bacc']:.4f} ece {rep['mean_ece']:.4f}")
