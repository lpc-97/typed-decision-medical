"""Zero-shot yes/no on VQA-RAD / SLAKE (no training). Arms:
  kev:<run>      Kev decision model + base vision tower (tdm.vision_kev.VisionKev), Noul question
  reflex         kshetrajna12/reflex@231f896 stable config (frozen Qwen3.5-4B, 2 option orders), Noul question
  labelprob:<m>  B1 baseline (audit 2609.32160): the plain instruct VLM <m>, chat template, next-token P(yes) vs P(no)
Usage: vqa_zeroshot.py <arm> <ds vqarad|slake> <split resplit:val|resplit:test|official:test> <out>
Saves per-question P(yes) with image ids (for image-clustered bootstrap)."""
import json, os, sys, time
import numpy as np, torch

from tdm.vqa import STATE, image, load, record

from tdm.paths import CACHE
arm, ds, split, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
kind, name = split.split(":")
df = load(ds); df = df[df[kind] == name].reset_index(drop=True)
IMGDIR = os.path.join(CACHE, f"{ds}_imgs"); os.makedirs(IMGDIR, exist_ok=True)
t0 = time.time(); p_yes = []

if arm.startswith("kev:"):
    from tdm.vision_kev import VisionKev
    vk = VisionKev.released(arm[4:])
    cache = {}
    for i, r in df.iterrows():
        if r.img_md5 not in cache:
            pv, g = vk.image_inputs(image(r)); cache[r.img_md5] = (vk.image_features(pv, g), g)
        f, g = cache[r.img_md5]
        with torch.no_grad():
            (z, _), = vk.logits(record(r.question), f, g)
        p_yes.append(float(torch.softmax(z, -1)[1]))
elif arm == "reflex":
    from reflex import Engine, SystemOneRequest
    from reflex.serving import engine_kwargs, load_stable
    eng = Engine.load(**engine_kwargs(load_stable()))
    for i, r in df.iterrows():
        path = f"{IMGDIR}/{r.img_md5}.png"
        if not os.path.exists(path): image(r).save(path)
        a = eng.answer(SystemOneRequest(state={"image": {"type": "image", "source": path}, "description": STATE},
                                        questions={"q": {"type": "noul", "instructions": r.question}}))
        p_yes.append(float(a.answers["q"].noul))
elif arm.startswith("labelprob:"):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from kev.checkpoint import resolve_run
    path = resolve_run(arm.split(":", 1)[1])
    proc = AutoProcessor.from_pretrained(path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    tok = proc.tokenizer
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    YES, NO = ids(["yes", "Yes", " yes", " Yes"]), ids(["no", "No", " no", " No"])
    print("yes ids", YES, "no ids", NO, flush=True)
    for i, r in df.iterrows():
        msgs = [{"role": "user", "content": [{"type": "image", "image": image(r).resize((448, 448))},
                                             {"type": "text", "text": f"{r.question}\nAnswer with yes or no."}]}]
        try:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
        except TypeError:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        if i == 0: print("prompt tail:", repr(tok.decode(inp["input_ids"][0][-24:])), flush=True)
        with torch.no_grad():
            lg = model(**inp.to("cuda")).logits[0, -1].float()
        ly, ln = torch.logsumexp(lg[YES], 0), torch.logsumexp(lg[NO], 0)
        p_yes.append(float(torch.sigmoid(ly - ln)))
else:
    raise SystemExit(f"unknown arm {arm}")

p = np.array(p_yes); y = df["y"].values
from sklearn.metrics import roc_auc_score
from tdm.metrics import ece
P2 = np.stack([1 - p, p], 1)
r = {"arm": arm, "ds": ds, "split": split, "n": int(len(y)), "images": int(df.img_md5.nunique()),
     "acc": float(((p > 0.5) == y).mean()), "auc": float(roc_auc_score(y, p)), "ece": ece(y, P2),
     "brier": float(((p - y) ** 2).mean()), "mean_p_yes": float(p.mean()), "sec": time.time() - t0}
np.savez(f"{out}/preds.npz", p_yes=p, y=y, img=df["img_md5"].values, qid=df["qid"].values)
json.dump(r, open(f"{out}/report.json", "w"), indent=2)
print("ZS", json.dumps(r))
