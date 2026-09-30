"""PubMedQA (reasoning-required setting: question + abstract contexts, no long answer) — official 500-question test
(pubmedqa/pubmedqa@1cbae8e, MIT). Choice over {yes, no, maybe}. Zero-shot only. Arms:
  jev            official hosted Jev (api.typesafe.ai /v1/systemone, model jev-latest); key read from env, never printed
  kev:<run>      Kev text decision model (kev.checkpoint, served temperature)
  reflex         reflex@231f896 stable config (frozen Qwen3.5-4B, 2 orders)
  labelprob:<m>  B1: plain instruct Qwen3.5, chat template, next-token logits of yes/no/maybe
Metrics: accuracy, macro-F1 (the official PubMedQA metrics), macro-AUC, ECE, Brier; latency per request.
Jev extra: 100 questions re-sent to measure determinism (audit 2609.32160 reported non-determinism)."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from tdm.paths import PUBMEDQA_DIR
arm, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
D = PUBMEDQA_DIR
pq = json.load(open(f"{D}/ori_pqal.json")); gt = json.load(open(f"{D}/test_ground_truth.json"))
ids = sorted(gt)
LAB = ["yes", "no", "maybe"]
y = np.array([LAB.index(gt[i]) for i in ids])
INSTR = "Based on the abstract, what is the answer to the research question?"
def state(i):
    r = pq[i]; return f"Research question: {r['QUESTION']}\nAbstract: " + " ".join(r["CONTEXTS"])
P, lat = [], []
t0 = time.time()

if arm == "jev":
    import urllib.request
    KEY = os.environ["TYPESAFE_API_KEY"]
    def ask(i):
        body = json.dumps({"model": "jev-latest", "state": state(i), "questions": {"answer": {"type": "choice", "instructions": INSTR,
                           "criteria": {"yes": None, "no": None, "maybe": None}}}}).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=body, method="POST",
                                             headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
                t = time.time(); r = json.loads(urllib.request.urlopen(req, timeout=60).read()); dt = time.time() - t
                a = r["answers"]["answer"]; pr = a.get("probabilities") or {}
                return [float(pr.get(k, 0.0)) for k in LAB], dt, r.get("model"), a
            except Exception as e:
                err = repr(e)[:200]; time.sleep(2 * (attempt + 1))
        return None, None, None, err
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(ask, ids))
    bad = [i for i, r in zip(ids, res) if r[0] is None]
    print("failed requests:", len(bad), res[[r[0] is None for r in res].index(True)][3] if bad else "", flush=True)
    first = next(r for r in res if r[0] is not None); print("served model:", first[2], "| answer object keys:", list(first[3].keys()), flush=True)
    P = [r[0] if r[0] is not None else [1 / 3] * 3 for r in res]; lat = [r[1] for r in res if r[1] is not None]
    # determinism: resend first 100
    with ThreadPoolExecutor(8) as ex:
        res2 = list(ex.map(ask, ids[:100]))
    a1 = np.array([r[0] for r in res[:100]]); a2 = np.array([r[0] if r[0] is not None else [np.nan] * 3 for r in res2])
    ok = ~np.isnan(a2).any(1)
    det = {"n": int(ok.sum()), "argmax_changed": int((a1[ok].argmax(1) != a2[ok].argmax(1)).sum()), "max_abs_dp": float(np.abs(a1[ok] - a2[ok]).max())}
    print("determinism (100 resent):", det, flush=True)
    json.dump({"failed": bad, "determinism": det, "served_model": first[2]}, open(f"{out}/jev_meta.json", "w"), indent=2)
elif arm.startswith("kev:"):
    import torch
    from kev.checkpoint import load as kload
    from kev.model import SERVE_MAX_STATE, SERVE_MAX_BRANCH   # serving context: the 384-token TRAINING state limit would truncate abstracts
    tok, m = kload(arm[4:], "cuda")
    for i in ids:
        rec = {"state": state(i), "questions": [{"instr": INSTR, "options": LAB, "label": 0}]}
        t = time.time(); enc = m.encode(tok, rec, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH); assert not enc["state_truncated"]
        P.append(m.probs(enc)[0].numpy().tolist()); lat.append(time.time() - t)
elif arm == "reflex":
    from reflex import Engine, SystemOneRequest
    from reflex.serving import engine_kwargs, load_stable
    eng = Engine.load(**engine_kwargs(load_stable()))
    for i in ids:
        t = time.time()
        a = eng.answer(SystemOneRequest(state=state(i), questions={"answer": {"type": "choice", "instructions": INSTR, "criteria": {k: None for k in LAB}}}))
        lat.append(time.time() - t); pr = a.answers["answer"].probabilities; P.append([pr[k] for k in LAB])
elif arm.startswith("labelprob:"):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from kev.checkpoint import resolve_run
    path = resolve_run(arm.split(":", 1)[1]); proc = AutoProcessor.from_pretrained(path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).cuda().eval(); tok = proc.tokenizer
    V = [sorted({tok.encode(w, add_special_tokens=False)[0] for w in (k, k.capitalize(), " " + k, " " + k.capitalize())}) for k in LAB]
    for i in ids:
        msgs = [{"role": "user", "content": [{"type": "text", "text": f"{state(i)}\n\n{INSTR}\nAnswer with yes, no, or maybe."}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False).to("cuda")
        t = time.time()
        with torch.no_grad(): lg = model(**inp).logits[0, -1].float()
        lat.append(time.time() - t)
        z = torch.stack([torch.logsumexp(lg[v], 0) for v in V]); P.append(torch.softmax(z, 0).cpu().numpy().tolist())
else:
    raise SystemExit(arm)

P = np.array(P, float); P = P / P.sum(1, keepdims=True)
from sklearn.metrics import f1_score, roc_auc_score
from tdm.metrics import ece
rep = {"arm": arm, "n": len(y), "acc": float((P.argmax(1) == y).mean()), "macro_f1": float(f1_score(y, P.argmax(1), average="macro")),
       "macro_auc": float(np.mean([roc_auc_score((y == k).astype(int), P[:, k]) for k in range(3)])), "ece": ece(y, P),
       "brier": float(((P - np.eye(3)[y]) ** 2).sum(1).mean()), "pred_dist": np.bincount(P.argmax(1), minlength=3).tolist(),
       "latency_median_s": float(np.median(lat)) if lat else None, "wall_s": time.time() - t0}
np.savez(f"{out}/preds.npz", P=P, y=y, ids=np.array(ids))
json.dump(rep, open(f"{out}/report.json", "w"), indent=2)
print("PUBMEDQA", json.dumps(rep))
