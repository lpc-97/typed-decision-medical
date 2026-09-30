"""FINAL test-set analysis (design frozen 2026-09-29; each test split read once). Writes runs/test/FINAL_RESULTS.{json,md}.
CIs: Derm7pt & PubMedQA bootstrap over items; SLAKE/VQA-RAD bootstrap over IMAGES (clustered). B=2000, 95% percentile.
Pre-registered comparisons: B1 typed vs label-prob (same size), Kev-init vs base-init (per seed, 'helps' iff all 3 CI>0),
fine-tuned TCB vs frozen probes (Derm7pt), real Jev vs open arms (PubMedQA)."""
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from tdm.derm7pt import NAMES
from tdm.metrics import concept_report, ece
from tdm.derm_stats import boot as derm_boot, load as derm_load, mean_auc as derm_mauc
from tdm.vqa_stats import cluster_boot, summ

from tdm.paths import RUNS
T = os.path.join(RUNS, "test"); PQ = os.path.join(RUNS, "pubmedqa")
R = {"derm7pt": {}, "slake": {}, "vqarad": {}, "pubmedqa": {}}; md = ["# FINAL test results (one-time pass)\n"]

# ---------------- Derm7pt ----------------
D = {f"ft_kev_s{s}": f"{T}/derm_ft_kev_s{s}/test_probs.npz" for s in (1, 2, 3)}
D.update({f"ft_base_s{s}": f"{T}/derm_ft_base_s{s}/test_probs.npz" for s in (1, 2, 3)})
D.update({"zs_kev08b": f"{T}/derm_zs_kev08b/valid_probs.npz", "zs_kev4b": f"{T}/derm_zs_kev4b/valid_probs.npz",   # zs script's file name; split=test
          "zs_reflex4b": f"{T}/derm_zs_reflex4b/valid_probs.npz", "lp4b": f"{T}/derm_lp4b/test_probs.npz", "lp08b": f"{T}/derm_lp08b/test_probs.npz"})
D.update({f"probe_biomedclip_s{s}": f"{T}/derm_probe_biomedclip/valid_probs_seed{s}.npz" for s in (1, 2, 3)})
D.update({f"probe_qwenvis_s{s}": f"{T}/derm_probe_qwenvis/valid_probs_seed{s}.npz" for s in (1, 2, 3)})
arms = {k: derm_load(v) for k, v in D.items() if os.path.exists(v)}
Y0 = next(iter(arms.values()))[0]
for k, (Y, P) in arms.items():
    assert all((Y[c] == Y0[c]).all() for c in NAMES), k
    r = concept_report(Y, P); d, lo, hi = derm_boot(Y, P, {c: np.full_like(P[c], 1 / P[c].shape[1]) for c in NAMES}, seed=0)
    R["derm7pt"][k] = {"mean_auc": r["mean_auc"], "auc_ci": [lo + 0.5, hi + 0.5], "mean_bacc": r["mean_bacc"], "mean_ece": r["mean_ece"],
                       "per_crit_auc": {c: r[c]["auc"] for c in NAMES}}
md.append("## Derm7pt test (n=395), mean over 7 criteria\n| arm | AUC [95% CI] | bal.acc | ECE |\n|---|---|---|---|")
for k, v in R["derm7pt"].items():
    md.append(f"| {k} | {v['mean_auc']:.3f} [{v['auc_ci'][0]:.3f}, {v['auc_ci'][1]:.3f}] | {v['mean_bacc']:.3f} | {v['mean_ece']:.3f} |")
cmp = {}
def dcmp(a, b, seed):
    (Y, Pa), (_, Pb) = arms[a], arms[b]; return derm_boot(Y, Pa, Pb, seed=seed)
for s in (1, 2, 3):
    cmp[f"kev_vs_base_s{s}"] = dcmp(f"ft_kev_s{s}", f"ft_base_s{s}", s)
    cmp[f"kev_vs_biomedclip_s{s}"] = dcmp(f"ft_kev_s{s}", f"probe_biomedclip_s{s}", 10 + s)
    cmp[f"kev_vs_qwenvis_s{s}"] = dcmp(f"ft_kev_s{s}", f"probe_qwenvis_s{s}", 20 + s)
cmp["reflex4b_vs_lp4b"] = dcmp("zs_reflex4b", "lp4b", 31); cmp["kev4b_vs_lp4b"] = dcmp("zs_kev4b", "lp4b", 32); cmp["kev08b_vs_lp08b"] = dcmp("zs_kev08b", "lp08b", 33)
R["derm7pt"]["_comparisons"] = cmp
md.append("\n| comparison (dAUC) | mean [95% CI] |\n|---|---|")
md += [f"| {k} | {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] |" for k, v in cmp.items()]

# ---------------- SLAKE / VQA-RAD ----------------
def vload(p):
    z = np.load(p, allow_pickle=True); return z["p_yes"], z["y"], z["img"], (z["seen"] if "seen" in z.files else None)
for ds in ("slake", "vqarad"):
    A = {f"ft_{i}_s{s}": f"{T}/{ds}_ft_{i}_s{s}/test_preds.npz" for i in ("kev", "base") for s in (1, 2, 3)}
    A.update({f"zs_{a}": f"{T}/{ds}_zs_{a}/preds.npz" for a in ("kev08b", "kev4b", "reflex4b", "lp4b", "lp08b")})
    V = {k: vload(v) for k, v in A.items() if os.path.exists(v)}
    md.append(f"\n## {ds} official test (yes/no)\n| arm | acc | AUC [95% CI, image-clustered] | ECE | Brier |\n|---|---|---|---|---|")
    for k, (p, y, img, seen) in V.items():
        s_ = summ(p, y); _, lo, hi = cluster_boot(y, img, p); R[ds][k] = {**s_, "auc_ci": [lo, hi]}
        if seen is not None and seen.any() and (~seen).any():
            R[ds][k]["seen_acc"] = float(((p[seen] > .5) == y[seen]).mean()); R[ds][k]["unseen_acc"] = float(((p[~seen] > .5) == y[~seen]).mean())
            R[ds][k]["n_seen"], R[ds][k]["n_unseen"] = int(seen.sum()), int((~seen).sum())
        md.append(f"| {k} | {s_['acc']:.3f} | {s_['auc']:.3f} [{lo:.3f}, {hi:.3f}] | {s_['ece']:.3f} | {s_['brier']:.3f} |")
    c = {}
    for s in (1, 2, 3):
        pk, y, img, _ = V[f"ft_kev_s{s}"]; pb = V[f"ft_base_s{s}"][0]; c[f"kev_vs_base_s{s}"] = cluster_boot(y, img, pk, pb, seed=40 + s)
    for a, b in (("zs_reflex4b", "zs_lp4b"), ("zs_kev4b", "zs_lp4b"), ("zs_kev08b", "zs_lp08b")):
        pa, y, img, _ = V[a]; c[f"{a}_vs_{b}"] = cluster_boot(y, img, pa, V[b][0], seed=50)
    R[ds]["_comparisons"] = c
    md.append("\n| comparison (dAUC) | mean [95% CI] |\n|---|---|"); md += [f"| {k} | {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] |" for k, v in c.items()]
    if ds == "vqarad":
        md.append("\nVQA-RAD seen/unseen-image accuracy (fine-tuned): " + "; ".join(
            f"{k} {v['seen_acc']:.3f} (n={v['n_seen']}) / {v['unseen_acc']:.3f} (n={v['n_unseen']})" for k, v in R[ds].items() if isinstance(v, dict) and "seen_acc" in v))

# ---------------- PubMedQA ----------------
PA = {a: np.load(f"{PQ}/{a}/preds.npz") for a in ("jev", "reflex4b", "kev4b", "lp4b", "kev08b")}
md.append("\n## PubMedQA official test (n=500, text only)\n| arm | acc | macro-F1 | macro-AUC | ECE | Brier | median latency s |\n|---|---|---|---|---|---|---|")
for a, z in PA.items():
    rep = json.load(open(f"{PQ}/{a}/report.json")); R["pubmedqa"][a] = rep
    md.append(f"| {a} | {rep['acc']:.3f} | {rep['macro_f1']:.3f} | {rep['macro_auc']:.3f} | {rep['ece']:.3f} | {rep['brier']:.3f} | {rep['latency_median_s']:.3f} |")
def pq_boot(a, b, B=2000, seed=0):
    Pa, Pb, y = PA[a]["P"], PA[b]["P"], PA[a]["y"]; rng = np.random.default_rng(seed); n = len(y); d = []; e = []
    while len(d) < B:
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 3: continue
        d.append(f1_score(y[i], Pa[i].argmax(1), average="macro") - f1_score(y[i], Pb[i].argmax(1), average="macro"))
        e.append(ece(y[i], Pa[i]) - ece(y[i], Pb[i]))
    q = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"d_macro_f1": q(d), "d_ece": q(e)}
pc = {f"jev_vs_{b}": pq_boot("jev", b, seed=60) for b in ("reflex4b", "kev4b", "lp4b")}
R["pubmedqa"]["_comparisons"] = pc
jm = json.load(open(f"{PQ}/jev/jev_meta.json")); R["pubmedqa"]["jev_determinism"] = jm["determinism"]
md.append("\n| comparison | d macro-F1 [95% CI] | d ECE [95% CI] |\n|---|---|---|")
md += [f"| {k} | {v['d_macro_f1'][0]:+.3f} [{v['d_macro_f1'][1]:+.3f}, {v['d_macro_f1'][2]:+.3f}] | {v['d_ece'][0]:+.3f} [{v['d_ece'][1]:+.3f}, {v['d_ece'][2]:+.3f}] |" for k, v in pc.items()]
md.append(f"\nJev determinism (100 re-sent): {jm['determinism']}")
json.dump(R, open(f"{T}/FINAL_RESULTS.json", "w"), indent=2); open(f"{T}/FINAL_RESULTS.md", "w").write("\n".join(md) + "\n")
print("\n".join(md)); print("\nFINAL_ANALYSIS_DONE")
