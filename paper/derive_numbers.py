"""Derived numbers used in the paper text (means/SDs over seeds, ranges), computed ONLY from ../results.
Writes results/derived_numbers.json, which the orphan-number audit also reads."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("TDM_RESULTS", os.path.join(HERE, "..", "results")); J = lambda p: json.load(open(f"{R}/{p}"))
F = J("test/FINAL_RESULTS.json"); out = {}
_prev = J("derived_numbers.json") if os.path.exists(f"{R}/derived_numbers.json") else {}
HAVE_PRED = os.path.exists(f"{R}/pubmedqa/jev/preds.npz")   # per-item predictions are not part of the public release
ms = lambda v: {"mean": round(float(np.mean(v)), 6), "sd": round(float(np.std(v, ddof=1)), 6), "min": round(float(min(v)), 6), "max": round(float(max(v)), 6)}   # full precision; round ONCE when writing the paper
for ds, key in (("derm7pt", "mean_auc"), ("slake", "auc"), ("vqarad", "auc")):
    for arm in ("ft_kev", "ft_base", "probe_biomedclip", "probe_qwenvis"):
        ks = [f"{arm}_s{s}" for s in (1, 2, 3) if f"{arm}_s{s}" in F[ds]]
        if ks:
            out[f"test_{ds}_{arm}_auc"] = ms([F[ds][k][key] for k in ks])
            ek = "mean_ece" if ds == "derm7pt" else "ece"; out[f"test_{ds}_{arm}_ece"] = ms([F[ds][k][ek] for k in ks])
            if ds != "derm7pt": out[f"test_{ds}_{arm}_acc"] = ms([F[ds][k]["acc"] for k in ks])
    c = F[ds]["_comparisons"]; out[f"test_{ds}_kev_vs_base_dauc"] = ms([c[f"kev_vs_base_s{s}"][0] for s in (1, 2, 3)])
out["test_derm7pt_kev_vs_biomedclip_dauc"] = ms([F["derm7pt"]["_comparisons"][f"kev_vs_biomedclip_s{s}"][0] for s in (1, 2, 3)])
out["test_derm7pt_kev_vs_qwenvis_dauc"] = ms([F["derm7pt"]["_comparisons"][f"kev_vs_qwenvis_s{s}"][0] for s in (1, 2, 3)])
# validation (Derm7pt)
G = J("g0/g0_decision.json"); out["val_derm7pt_tcb_auc"] = ms([G["seeds"][s]["TCB"]["mean_auc"] for s in ("1", "2", "3")])
out["val_derm7pt_biomedclip_auc"] = ms([G["seeds"][s]["PROBE"]["mean_auc"] for s in ("1", "2", "3")])
A = J("g0/a_compare.json"); out["val_derm7pt_a1_auc"] = ms([r["A1_no_kev_init"] for r in A["A1_no_kev_init"]["rows"]])
out["val_derm7pt_qwenvis_auc"] = ms([r["A2_qwen_vision_probe"] for r in A["A2_qwen_vision_probe"]["rows"]])
LD = J("lowdata/lowdata_decision.json")
for f, v in LD["fractions"].items():
    out[f"val_lowdata_{f}_kev"] = ms([r["kev"] for r in v["rows"]]); out[f"val_lowdata_{f}_base"] = ms([r["base"] for r in v["rows"]])
S = J("vqa_slake_official_val_analysis.json")
out["val_slake_ft_kev_auc"] = ms([r["kev"]["auc"] for r in S["ft"]]); out["val_slake_ft_base_auc"] = ms([r["base"]["auc"] for r in S["ft"]])
out["val_slake_ft_kev_ece"] = ms([r["kev"]["ece"] for r in S["ft"]]); out["val_slake_ft_base_ece"] = ms([r["base"]["ece"] for r in S["ft"]])
# PubMedQA percentages
P = F["pubmedqa"]; out["pubmedqa_jev_acc_pct"] = round(100 * P["jev"]["acc"], 1)
out["pubmedqa_lp4b_pred_maybe"] = P["lp4b"]["pred_dist"][2]
if HAVE_PRED:
    out["pubmedqa_true_maybe"] = int((np.load(f"{R}/pubmedqa/jev/preds.npz")["y"] == 2).sum())
    out["pubmedqa_true_dist"] = np.bincount(np.load(f"{R}/pubmedqa/jev/preds.npz")["y"], minlength=3).tolist()
else:
    out["pubmedqa_true_maybe"], out["pubmedqa_true_dist"] = _prev["pubmedqa_true_maybe"], _prev["pubmedqa_true_dist"]
# leakage percentages
L = J("leakage_stats.json")
for k, v in L.items():
    for kk in ("seen_yesno_train", "seen_anytype_train"):
        if kk in v: out[f"leak_{k}_{kk}_pct"] = round(100 * v[kk] / v["n_questions"], 1)
# option-name test (validation): meaning-preserving rewrites = C1, C2, C4, C5
MP = ("C1_neutral", "C2_random", "C4_defonly", "C5_reversed")
fl = lambda d: [J(d)["conditions"][c]["flips_per_100_vs_C0"] for c in MP]
zs = sum([fl(f"optname/{m}/option_name_report.json") for m in ("kev08b_zs", "kev4b_zs", "reflex4b_zs")], [])
ftk = sum([fl(f"optname3/ft_kev_s{s}/option_name_report.json") for s in (1, 2, 3)], [])
ftb = sum([fl(f"optname3/ft_base_s{s}/option_name_report.json") for s in (1, 2, 3)], [])
out["optname_mp_flips_zs"] = ms(zs); out["optname_mp_flips_ft_kev"] = ms(ftk); out["optname_mp_flips_ft_base"] = ms(ftb)
out["optname_mp_flips_all"] = ms(zs + ftk + ftb)
O3 = J("optname3/optname3_decision.json")["seeds"]
out["optname3_did"] = ms([O3[s]["DiD"][0] for s in ("1", "2", "3")])
out["optname3_kev_swap"] = ms([O3[s]["kev_swap_dAUC"][0] for s in ("1", "2", "3")]); out["optname3_base_swap"] = ms([O3[s]["base_swap_dAUC"][0] for s in ("1", "2", "3")])
out["optname3_kev_swapflips"] = ms([O3[s]["kev_flips"] for s in ("1", "2", "3")]); out["optname3_base_swapflips"] = ms([O3[s]["base_flips"] for s in ("1", "2", "3")])
# ---- point estimates of every reported difference (full-sample difference; CIs stay the bootstrap percentiles) ----
PT = {}
for ds, key in (("derm7pt", "mean_auc"), ("slake", "auc"), ("vqarad", "auc")):
    A = {k: v for k, v in F[ds].items() if not k.startswith("_")}
    for s in (1, 2, 3):
        PT[f"test_{ds}_kev_vs_base_s{s}"] = A[f"ft_kev_s{s}"][key] - A[f"ft_base_s{s}"][key]
        if ds == "derm7pt":
            PT[f"test_derm7pt_kev_vs_qwenvis_s{s}"] = A[f"ft_kev_s{s}"][key] - A[f"probe_qwenvis_s{s}"][key]
            PT[f"test_derm7pt_kev_vs_biomedclip_s{s}"] = A[f"ft_kev_s{s}"][key] - A[f"probe_biomedclip_s{s}"][key]
    zs = "" if ds == "derm7pt" else "zs_"
    PT[f"test_{ds}_reflex4b_vs_lp4b"] = A["zs_reflex4b"][key] - A[f"{zs}lp4b"][key]
    PT[f"test_{ds}_kev4b_vs_lp4b"] = A["zs_kev4b"][key] - A[f"{zs}lp4b"][key]
    PT[f"test_{ds}_kev08b_vs_lp08b"] = A["zs_kev08b"][key] - A[f"{zs}lp08b"][key]
for b in ("reflex4b", "kev4b", "lp4b"):
    PT[f"pubmedqa_jev_vs_{b}_f1"] = P["jev"]["macro_f1"] - P[b]["macro_f1"]; PT[f"pubmedqa_jev_vs_{b}_ece"] = P["jev"]["ece"] - P[b]["ece"]
rc = lambda d: J(f"{d}/option_name_report.json")["conditions"]
for s in (1, 2, 3):
    k, b = rc(f"optname3/ft_kev_s{s}"), rc(f"optname3/ft_base_s{s}")
    PT[f"optname_kev_swap_s{s}"] = k["C3_swap"]["mean_auc"] - k["C0_orig"]["mean_auc"]
    PT[f"optname_base_swap_s{s}"] = b["C3_swap"]["mean_auc"] - b["C0_orig"]["mean_auc"]
    PT[f"optname_did_s{s}"] = PT[f"optname_kev_swap_s{s}"] - PT[f"optname_base_swap_s{s}"]
z = rc("optname/kev4b_zs"); PT["optname_kev4b_zs_swap"] = z["C3_swap"]["mean_auc"] - z["C0_orig"]["mean_auc"]
for s, r in zip((1, 2, 3), J("g0/a_compare.json")["A1_no_kev_init"]["rows"]):
    PT[f"val_derm7pt_kev_vs_base_s{s}"] = r["TCB"] - r["A1_no_kev_init"]
for f, v in LD["fractions"].items():
    for r in v["rows"]: PT[f"val_lowdata_{f}_kev_vs_base_s{r['seed']}"] = r["kev"] - r["base"]
for f in ("0.1", "0.25", "0.5"):
    PT[f"val_lowdata_{f}_mean_gap"] = float(np.mean([PT[f"val_lowdata_{f}_kev_vs_base_s{s}"] for s in (1, 2, 3)]))
PT["test_derm7pt_kev_vs_base_mean"] = float(np.mean([PT[f"test_derm7pt_kev_vs_base_s{s}"] for s in (1, 2, 3)]))
PT["optname_kev_swap_mean"] = float(np.mean([PT[f"optname_kev_swap_s{s}"] for s in (1, 2, 3)]))
PT["optname_base_swap_mean"] = float(np.mean([PT[f"optname_base_swap_s{s}"] for s in (1, 2, 3)]))
out["point"] = {k: round(float(v), 6) for k, v in PT.items()}
# VQA-RAD fine-tuned over-confidence, split by whether the test image occurs in the yes/no training questions
import sys as _sys
def _ece(y, p, bins=10):
    P = np.stack([1 - p, p], 1); conf = P.max(1); acc = (P.argmax(1) == y).astype(float); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)
OC = {}
for init in (("kev", "base") if HAVE_PRED else ()):
    for s in (1, 2, 3):
        z = np.load(f"{R}/test/vqarad_ft_{init}_s{s}/test_preds.npz", allow_pickle=True); p, y, seen = z["p_yes"], z["y"], z["seen"]
        conf = np.maximum(p, 1 - p); acc = ((p > .5) == y)
        OC[f"{init}_s{s}"] = {"mean_conf": round(float(conf.mean()), 4), "acc": round(float(acc.mean()), 4),
                              "ece_seen": round(_ece(y[seen], p[seen]), 4), "ece_unseen": round(_ece(y[~seen], p[~seen]), 4),
                              "n_seen": int(seen.sum()), "n_unseen": int((~seen).sum())}
out["vqarad_ft_overconfidence"] = OC if HAVE_PRED else _prev["vqarad_ft_overconfidence"]
json.dump(out, open(f"{R}/derived_numbers.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
