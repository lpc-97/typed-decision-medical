"""All paper figures, read ONLY from ../results (aggregated result files). Vector PDF output in figs/.
Palette: fixed colorblind-checked categorical palette (light background); each entity keeps one color across figures."""
import json, os
import numpy as np
import matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.environ.get("TDM_RESULTS", os.path.join(HERE, "..", "results")); O = os.path.join(HERE, "figs"); os.makedirs(O, exist_ok=True)
J = lambda p: json.load(open(f"{R}/{p}"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "pdf.fonttype": 42,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False})
INK2 = "#52514e"; GRID = "#e6e5e0"
C = {"ft_kev": "#2a78d6", "ft_base": "#eb6834", "zs_kev08b": "#1baf7a", "zs_kev4b": "#4a3aa7", "zs_reflex4b": "#e87ba4",
     "lp4b": "#eda100", "lp08b": "#b8a26a", "probe_qwenvis": "#008300", "probe_biomedclip": "#8a8a85", "jev": "#e34948"}
LBL = {"ft_kev": "Fine-tuned, Kev init (0.8B)", "ft_base": "Fine-tuned, base init (0.8B)", "zs_kev08b": "Kev-0.8B + vision (zero-shot)",
       "zs_kev4b": "Kev-4B + vision (zero-shot)", "zs_reflex4b": "reflex-4B (zero-shot)", "lp4b": "Label-prob Qwen3.5-4B (zero-shot)",
       "lp08b": "Label-prob Qwen3.5-0.8B (zero-shot)", "probe_qwenvis": "Qwen vision + linear probe", "probe_biomedclip": "BiomedCLIP + linear probe",
       "jev": "Jev (hosted, text only)"}
F = J("test/FINAL_RESULTS.json")
PT = J("derived_numbers.json")["point"]   # full-sample point estimates; CIs are bootstrap percentiles
pv = lambda ci, key: (PT[key], ci[1], ci[2])
def save(fig, name):
    fig.savefig(f"{O}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{O}/{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Fig 2: test landscape (AUC + 95% CI), three image datasets ----------
fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
order = ["ft_kev", "ft_base", "probe_qwenvis", "probe_biomedclip", "zs_kev4b", "zs_reflex4b", "lp4b", "zs_kev08b", "lp08b"]
for ax, (ds, title) in zip(axs, (("derm7pt", "a  Derm7pt, 7 criteria (n=395)"), ("slake", "b  SLAKE yes/no (n=355)"), ("vqarad", "c  VQA-RAD yes/no (n=251)"))):
    d = F[ds]; key = "mean_auc" if ds == "derm7pt" else "auc"
    for yi, arm in enumerate(order):
        ks = [k for k in d if not k.startswith("_") and (k == arm or k == f"zs_{arm}" or k.startswith(arm + "_s"))]
        if ds != "derm7pt": ks = [k for k in d if not k.startswith("_") and (k == arm or k == f"zs_{arm}" or k.startswith(arm + "_s"))]
        if not ks: continue
        vals = [d[k][key] for k in ks]; cis = [d[k]["auc_ci"] for k in ks]
        jit = np.linspace(-0.18, 0.18, len(ks)) if len(ks) > 1 else [0]
        for v, ci, j in zip(vals, cis, jit):
            ax.plot(ci, [yi + j] * 2, color=C[arm], lw=1.2, solid_capstyle="round")
            ax.plot(v, yi + j, "o", ms=4.2, color=C[arm], mec="white", mew=0.6, zorder=3)
    ax.set_title(title, loc="left", fontsize=7.8); ax.set_xlabel("AUC (95% CI)"); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([LBL[a] for a in order]); axs[0].invert_yaxis()
for ax in axs: ax.set_ylim(len(order) - 0.5, -0.5)
fig.text(0.99, -0.02, "Fine-tuned and probe arms: one marker per seed (3 seeds). Probes are Derm7pt only.", ha="right", color=INK2, fontsize=6.5)
fig.tight_layout(); save(fig, "fig2_landscape")

# ---------- Fig 3: forest plot of pre-registered paired comparisons (test) ----------
rows = []
for ds, nm in (("derm7pt", "Derm7pt"), ("slake", "SLAKE"), ("vqarad", "VQA-RAD")):
    c = F[ds]["_comparisons"]
    for s in (1, 2, 3): rows.append(("Kev init − base init", f"{nm}, seed {s}", pv(c[f"kev_vs_base_s{s}"], f"test_{ds}_kev_vs_base_s{s}"), C["ft_kev"]))
for ds, nm in (("derm7pt", "Derm7pt"), ("slake", "SLAKE"), ("vqarad", "VQA-RAD")):
    c = F[ds]["_comparisons"]
    pre = "" if ds == "derm7pt" else "zs_"
    for a, b, lab, col in ((f"{'zs_' if ds=='derm7pt' else 'zs_'}reflex4b", f"{pre}lp4b", "reflex-4B − label-prob 4B", C["zs_reflex4b"]),
                           (f"zs_kev4b", f"{pre}lp4b", "Kev-4B − label-prob 4B", C["zs_kev4b"]),
                           (f"zs_kev08b", f"{pre}lp08b", "Kev-0.8B − label-prob 0.8B", C["zs_kev08b"])):
        k = f"{a[3:] if ds=='derm7pt' else a}_vs_{b}"
        pkey = f"test_{ds}_" + {"reflex-4B − label-prob 4B": "reflex4b_vs_lp4b", "Kev-4B − label-prob 4B": "kev4b_vs_lp4b", "Kev-0.8B − label-prob 0.8B": "kev08b_vs_lp08b"}[lab]
        rows.append(("Typed readout − label-prob readout (zero-shot)", f"{nm}: {lab}", pv(c[k], pkey), col))
fig, ax = plt.subplots(figsize=(6.2, 5.2))
y = 0; yt, yl = [], []; last = None
for grp, lab, (m, lo, hi), col in rows:
    if grp != last:
        if last is not None: y += 0.8
        ax.text(0.01, y, grp, fontweight="bold", fontsize=7.5, va="center", transform=ax.get_yaxis_transform(), bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5); y += 1; last = grp
    sig = lo > 0 or hi < 0
    ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round")
    ax.plot(m, y, "o" if sig else "o", ms=5, mfc=col if sig else "white", mec=col, mew=1.2, zorder=3)
    yt.append(y); yl.append(lab); y += 1
ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(yt); ax.set_yticklabels(yl); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("ΔAUC on official test set (95% bootstrap CI; SLAKE/VQA-RAD image-clustered)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, label="CI excludes 0"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, label="CI includes 0")], loc="lower right")
fig.tight_layout(); save(fig, "fig3_forest")

# ---------- Fig 4: Kev init vs base init across training-set sizes (Derm7pt valid) + test ----------
LD = J("lowdata/lowdata_decision.json"); AC = J("g0/a_compare.json")
fr = [0.1, 0.25, 0.5]; kev = {f: [r["kev"] for r in LD["fractions"][str(f)]["rows"]] for f in fr}; base = {f: [r["base"] for r in LD["fractions"][str(f)]["rows"]] for f in fr}
kev[1.0] = [r["TCB"] for r in AC["A1_no_kev_init"]["rows"]]; base[1.0] = [r["A1_no_kev_init"] for r in AC["A1_no_kev_init"]["rows"]]
zs08 = J("g0/tcb_s1/g0_log.json")["log"][0]["valid"]["mean_auc"]; zs4 = J("zs/kev4b/report.json")["mean_auc"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axs[0]; xs = [0.1, 0.25, 0.5, 1.0]; ns = [41, 103, 206, 413]
for arm, dd, off in (("ft_kev", kev, -0.012), ("ft_base", base, 0.012)):
    mu = [np.mean(dd[x]) for x in xs]; sd = [np.std(dd[x], ddof=1) for x in xs]
    ax.errorbar(np.log2(xs) + off, mu, yerr=sd, color=C[arm], lw=1.6, marker="o", ms=4.5, capsize=2, label=LBL[arm])
    for x in xs: ax.scatter([np.log2(x) + off] * 3, dd[x], s=6, color=C[arm], alpha=0.45, zorder=2, lw=0)
ax.axhline(zs08, color=C["zs_kev08b"], ls="--", lw=1); ax.text(np.log2(0.5), zs08 + 0.003, "Kev-0.8B zero-shot", color=INK2, fontsize=6.5, ha="center")
ax.axhline(zs4, color=C["zs_kev4b"], ls="--", lw=1); ax.text(np.log2(0.5), zs4 - 0.008, "Kev-4B zero-shot", color=INK2, fontsize=6.5, ha="center")
ax.set_xticks(np.log2(xs)); ax.set_xticklabels([f"{int(f*100)}%\n(n={n})" for f, n in zip(xs, ns)])
ax.set_xlabel("Derm7pt training images used"); ax.set_ylabel("Mean concept AUC (validation)"); ax.set_title("a  Training-set size (3 seeds, mean ± SD)", loc="left")
ax.legend(loc="upper left")
ax = axs[1]
val = [pv(r["dAUC"], f"val_derm7pt_kev_vs_base_s{r['seed']}") for r in AC["A1_no_kev_init"]["rows"]]; tst = [pv(F["derm7pt"]["_comparisons"][f"kev_vs_base_s{s}"], f"test_derm7pt_kev_vs_base_s{s}") for s in (1, 2, 3)]
for i, (v, t) in enumerate(zip(val, tst)):
    for yy, (m, lo, hi), mk in ((i - 0.15, v, "s"), (i + 0.15, t, "o")):
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [yy, yy], color=C["ft_kev"], lw=1.3); ax.plot(m, yy, mk, ms=5, mfc=C["ft_kev"] if sig else "white", mec=C["ft_kev"], mew=1.1)
ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["seed 1", "seed 2", "seed 3"]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("ΔAUC, Kev init − base init (full data)"); ax.set_title("b  Validation (n=203) vs test (n=395)", loc="left")
ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc=C["ft_kev"], mec=C["ft_kev"], label="validation"),
                   Line2D([], [], marker="o", ls="", mfc=C["ft_kev"], mec=C["ft_kev"], label="test"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec=C["ft_kev"], label="CI includes 0")], loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=6.5)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
fig.tight_layout(); save(fig, "fig4_kev_init")

# ---------- Fig 5: reliability diagrams (needs per-item predictions, not in the public release) ----------
if not os.path.exists(f"{R}/pubmedqa/jev/preds.npz"):
    print("fig5_reliability: per-item predictions not found; keeping the shipped PDF")
else:
    def rel(ax, y, P, col, lab, bins=10):
        conf = P.max(1); acc = (P.argmax(1) == y).astype(float); e = np.linspace(0, 1, bins + 1); xs, ys, ns = [], [], []
        for lo, hi in zip(e[:-1], e[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5: xs.append(conf[m].mean()); ys.append(acc[m].mean()); ns.append(m.sum())
        E = sum(n / len(y) * abs(a - c) for c, a, n in zip(xs, ys, ns))
        ax.plot(xs, ys, "-o", color=col, ms=3.5, lw=1.4, label=lab)
        return E
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.2))
    ax = axs[0]
    for a in ("jev", "kev4b", "reflex4b", "lp4b"):
        z = np.load(f"{R}/pubmedqa/{a}/preds.npz"); ck = {"jev": "jev", "kev4b": "zs_kev4b", "reflex4b": "zs_reflex4b", "lp4b": "lp4b"}[a]
        rel(ax, z["y"], z["P"], C[ck], {"jev": "Jev", "kev4b": "Kev-4B", "reflex4b": "reflex-4B", "lp4b": "Label-prob 4B"}[a] + f" (ECE {F['pubmedqa'][a]['ece']:.3f})")
    ax.set_title("a  PubMedQA (text, n=500)", loc="left")
    for ax, ds, ttl in ((axs[1], "slake", "b  SLAKE yes/no"), (axs[2], "vqarad", "c  VQA-RAD yes/no")):
        for arm, path in (("zs_kev4b", f"test/{ds}_zs_kev4b/preds.npz"), ("lp4b", f"test/{ds}_zs_lp4b/preds.npz"), ("ft_kev", f"test/{ds}_ft_kev_s1/test_preds.npz"), ("ft_base", f"test/{ds}_ft_base_s1/test_preds.npz")):
            z = np.load(f"{R}/{path}", allow_pickle=True); p = z["p_yes"]; P = np.stack([1 - p, p], 1)
            k = {"zs_kev4b": "zs_kev4b", "lp4b": "zs_lp4b", "ft_kev": "ft_kev_s1", "ft_base": "ft_base_s1"}[arm]
            rel(ax, z["y"], P, C[arm], {"zs_kev4b": "Kev-4B zs", "lp4b": "Label-prob 4B zs", "ft_kev": "FT Kev init s1", "ft_base": "FT base init s1"}[arm] + f" ({F[ds][k]['ece']:.3f})")
        ax.set_title(ttl + " (test)", loc="left")
    for ax in axs:
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":"); ax.set_xlim(0.3, 1.0); ax.set_ylim(0.0, 1.02); ax.set_aspect("auto")
        ax.set_xlabel("Confidence (top-1 probability)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6, handlelength=1.4, ncol=1)
    axs[0].set_ylabel("Accuracy")
    fig.tight_layout(); save(fig, "fig5_reliability")


# ---------- Fig 6: option-name / order robustness (Derm7pt valid, 5 Choice criteria) ----------
conds = [("C1_neutral", "neutral names\n(A/B/C)"), ("C2_random", "random\nnames"), ("C4_defonly", "definition\nonly"), ("C5_reversed", "reversed\norder"), ("C3_swap", "names\ndisplaced")]
ZSM = [("optname/kev08b_zs", "Kev-0.8B zs", "zs_kev08b"), ("optname/kev4b_zs", "Kev-4B zs", "zs_kev4b"), ("optname/reflex4b_zs", "reflex-4B zs", "zs_reflex4b")]
FTM = [("ft_kev", "FT Kev init (3 seeds)", "ft_kev"), ("ft_base", "FT base init (3 seeds)", "ft_base")]
rep = lambda d: J(f"{d}/option_name_report.json")["conditions"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axs[0]; w = 0.16; x = np.arange(len(conds))
for i, (d, lab, ck) in enumerate(ZSM):
    r = rep(d); ax.bar(x + (i - 2) * w, [r[c]["flips_per_100_vs_C0"] for c, _ in conds], width=w * 0.88, color=C[ck], label=lab)
for j, (a, lab, ck) in enumerate(FTM):
    vals = np.array([[rep(f"optname3/{a}_s{s}")[c]["flips_per_100_vs_C0"] for c, _ in conds] for s in (1, 2, 3)])
    ax.bar(x + (j + 1) * w, vals.mean(0), width=w * 0.88, color=C[ck], label=lab, yerr=vals.std(0, ddof=1), error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK2))
ax.set_xticks(x); ax.set_xticklabels([l for _, l in conds]); ax.set_ylabel("Decisions changed per 100\n(vs original options)")
ax.set_title("a  Answer instability under option rewrites", loc="left"); ax.grid(axis="x", visible=False); ax.legend(ncol=2, fontsize=6.1, loc="upper left")
ax = axs[1]; rows = []
for d, lab, ck in ZSM:
    cc = rep(d); ci = cc["C3_swap"]["dAUC_vs_C0"]; rows.append((lab, (cc["C3_swap"]["mean_auc"] - cc["C0_orig"]["mean_auc"], ci[1], ci[2]), ck))
O3 = J("optname3/optname3_decision.json")["seeds"]
for s in ("1", "2", "3"): rows.append((f"FT Kev init, seed {s}", pv(O3[s]["kev_swap_dAUC"], f"optname_kev_swap_s{s}"), "ft_kev"))
for s in ("1", "2", "3"): rows.append((f"FT base init, seed {s}", pv(O3[s]["base_swap_dAUC"], f"optname_base_swap_s{s}"), "ft_base"))
for i, (lab, (m, lo, hi), ck) in enumerate(rows):
    sig = hi < 0 or lo > 0
    ax.plot([lo, hi], [i, i], color=C[ck], lw=1.4); ax.plot(m, i, "o", ms=5, mfc=C[ck] if sig else "white", mec=C[ck], mew=1.2)
ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("ΔAUC, names displaced − original"); ax.set_title("b  Following the option name", loc="left")
fig.tight_layout(); save(fig, "fig6_option_names")

# ---------- Fig 7: Derm7pt per-criterion AUC heatmap (test) ----------
crit = ["pigment_network", "blue_whitish_veil", "vascular_structures", "pigmentation", "streaks", "dots_and_globules", "regression_structures"]
cl = ["Pigment\nnetwork", "Blue-whitish\nveil", "Vascular\nstructures", "Pigmen-\ntation", "Streaks", "Dots &\nglobules", "Regression\nstructures"]
arms = [("ft_kev", "Fine-tuned, Kev init"), ("ft_base", "Fine-tuned, base init"), ("probe_qwenvis", "Qwen vision probe"), ("probe_biomedclip", "BiomedCLIP probe"),
        ("zs_reflex4b", "reflex-4B zs"), ("zs_kev4b", "Kev-4B zs"), ("lp4b", "Label-prob 4B zs"), ("zs_kev08b", "Kev-0.8B zs"), ("lp08b", "Label-prob 0.8B zs")]
A = np.array([[np.mean([F["derm7pt"][k]["per_crit_auc"][c] for k in F["derm7pt"] if not k.startswith("_") and (k == a or k.startswith(a + "_s"))]) for c in crit] for a, _ in arms])
fig, ax = plt.subplots(figsize=(6.4, 3.0))
im = ax.imshow(A, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
for i in range(A.shape[0]):
    for j in range(A.shape[1]): ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if A[i, j] > 0.82 else "#0b0b0b")
ax.set_xticks(range(len(crit))); ax.set_xticklabels(cl, fontsize=6.5); ax.set_yticks(range(len(arms))); ax.set_yticklabels([l for _, l in arms]); ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("AUC (test, mean over seeds)", fontsize=7); cb.outline.set_visible(False)
fig.tight_layout(); save(fig, "fig7_derm_heatmap")

# ---------- Fig 8: leakage ----------
L = J("leakage_stats.json")
fig, ax = plt.subplots(figsize=(3.4, 2.0))
bars = [("VQA-RAD test", L["vqarad_test"]), ("SLAKE val", L["slake_val"]), ("SLAKE test", L["slake_test"])]
for i, (nm, d) in enumerate(bars):
    for j, (k, col, lab) in enumerate((("seen_yesno_train", C["ft_kev"], "image in yes/no train"), ("seen_anytype_train", C["probe_biomedclip"], "image in any-question train"))):
        v = 100 * d[k] / d["n_questions"]; ax.barh(i + (j - 0.5) * 0.36, v, height=0.32, color=col, label=lab if i == 0 else None)
        ax.text(v + 1.5, i + (j - 0.5) * 0.36, f"{d[k]}/{d['n_questions']}", va="center", fontsize=6.3, color=INK2)
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars]); ax.invert_yaxis(); ax.set_xlim(0, 118); ax.grid(axis="y", visible=False)
ax.set_xlabel("% of yes/no questions whose image\nappears in the official training split"); ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, fontsize=6)
fig.tight_layout(); save(fig, "fig8_leakage")

# ---------- Fig 9: PubMedQA quality vs latency ----------
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for a, ck, nm in (("jev", "jev", "Jev (hosted API)"), ("reflex4b", "zs_reflex4b", "reflex-4B"), ("kev4b", "zs_kev4b", "Kev-4B"), ("lp4b", "lp4b", "Label-prob 4B"), ("kev08b", "zs_kev08b", "Kev-0.8B")):
    r = F["pubmedqa"][a]; ax.scatter(r["latency_median_s"] * 1000, r["macro_f1"], s=36, color=C[ck], edgecolor="white", lw=0.6, zorder=3)
    ax.annotate(nm, (r["latency_median_s"] * 1000, r["macro_f1"]), xytext=(-6, 4) if a == "jev" else (6, 2), ha="right" if a == "jev" else "left",
                textcoords="offset points", fontsize=6.5, color="#0b0b0b")
ax.set_xscale("log"); ax.set_xlim(40, 1600); ax.set_xlabel("Median latency per request (ms, log scale)"); ax.set_ylabel("Macro-F1 (PubMedQA test)")
fig.tight_layout(); save(fig, "fig9_pubmedqa_latency")
print("FIGS_DONE", sorted(os.listdir(O)))
