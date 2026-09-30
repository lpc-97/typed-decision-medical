"""VQA-RAD / SLAKE analysis with IMAGE-CLUSTERED bootstrap (questions sharing an image are resampled together).
Pre-registered comparisons (2026-09-29):
  B1 typed vs label-prob, same backbone : reflex(Qwen3.5-4B) vs labelprob:Qwen3.5-4B ; kev-0.8b vs labelprob:Qwen3.5-0.8B
  Kev training effect (zero-shot)       : kev-4b vs labelprob:Qwen3.5-4B  (Kev-4B = Qwen3.5-4B-Base + Kev LoRA; different base
                                          from the instruct model -> reported, not a clean ablation)
  Kev init under fine-tuning            : kev vs base init, 3 seeds, paired; 'helps' iff all 3 seeds' CI > 0
Metric for tests: AUC (threshold-free); acc/ECE/Brier reported."""
import glob, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

from tdm.metrics import ece

from tdm.paths import RUNS
ZS, FT = os.path.join(RUNS, "vqa_zs"), os.path.join(RUNS, "vqa_ft")


def load(path):
    z = np.load(path, allow_pickle=True); return z["p_yes"], z["y"], z["img"]


def cluster_boot(y, img, pa, pb=None, B=2000, seed=0):
    rng = np.random.default_rng(seed); ids = np.unique(img); groups = {i: np.where(img == i)[0] for i in ids}; d = []
    while len(d) < B:
        idx = np.concatenate([groups[i] for i in rng.choice(ids, len(ids))])
        if len(np.unique(y[idx])) < 2: continue
        a = roc_auc_score(y[idx], pa[idx]); d.append(a - (roc_auc_score(y[idx], pb[idx]) if pb is not None else 0))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def summ(p, y):
    return {"acc": float(((p > .5) == y).mean()), "auc": float(roc_auc_score(y, p)), "ece": ece(y, np.stack([1 - p, p], 1)), "brier": float(((p - y) ** 2).mean())}


if __name__ == "__main__":
    out = {}
    for ds in ("vqarad", "slake"):
        print(f"\n######## {ds} — zero-shot on image-disjoint VAL")
        arms = {}
        for f in sorted(glob.glob(f"{ZS}/*_{ds}_val/preds.npz")):
            name = os.path.basename(os.path.dirname(f)).replace(f"_{ds}_val", ""); arms[name] = load(f)
        for n, (p, y, img) in arms.items():
            s = summ(p, y); _, lo, hi = cluster_boot(y, img, p)
            print(f"  {n:10s} acc {s['acc']:.3f}  AUC {s['auc']:.3f} [{lo:.3f},{hi:.3f}]  ECE {s['ece']:.3f}  Brier {s['brier']:.3f}")
            out.setdefault(ds, {}).setdefault("zs", {})[n] = {**s, "auc_ci": [lo, hi]}
        for a_, b_, what in (("reflex4b", "lp4b", "B1 typed(reflex) vs label-prob, Qwen3.5-4B"), ("kev08b", "lp08b", "B1 Kev-0.8B vs label-prob Qwen3.5-0.8B"),
                             ("kev4b", "lp4b", "Kev-4B vs label-prob Qwen3.5-4B (different base)")):
            if a_ in arms and b_ in arms:
                pa, y, img = arms[a_]; pb, yb, _ = arms[b_]; assert (y == yb).all()
                d, lo, hi = cluster_boot(y, img, pa, pb, seed=1)
                print(f"  dAUC {what}: {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
                out[ds].setdefault("zs_diff", {})[f"{a_}-{b_}"] = [d, lo, hi]
        print(f"######## {ds} — fine-tuned (image-disjoint train -> val), last epoch")
        rows = []
        for s in (1, 2, 3):
            fk, fb = f"{FT}/{ds}_resplit_kev_s{s}/eval_preds.npz", f"{FT}/{ds}_resplit_base_s{s}/eval_preds.npz"
            if not (os.path.exists(fk) and os.path.exists(fb)): continue
            pk, y, img = load(fk); pb, _, _ = load(fb)
            d, lo, hi = cluster_boot(y, img, pk, pb, seed=10 + s)
            rows.append({"seed": s, "kev": summ(pk, y), "base": summ(pb, y), "dAUC": [d, lo, hi]})
            print(f"  seed {s}: kev AUC {rows[-1]['kev']['auc']:.3f} acc {rows[-1]['kev']['acc']:.3f} | base AUC {rows[-1]['base']['auc']:.3f} acc {rows[-1]['base']['acc']:.3f} | dAUC {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
        if rows:
            helps = len(rows) == 3 and all(r["dAUC"][1] > 0 for r in rows)
            print(f"  -> Kev init {'HELPS (all 3 CI>0)' if helps else 'no significant help'}")
            out[ds]["ft"] = {"rows": rows, "kev_helps": helps}
        if ds == "slake":
            for init in ("kev", "base"):
                f = f"{FT}/slake_official_{init}_s1/report.json"
                if os.path.exists(f):
                    r = json.load(open(f)); print(f"  official split ({init}): seen-image acc {r['seen_img']['acc']:.3f} (n={r['seen_img']['n']}) vs unseen-image acc {r['unseen_img']['acc']:.3f} (n={r['unseen_img']['n']}); AUC {r['seen_img']['auc']:.3f} vs {r['unseen_img']['auc']:.3f}")
                    out[ds].setdefault("leak", {})[init] = r
    json.dump(out, open(os.path.join(RUNS, "vqa_analysis.json"), "w"), indent=2)
    print("\nVQA_ANALYSIS_DONE")
