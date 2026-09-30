"""Derm7pt 7-point-checklist criteria as Kev typed questions.

Label groupings and 7-point scores are the official reduced ones from jeremykawahara/derm7pt@ce43687
(derm7pt/dataset.py, Kawahara et al., IEEE JBHI 2019) — fixed BEFORE any training (pre-registered here).
Splits: the official train/valid/test index files shipped with release_v0 (413/203/395).
"""
import pandas as pd

import os
from tdm.paths import DERM7PT_IMAGES, DERM7PT_META
META = DERM7PT_META
IMG = os.path.join(DERM7PT_IMAGES, "{:04d}.png")
STATE = "Dermoscopic image of a skin lesion."

# criterion -> (qtype, instruction, [(option name, option text, raw labels grouped into it, 7pt score)])
CRITERIA = {
    "pigment_network": ("choice", "What is the pigment network of this lesion?", [
        ("absent", "absent: no pigment network visible", ["absent"], 0),
        ("typical", "typical: regular, uniformly spaced network lines", ["typical"], 0),
        ("atypical", "atypical: irregular network with thickened lines or uneven holes", ["atypical"], 2)]),
    "blue_whitish_veil": ("noul", "Is a blue-whitish veil present in this lesion?", [
        ("absent", "no", ["absent"], 0),
        ("present", "yes", ["present"], 2)]),
    "vascular_structures": ("choice", "What vascular structures does this lesion show?", [
        ("absent", "absent: no vessels visible", ["absent"], 0),
        ("regular", "regular: arborizing, comma, hairpin, wreath or within-regression vessels",
         ["arborizing", "comma", "hairpin", "within regression", "wreath"], 0),
        ("irregular", "irregular: dotted or linear irregular vessels", ["dotted", "linear irregular"], 2)]),
    "pigmentation": ("choice", "What is the pigmentation (blotches) of this lesion?", [
        ("absent", "absent: no blotches", ["absent"], 0),
        ("regular", "regular: diffuse or localized regular pigmentation", ["diffuse regular", "localized regular"], 0),
        ("irregular", "irregular: diffuse or localized irregular pigmentation", ["diffuse irregular", "localized irregular"], 1)]),
    "streaks": ("choice", "What streaks does this lesion show?", [
        ("absent", "absent: no streaks", ["absent"], 0),
        ("regular", "regular: streaks symmetrically distributed around the lesion", ["regular"], 0),
        ("irregular", "irregular: streaks irregularly distributed", ["irregular"], 1)]),
    "dots_and_globules": ("choice", "What dots and globules does this lesion show?", [
        ("absent", "absent: no dots or globules", ["absent"], 0),
        ("regular", "regular: dots/globules of similar size and even distribution", ["regular"], 0),
        ("irregular", "irregular: dots/globules of varying size, irregularly distributed", ["irregular"], 1)]),
    "regression_structures": ("noul", "Are regression structures (white scar-like or blue-grey peppered areas) present?", [
        ("absent", "no", ["absent"], 0),
        ("present", "yes", ["blue areas", "white areas", "combinations"], 1)]),
}
NAMES = list(CRITERIA)


def label_index(crit, raw):
    for j, (_, _, raws, _) in enumerate(CRITERIA[crit][2]):
        if raw in raws:
            return j
    raise ValueError(f"{crit}: unmapped raw label {raw!r}")


def scores(crit):
    return [s for *_, s in CRITERIA[crit][2]]


def load_split(split):
    """-> DataFrame with case_num, img path, mel (0/1), diagnosis and one integer label column per criterion."""
    m = pd.read_csv(f"{META}/meta.csv").reset_index(drop=True)
    idx = pd.read_csv(f"{META}/{split}_indexes.csv")["indexes"].tolist()
    d = m.iloc[idx].copy()
    d["img"] = [IMG.format(int(c)) for c in d["case_num"]]
    d["mel"] = d["diagnosis"].str.strip().str.lower().str.startswith("melanoma").astype(int)
    for c in NAMES:
        d[c + "_y"] = [label_index(c, str(v).strip()) for v in d[c]]
    return d.reset_index(drop=True)


def record(labels=None, option_order=None):
    """Internal Kev record (kev.model.encode input). option_order[c] = permutation (new -> old) for augmentation."""
    qs = []
    for c in NAMES:
        qtype, instr, opts = CRITERIA[c]
        order = (option_order or {}).get(c) or list(range(len(opts)))
        y = 0 if labels is None else order.index(labels[c])
        qs.append({"instr": instr, "options": [opts[j][1] for j in order], "label": y, "qtype": qtype, "crit": c})
    return {"state": STATE, "questions": qs}
