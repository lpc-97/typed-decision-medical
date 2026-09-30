"""VQA-RAD and SLAKE (English) CLOSED yes/no questions as Kev Noul questions.

Leakage (byte-identical image files, MD5; see leakage_stats.py): official splits are question-level, not image-level.
  vs yes/no-English train : VQA-RAD test 227/251; SLAKE val 0/358, test 0/355.
  vs ANY train question   : VQA-RAD test 250/251; SLAKE val 248/358, test 264/355.
PRIMARY protocol (user decision 2026-09-29): OFFICIAL splits for all arms. The image-disjoint `resplit` column
(all yes/no questions pooled, grouped by image MD5, 60/20/20 with seed 0) is kept for supplementary use only.
Label: 1 = yes. Question = the dataset question verbatim; options follow kev.api.to_record for noul (["no", "yes"]).
"""
import hashlib, io, json
import numpy as np, pandas as pd
from PIL import Image

import os
from tdm.paths import VQA_RAW
RAW = VQA_RAW
STATE = "Medical image."


def _vqarad():
    rows = []
    for split, f in (("train", "train-00000-of-00001-eb8844602202be60.parquet"), ("test", "test-00000-of-00001-e5bc3d208bb4deeb.parquet")):
        d = pd.read_parquet(f"{RAW}/vqa_rad/data/{f}")
        for i, r in d.iterrows():
            a = str(r["answer"]).strip().lower()
            if a in ("yes", "no"):
                b = r["image"]["bytes"]
                rows.append({"ds": "vqarad", "official": split, "qid": f"{split}{i}", "question": r["question"].strip(),
                             "y": int(a == "yes"), "img_md5": hashlib.md5(b).hexdigest(), "img_bytes": b, "img_path": None})
    return rows


def _slake():
    rows = []
    for split in ("train", "validation", "test"):
        for x in json.load(open(f"{RAW}/slake/{split}.json")):
            a = str(x["answer"]).strip().lower()
            if x["q_lang"] == "en" and x["answer_type"] == "CLOSED" and a in ("yes", "no"):
                p = f"{RAW}/slake/imgs/{x['img_name']}"
                rows.append({"ds": "slake", "official": {"validation": "val"}.get(split, split), "qid": str(x["qid"]),
                             "question": x["question"].strip(), "y": int(a == "yes"),
                             "img_md5": hashlib.md5(open(p, "rb").read()).hexdigest(), "img_bytes": None, "img_path": p,
                             "modality": x["modality"], "content_type": x["content_type"]})
    return rows


def load(ds):
    """-> DataFrame with official split and image-disjoint `resplit` (train/val/test)."""
    df = pd.DataFrame(_vqarad() if ds == "vqarad" else _slake())
    imgs = sorted(df["img_md5"].unique())
    rng = np.random.default_rng(0); rng.shuffle(imgs)
    n = len(imgs); cut1, cut2 = int(0.6 * n), int(0.8 * n)
    assign = {h: ("train" if i < cut1 else "val" if i < cut2 else "test") for i, h in enumerate(imgs)}
    df["resplit"] = df["img_md5"].map(assign)
    return df


def image(row):
    return Image.open(io.BytesIO(row["img_bytes"]) if row["img_bytes"] is not None else row["img_path"]).convert("RGB")


def record(question, y=0, reverse=False):
    opts = ["no", "yes"]
    if reverse:
        return {"state": STATE, "questions": [{"instr": question, "options": opts[::-1], "label": 1 - y, "qtype": "noul"}]}
    return {"state": STATE, "questions": [{"instr": question, "options": opts, "label": y, "qtype": "noul"}]}


if __name__ == "__main__":
    for ds in ("vqarad", "slake"):
        d = load(ds)
        print(ds, "questions", len(d), "images", d["img_md5"].nunique())
        print("  official:", d.groupby("official").agg(n=("y", "size"), yes=("y", "mean"), imgs=("img_md5", "nunique")).round(3).to_dict("index"))
        print("  resplit :", d.groupby("resplit").agg(n=("y", "size"), yes=("y", "mean"), imgs=("img_md5", "nunique")).round(3).to_dict("index"))
        S = {s: set(d[d.resplit == s].img_md5) for s in ("train", "val", "test")}
        print("  resplit image overlap tr-val", len(S["train"] & S["val"]), "tr-te", len(S["train"] & S["test"]), "val-te", len(S["val"] & S["test"]))
        # exact duplicate question+image pairs across resplit (would be leakage too)
        k = d["img_md5"] + "|" + d["question"].str.lower()
        print("  duplicate (image,question) pairs:", int(k.duplicated().sum()))
