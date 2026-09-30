"""Image leakage of official VQA-RAD / SLAKE splits for the yes/no (English) questions -> runs/leakage_stats.json.
seen_yesno_train: image appears among yes/no-English TRAIN questions (what our models train on);
seen_anytype_train: image appears among ANY train question (all types / languages; the usual full-dataset training)."""
import hashlib, json, sys
import pandas as pd
from tdm.vqa import RAW, load
import os
from tdm.paths import RUNS
anyv = {"vqarad": {hashlib.md5(b["bytes"]).hexdigest() for b in pd.read_parquet(f"{RAW}/vqa_rad/data/train-00000-of-00001-eb8844602202be60.parquet")["image"]},
        "slake": {hashlib.md5(open(f"{RAW}/slake/imgs/" + x["img_name"], "rb").read()).hexdigest() for x in json.load(open(f"{RAW}/slake/train.json"))}}
out = {}
for ds in ("vqarad", "slake"):
    d = load(ds); tr = set(d[d.official == "train"].img_md5)
    for s in ("val", "test"):
        e = d[d.official == s]
        if len(e): out[f"{ds}_{s}"] = {"n_questions": int(len(e)), "n_images": int(e.img_md5.nunique()),
                                       "seen_yesno_train": int(e.img_md5.isin(tr).sum()), "seen_anytype_train": int(e.img_md5.isin(anyv[ds]).sum())}
    out[f"{ds}_train_yesno"] = {"n_questions": int((d.official == "train").sum()), "n_images": len(tr)}
json.dump(out, open(os.path.join(RUNS, "leakage_stats.json"), "w"), indent=2); print(json.dumps(out, indent=1))
