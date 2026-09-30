"""All data/run locations, configurable by environment variables (see README)."""
import os

RUNS = os.environ.get("TDM_RUNS", "runs")                      # experiment outputs
CACHE = os.environ.get("TDM_CACHE", "cache")                   # decoded image cache (reflex needs files)
DERM7PT_META = os.environ.get("DERM7PT_META", "data/derm7pt/meta")        # meta.csv, {train,valid,test}_indexes.csv (release_v0)
DERM7PT_IMAGES = os.environ.get("DERM7PT_IMAGES", "data/derm7pt/images")  # dermoscopic images named {case_num:04d}.png
VQA_RAW = os.environ.get("VQA_RAW_DIR", "data/raw")            # vqa_rad/data/*.parquet ; slake/{train,validation,test}.json + slake/imgs/
PUBMEDQA_DIR = os.environ.get("PUBMEDQA_DIR", "data/pubmedqa") # ori_pqal.json, test_ground_truth.json (pubmedqa@1cbae8e)
KEV_REPO = os.environ.get("KEV_REPO", "third_party/kev")       # clone of jaredpalmer/kev@5920c5f (for its eval suite)
