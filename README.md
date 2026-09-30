# Typed Is Not Always Better: Jev-Style Decision Models for Medical Vision–Language Tasks

Code for the paper *Typed Is Not Always Better: Jev-Style Decision Models for Medical Vision–Language Tasks* (preprint, 2026).

The study compares typed decision readouts (Jev-style models: one probability per caller-defined option, no text generation) with label-probability readouts, frozen-feature probes and fine-tuned models on Derm7pt, SLAKE, VQA-RAD and PubMedQA. It contains:

- an adapter that connects the native Qwen3.5 vision encoder to [Kev](https://github.com/jaredpalmer/kev) (`src/tdm/vision_kev.py`), exact to Kev on text inputs;
- training, evaluation and analysis scripts for every table and figure in the paper;
- the aggregated result files from which all reported numbers are computed (`results/`).

## Repository layout

```
src/tdm/            library: vision adapter, task definitions, metrics, bootstrap statistics, paths
scripts/train/      fine-tuning (Derm7pt; SLAKE / VQA-RAD)
scripts/eval/       zero-shot typed and label-probability readouts, probes, PubMedQA (incl. hosted Jev),
                    test-set evaluation of fine-tuned weights, option-name test, split-overlap statistics
scripts/analysis/   prespecified decision scripts and the final test-set analysis
scripts/checks/     adapter checks (text parity with Kev, gradient flow, image dependence)
paper/              LaTeX source, figures, figure and derived-number scripts
results/            aggregated result files (JSON); per-item predictions are not included
```

## Installation

Tested with Python 3.13, CUDA 12.8 and one RTX 3090 (24 GB) per run.

```bash
pip install -r requirements.txt
pip install -e .
# base repositories used by the paper (pinned commits)
git clone https://github.com/jaredpalmer/kev third_party/kev && git -C third_party/kev checkout 5920c5f && pip install -e third_party/kev
git clone https://github.com/kshetrajna12/reflex third_party/reflex && git -C third_party/reflex checkout 231f896 && pip install --no-deps -e third_party/reflex
```

The BiomedCLIP probe uses `open_clip_torch` (3.3.0) and can run in a separate environment.

## Data

Datasets are not redistributed. Obtain them from their original sources and point the environment variables below to them (defaults in `src/tdm/paths.py`):

| Variable | Contents |
|---|---|
| `DERM7PT_META` | Derm7pt `release_v0/meta` (`meta.csv`, `{train,valid,test}_indexes.csv`) |
| `DERM7PT_IMAGES` | dermoscopic images named `{case_num:04d}.png` |
| `VQA_RAW_DIR` | `vqa_rad/data/*.parquet` (HF `flaviagiammarino/vqa-rad`) and `slake/{train,validation,test}.json`, `slake/imgs/` |
| `PUBMEDQA_DIR` | `ori_pqal.json`, `test_ground_truth.json` from pubmedqa@1cbae8e |
| `KEV_REPO` | clone of Kev (used for its decision-v7 development suite) |
| `TDM_RUNS` | output directory for all runs |
| `TDM_CACHE` | cache for decoded images passed to reflex |

Hugging Face checkpoints: `jaredpalmer/kev-0.8b`, `jaredpalmer/kev-4b` (and their Qwen3.5 Base checkpoints), `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B`. The hosted Jev is queried only on PubMedQA and needs `TYPESAFE_API_KEY` in the environment.

## Reproducing the paper

`scripts/run_all.sh` lists every command in order. The mapping from paper items to scripts:

| Paper item | Script(s) |
|---|---|
| Interface checks (Method; Appendix A) | `scripts/checks/text_parity.py`, `scripts/checks/adapter_checks.py` |
| Fine-tuning with Kev or base initialization | `scripts/train/train_derm7pt.py`, `scripts/train/train_vqa.py` |
| Zero-shot typed readouts and label readers | `scripts/eval/zeroshot_derm7pt.py`, `scripts/eval/zeroshot_vqa.py`, `scripts/eval/labelreader_derm7pt.py` |
| Linear probes | `scripts/eval/probe_biomedclip.py`, `scripts/eval/probe_qwen_vision.py` |
| PubMedQA including the hosted Jev (Table 4) | `scripts/eval/eval_pubmedqa.py` |
| Test-set evaluation of fine-tuned weights | `scripts/eval/eval_finetuned_test.py` |
| Main results, readout comparison (Tables 3–4, forest plot) | `scripts/analysis/final_test_analysis.py` |
| Decision-oriented initialization on validation (low-data study) | `scripts/analysis/decide_derm7pt_g0.py`, `compare_ablations.py`, `decide_lowdata.py` |
| Option-name analysis | `scripts/eval/option_name_test.py`, `scripts/analysis/decide_option_names.py` |
| Split overlap (Table 1) | `scripts/eval/leakage_stats.py` |
| Point estimates, derived numbers and figures | `paper/derive_numbers.py`, `paper/make_figs.py` (read `results/`) |
| Writing-style self-check | `paper/check_style.py` |

Differences are reported as full-sample point estimates (`results/derived_numbers.json`, key `point`) with 95% paired bootstrap percentile intervals. Per-item predictions are not included; without them `paper/make_figs.py` keeps the shipped reliability-diagram PDF and `paper/derive_numbers.py` keeps the shipped prediction-based statistics.

Decision rules were fixed before each test set was read; they are stated in the docstrings of the `decide_*` and `final_test_analysis.py` scripts. Bootstrap seeds are fixed, so the analysis scripts reproduce `results/test/FINAL_RESULTS.json` exactly from the prediction files.

## Licence

Apache-2.0 (see `LICENSE` and `NOTICE`). `src/tdm/vision_kev.py` adapts code from Kev (Apache-2.0). Datasets and model weights are subject to their own licences.
