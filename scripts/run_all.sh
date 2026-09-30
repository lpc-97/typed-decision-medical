#!/usr/bin/env bash
# Full pipeline in the order used for the paper. Set the environment variables described in README.md first.
# Each command writes to $TDM_RUNS; GPU assignment is left to the user (CUDA_VISIBLE_DEVICES).
set -euo pipefail
R=${TDM_RUNS:-runs}

# 0. adapter checks
python scripts/checks/text_parity.py
python scripts/checks/adapter_checks.py

# 1. Derm7pt fine-tuning (Kev init and base init, 3 seeds, 6 epochs, last epoch evaluated)
for s in 1 2 3; do
  python scripts/train/train_derm7pt.py --seed $s --init kev  --out $R/g0/tcb_s$s
  python scripts/train/train_derm7pt.py --seed $s --init base --out $R/g0/a1_base_s$s
done
# low-data study (validation only; 300 optimiser steps)
for f in 0.1 0.25 0.5; do for s in 1 2 3; do for init in kev base; do
  python scripts/train/train_derm7pt.py --frac $f --seed $s --init $init --steps 300 --out $R/lowdata/f${f}_${init}_s$s
done; done; done

# 2. SLAKE / VQA-RAD fine-tuning on official splits
for ds in slake vqarad; do for s in 1 2 3; do for init in kev base; do
  python scripts/train/train_vqa.py --ds $ds --init $init --seed $s --split official --out $R/vqa_ft_official/${ds}_${init}_s$s
done; done; done

# 3. validation decisions
python scripts/analysis/decide_derm7pt_g0.py
python scripts/analysis/compare_ablations.py
python scripts/analysis/decide_lowdata.py

# 4. option-name test (validation) and its prespecified comparison
for s in 1 2 3; do
  python scripts/eval/option_name_test.py ft:kev:$R/g0/tcb_s$s/final_trainable.pt $R/optname3/ft_kev_s$s
  python scripts/eval/option_name_test.py ft:base:$R/g0/a1_base_s$s/final_trainable.pt $R/optname3/ft_base_s$s
done
python scripts/eval/option_name_test.py jaredpalmer/kev-0.8b $R/optname/kev08b_zs
python scripts/eval/option_name_test.py jaredpalmer/kev-4b   $R/optname/kev4b_zs
python scripts/eval/option_name_test.py reflex               $R/optname/reflex4b_zs
python scripts/analysis/decide_option_names.py

# 5. one-time test-set pass
T=$R/test
for s in 1 2 3; do
  python scripts/eval/eval_finetuned_test.py derm7pt kev  $R/g0/tcb_s$s/final_trainable.pt $T/derm_ft_kev_s$s
  python scripts/eval/eval_finetuned_test.py derm7pt base $R/g0/a1_base_s$s/final_trainable.pt $T/derm_ft_base_s$s
  for ds in slake vqarad; do for init in kev base; do
    python scripts/eval/eval_finetuned_test.py $ds $init $R/vqa_ft_official/${ds}_${init}_s$s/final_trainable.pt $T/${ds}_ft_${init}_s$s
  done; done
done
python scripts/eval/zeroshot_derm7pt.py jaredpalmer/kev-0.8b $T/derm_zs_kev08b test
python scripts/eval/zeroshot_derm7pt.py jaredpalmer/kev-4b   $T/derm_zs_kev4b  test
python scripts/eval/zeroshot_derm7pt.py reflex               $T/derm_zs_reflex4b test
python scripts/eval/labelreader_derm7pt.py Qwen/Qwen3.5-4B   test $T/derm_lp4b
python scripts/eval/labelreader_derm7pt.py Qwen/Qwen3.5-0.8B test $T/derm_lp08b
python scripts/eval/probe_qwen_vision.py $T/derm_probe_qwenvis test
python scripts/eval/probe_biomedclip.py  $T/derm_probe_biomedclip test
for ds in slake vqarad; do
  python scripts/eval/zeroshot_vqa.py kev:jaredpalmer/kev-0.8b   $ds official:test $T/${ds}_zs_kev08b
  python scripts/eval/zeroshot_vqa.py kev:jaredpalmer/kev-4b     $ds official:test $T/${ds}_zs_kev4b
  python scripts/eval/zeroshot_vqa.py reflex                     $ds official:test $T/${ds}_zs_reflex4b
  python scripts/eval/zeroshot_vqa.py labelprob:Qwen/Qwen3.5-4B   $ds official:test $T/${ds}_zs_lp4b
  python scripts/eval/zeroshot_vqa.py labelprob:Qwen/Qwen3.5-0.8B $ds official:test $T/${ds}_zs_lp08b
done
python scripts/eval/eval_pubmedqa.py jev                        $R/pubmedqa/jev      # needs TYPESAFE_API_KEY
python scripts/eval/eval_pubmedqa.py kev:jaredpalmer/kev-4b      $R/pubmedqa/kev4b
python scripts/eval/eval_pubmedqa.py kev:jaredpalmer/kev-0.8b    $R/pubmedqa/kev08b
python scripts/eval/eval_pubmedqa.py reflex                      $R/pubmedqa/reflex4b
python scripts/eval/eval_pubmedqa.py labelprob:Qwen/Qwen3.5-4B   $R/pubmedqa/lp4b
python scripts/eval/leakage_stats.py
python scripts/analysis/final_test_analysis.py

# 6. derived numbers and figures (read the aggregated results/ directory)
python paper/derive_numbers.py
python paper/make_figs.py
