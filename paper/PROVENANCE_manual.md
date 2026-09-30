# Manual provenance notes (numbers the automated audit could not match)
Automated audit (every decimal number in the draft matched against the files in results/, commit hashes masked): v3 draft 320 numbers, 317 traced automatically.
The remaining 3 are correct roundings missed by the matcher, which truncates rather than rounds:
- "+0.083" (Table 4, Jev - Kev-4B ECE point estimate) <- results/pubmedqa/jev/report.json ece 0.129620 - results/pubmedqa/kev4b/report.json ece 0.046567 = 0.083053 (also results/derived_numbers.json point.pubmedqa_jev_vs_kev4b_ece)
- "+0.027" (Sec. 4.6, seed-2 DiD lower CI) <- results/optname3/optname3_decision.json seeds.2.DiD[1] = 0.026626
- "+0.034" (Sec. 4.6, seed-3 DiD upper CI) <- results/optname3/optname3_decision.json seeds.3.DiD[2] = 0.034296
Point estimates of differences: results/derived_numbers.json -> "point"; intervals: bootstrap percentiles in FINAL_RESULTS.json / optname3_decision.json / a_compare.json.
External numbers quoted from cited papers: 0.94->0.23 and 0.81->0.58 (Sun et al., arXiv 2609.26758).
