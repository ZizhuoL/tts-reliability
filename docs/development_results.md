# Six-System Development Results: Ordinary Continuation and Two Inference Strategies

Evidence cutoff: 2026-10-20. All six systems completed the same 45 natural development texts × generation seeds17/42, totaling 540 scored attempts. Overall WER was **10.1724% for BNoLM**, versus **25.9052% for A**. BNoLM was the sole candidate passing the preregistered conservative point-estimate filter, and the coordinator recorded it as the development choice. This changes both trained weights and the LM reference prefix; the full gain is not attributable to training alone. This chapter audits development results/eligibility; **it is neither the formal frozen protocol nor an independent final-test conclusion**. Seed 183 remains primary. Seed 184 had started but had no completed results at this cutoff and cannot replace the primary seed based on subsequent performance. Sealed final contents were not read.

The outer development shell exited **127** (session 27155), ending with `command not found: ithin-B`. This chapter did not rerun GPU work or claim pipeline exit 0. Completion evidence consists of four analysis completions/artifact hashes, six raw result sets, and two verified listening indices. The [closing reconciliation](../results/analysis/development-execution-reconciliation/incident.json) separately records seed 184 read-only preflight exit 0. Editing the shell while it ran is consistent with the tail-fragment error, but launch-time bytes are unavailable, so the sole cause cannot be established.

## 1. Design, Identity, and Denominators

A is the course starter. B is seed 183 full-epoch `latest.pt` after continuation on the original distribution: 44,348 steps/50,450,231 supervised targets, not the subset-validation-selected best. See [training accounting](training_results.md). Each weight set uses standard, LM-reference-prefix removal (NoLM), and sentence synthesis/concatenation (Split), giving A, ANoLM, ASplit, B, BNoLM, BSplit. NoLM removes only reference text/speech-token prefixes from the LM; acoustic conditioning remains fixed. This does not prove unchanged speaker similarity.

All cells share reference audio/normalized reference text, temperature 1, top-k 50, minimum speech/text ratio 2, KV cache, and budget `max(500, ceil(11 * target_text_tokens))`; split boundaries add 0.15-second pauses. Short/medium/long groups contain 6–17/18–43/44–72 actual text tokens, 15 texts/30 generations each. Public literary development texts were filtered for exact/local overlap with course training/development data. This does not exclude unknown pretraining overlap or represent every reading scenario.

Unified-normalizer rescoring retains original ASR and prior metrics. All six id/seed sets match: 90 complete/scored per cell, 0 exceptions/missing ASR, and all 540 disk WAV hashes match records. Overall, length, and seed S/D/I/N, WER, and stop counts were independently recomputed. Reference-word N is 2,320 per cell; short/medium/long254/660/1,406; each generation seed 1,160. Denominators count generations, but independent texts remain 45.

## 2. Overall, Length, and Generation-Seed Results

WER,S/N,D/N,I/N are percentages; S/D/I are raw substitution/deletion/insertion counts, with `WER=(S+D+I)/N`.

| System | WER % | S/D/I | S/N % | D/N % | I/N % |
| --- | --- | --- | --- | --- | --- |
| A | 25.9052 | 359/152/90 | 15.4741 | 6.5517 | 3.8793 |
| ANoLM | 17.6293 | 238/80/91 | 10.2586 | 3.4483 | 3.9224 |
| ASplit | 21.2069 | 266/122/104 | 11.4655 | 5.2586 | 4.4828 |
| B | 19.1379 | 232/117/95 | 10.0000 | 5.0431 | 4.0948 |
| BNoLM | 10.1724 | 135/46/55 | 5.8190 | 1.9828 | 2.3707 |
| BSplit | 12.0690 | 174/48/58 | 7.5000 | 2.0690 | 2.5000 |

Length groups: 15 texts × 2 seeds per row; N is identical across six systems within each group.

| System | Length | WER % | S/D/I | N |
| --- | --- | --- | --- | --- |
| A | short | 26.7717 | 41/16/11 | 254 |
| A | medium | 22.1212 | 92/32/22 | 660 |
| A | long | 27.5249 | 226/104/57 | 1406 |
| ANoLM | short | 11.8110 | 21/5/4 | 254 |
| ANoLM | medium | 17.1212 | 80/9/24 | 660 |
| ANoLM | long | 18.9189 | 137/66/63 | 1406 |
| ASplit | short | 29.1339 | 37/12/25 | 254 |
| ASplit | medium | 22.5758 | 98/24/27 | 660 |
| ASplit | long | 19.1323 | 131/86/52 | 1406 |
| B | short | 13.3858 | 24/6/4 | 254 |
| B | medium | 16.2121 | 75/17/15 | 660 |
| B | long | 21.5505 | 133/94/76 | 1406 |
| BNoLM | short | 7.4803 | 12/4/3 | 254 |
| BNoLM | medium | 10.1515 | 49/5/13 | 660 |
| BNoLM | long | 10.6686 | 74/37/39 | 1406 |
| BSplit | short | 14.1732 | 28/4/4 | 254 |
| BSplit | medium | 16.5152 | 75/17/17 | 660 |
| BSplit | long | 9.6017 | 71/27/37 | 1406 |

Generation-seed groups: 45 texts, N=1,160 per row. These are not two training seeds.

| System | Generation seed | WER % | S/D/I |
| --- | --- | --- | --- |
| A | 17 | 25.9483 | 179/80/42 |
| A | 42 | 25.8621 | 180/72/48 |
| ANoLM | 17 | 16.8103 | 117/27/51 |
| ANoLM | 42 | 18.4483 | 121/53/40 |
| ASplit | 17 | 21.4655 | 130/75/44 |
| ASplit | 42 | 20.9483 | 136/47/60 |
| B | 17 | 19.5690 | 123/55/49 |
| B | 42 | 18.7069 | 109/62/46 |
| BNoLM | 17 | 9.9138 | 66/23/26 |
| BNoLM | 42 | 10.4310 | 69/23/29 |
| BSplit | 17 | 13.0172 | 95/27/29 |
| BSplit | 42 | 11.1207 | 79/21/29 |

Full length × seed groups, item-level rescoring, and raw transcripts are in [metrics.csv](../results/analysis/development-comparisons/metrics.csv) and adjacent `*.rescored.jsonl`. Use the corrected [length plot PNG](../results/analysis/development-plots-v2/wer_by_length.png)/[PDF](../results/analysis/development-plots-v2/wer_by_length.pdf). It redraws the same frozen summary to fix clipping of the ASplit 29.13% bar, without rescoring or modifying original evidence. BNoLM has the lowest overall point estimate for both generation seeds; this does not establish stability across training seeds/domains.

## 3. Paired Intervals and Attribution

Δ is right-system minus left-system WER in **percentage points**; negative favors the right. Intervals use 2,000 length-stratified text-cluster percentile-bootstrap replicates, seed 1832026, retaining both generation seeds per sampled text. There are 45 overall clusters/15 per length, not 90 independent texts. Intervals condition on these texts, generation seeds, and ASR; they exclude training randomness, ASR-model error, and development-selection bias, and have no multiplicity correction.

| Comparison | Attribution | Overall Δ [95% CI] | Short Δ [95% CI] | Long Δ [95% CI] |
| --- | --- | --- | --- | --- |
| A → B | Training; standard fixed | -6.767 [-9.613, -3.902] | -13.386 [-21.553, -5.200] | -5.974 [-9.879, -2.341] |
| ANoLM → BNoLM | Training; NoLM fixed | -7.457 [-9.793, -5.177] | -4.331 [-11.071, +1.969] | -8.250 [-11.499, -5.097] |
| ASplit → BSplit | Training; Split fixed | -9.138 [-12.267, -6.244] | -14.961 [-22.693, -7.518] | -9.531 [-14.196, -5.403] |
| A → ANoLM | Prefix strategy; A fixed | -8.276 [-11.652, -5.099] | -14.961 [-25.611, -6.835] | -8.606 [-13.041, -4.178] |
| B → BNoLM | Prefix strategy; B fixed | -8.966 [-13.074, -5.252] | -5.906 [-14.116, +0.388] | -10.882 [-17.379, -4.999] |
| A → ASplit | Splitting; A fixed | -4.698 [-8.005, -1.140] | +2.362 [-4.580, +11.157] | -8.393 [-13.630, -2.358] |
| B → BSplit | Splitting; B fixed | -7.069 [-11.041, -3.471] | +0.787 [-1.551, +3.782] | -11.949 [-18.448, -6.471] |
| A → BNoLM | Training + prefix | -15.733 [-19.640, -11.958] | -19.291 [-31.350, -9.054] | -16.856 [-22.017, -11.549] |
| A → BSplit | Training + splitting | -13.836 [-18.028, -9.947] | -12.598 [-21.483, -4.014] | -17.923 [-24.072, -11.956] |

Overall A → B-standard changes by−6.767 pp, ANoLM → BNoLM by−7.457 pp, and ASplit → BSplit by−9.138 pp. Each fixes inference strategy and supports a development WER improvement from ordinary continuation, still for only one training seed. B → BNoLM,−8.966 pp, is a prefix-strategy effect with fixed B weights. A → BNoLM,−15.733 pp, combines both factors and is not a single training-method or novel-architecture gain.

Short-group intervals for B → BNoLM and ANoLM → BNoLM cross 0. Splitting increases short-group point estimates by 2.362/0.787 pp for A/B while decreasing long-group WER; long-text gains do not establish no short-text cost. WER differences are not human-confirmed omission/repetition differences. All 90 pairs are scored, so common-success and missing-as-whole-reference-deletion sensitivity results/intervals are identical because scores are actually complete, not because failures were ignored.

Evidence: [six-system pairs](../results/analysis/development-comparisons/paired.csv), [within-B strategies](../results/analysis/development-within-B/paired.csv), [fixed-NoLM training](../results/analysis/development-training-no-lm/paired.csv), [fixed-Split training](../results/analysis/development-training-split/paired.csv).

## 4. Stopping Behavior, Eligibility, and Tail Candidates

| System | EOS /90 | output_limit /90 | context_limit /90 | Exceptions /90 | Scored /90 | Tail-deletion candidates ≥3 words /90 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 90 | 0 | 0 | 0 | 90 | 3 |
| ANoLM | 89 | 1 | 0 | 0 | 90 | 1 |
| ASplit | 89 | 1 | 0 | 0 | 90 | 3 |
| B | 88 | 2 | 0 | 0 | 90 | 2 |
| BNoLM | 90 | 0 | 0 | 0 | 90 | 0 |
| BSplit | 88 | 2 | 0 | 0 | 90 | 0 |

Caps and exceptions are separate: saved/scorable audio need not end in EOS, and hitting a cap does not automatically prove omitted tail content. EOS likewise does not prove all words were spoken. Tail candidates have at least 3 consecutive final reference words deleted in ASR alignment. A has 3 such outputs across 2 texts; zero detections for BNoLM/BSplit do not mean no omissions. Without actual listening, do not claim naturalness, voice quality, or human-confirmed completeness.

Eligibility under the [predefined development rule](../configs/development_selection_rule.json), with 90 scores and no exception-rate increase for every candidate:

| Candidate | Overall WER lower | Short no worse | Long no worse | Cap rate no higher | Eligible |
| --- | --- | --- | --- | --- | --- |
| ANoLM | Pass | Pass | Pass | Fail | No |
| ASplit | Pass | Fail | Pass | Fail | No |
| B | Pass | Pass | Pass | Fail | No |
| BNoLM | Pass | Pass | Pass | Pass | Yes |
| BSplit | Pass | Pass | Pass | Fail | No |

ANoLM, B, and BSplit fail because caps increase; ASplit fails both short-WER and cap criteria. Only BNoLM passes. This is point-rule eligibility, not statistical noninferiority or freezing. The coordinator’s [selection record](../results/analysis/development-selection.json) matches this independent calculation. The candidate is saved, but the final protocol was not yet frozen. A vs B-standard remains mandatory and cannot be hidden by BNoLM selection. Final evaluation must not select a new winner; 184 is training-seed sensitivity only.

## 5. What Sentence Splitting Actually Changes

A/B Split actually split36/90 generations (18/45 texts), leaving 54 single-segment outputs. B’s independent audit found all 54 unsplit pairs identical in segment text, generated-token count, stops, WAV SHA, and normalized ASR. Full token-ID sequences were not recorded, so sequence identity is not additionally claimed. Counts for1/2/3/4 segments are54/28/6/2.

| B → BSplit scope | Count | B WER % | BSplit WER % | B S/D/I | BSplit S/D/I |
| --- | --- | --- | --- | --- | --- |
| All | 90 | 19.1379 | 12.0690 | 232/117/95 | 174/48/58 |
| Actually split | 36 | 24.0859 | 11.0493 | 147/88/68 | 89/19/31 |
| Unsplit | 54 | 13.2768 | 13.2768 | 85/29/27 | 85/29/27 |

The split subset is an explanatory decomposition of a fixed strategy, not a new independent test. Total segments rise 90 → 136, adding 46 boundaries/6.9 seconds configured pauses; total caps 48,370 → 68,720, actual tokens 23,036 → 22,986. Splitting resets each segment’s context, allocates separate budgets/acoustic synthesis, and advances RNG state. This is a complete engineering strategy, not pure splitting under identical total budgets. Ten of 136 segments (5 fragments × 2 seeds) fall below the evaluation whole-text 6-token minimum, min 3; whole-text groups do not directly classify segments.

B and BSplit each have 2 output_limit cases, but different pairs. The shared `alice-p00032-s01-n1`/seed 17 case is unsplit; B hits the cap on `alice-p00467-s00-n2`/seed 42, while BSplit’s first segment caps for seed 17. Splitting did not eliminate caps. See B split audit (local only: `../results/analysis/B-split-audit/audit.md`) for alignment improvements/regressions and 180 WAV header/hash checks.

## 6. Cost, Memory, and ASR Boundaries

| System | Generation s | ASR s | Attempt s | Audio s | Actual tokens | Total configured budget |
| --- | --- | --- | --- | --- | --- | --- |
| A | 417.148 | 312.762 | 730.326 | 855.60 | 21390 | 48370 |
| ANoLM | 398.324 | 327.171 | 725.958 | 822.76 | 20569 | 48370 |
| ASplit | 439.845 | 313.084 | 753.370 | 872.94 | 21651 | 68720 |
| B | 525.280 | 327.929 | 853.693 | 921.44 | 23036 | 48370 |
| BNoLM | 387.345 | 318.587 | 706.397 | 827.64 | 20691 | 48370 |
| BSplit | 460.694 | 321.417 | 782.599 | 926.34 | 22986 | 68720 |

Times sum latest per-item records, not total wall time including loading, hashing, waiting, and historical retries. Generation, ASR, and attempt times are not additive; ASR-only resume may retain earlier generation time while replacing attempt time. Audio includes inserted pauses, so RTF is not speech-only efficiency. Although B/BSplit’s54 unsplit WAV pairs match exactly, their summed generation time changes 231.924 → 201.210 seconds, showing runtime/order confounding. Neither the table nor split time differences establish causal acceleration.

| System | Maximum observed process RSS high-water GiB | Maximum sampled MPS current GiB | Maximum sampled MPS driver GiB | >30-second outputs/90 |
| --- | --- | --- | --- | --- |
| A | 7.802 | 1.252 | 13.181 | 0 |
| ANoLM | 8.471 | 1.252 | 11.417 | 0 |
| ASplit | 8.639 | 1.252 | 7.791 | 0 |
| B | 8.023 | 1.252 | 18.894 | 1 |
| BNoLM | 9.891 | 1.252 | 7.089 | 0 |
| BSplit | 9.721 | 1.252 | 11.812 | 0 |

RSS is cumulative process high-water, not incremental sample memory; a complete peak timeline across resumed processes was not established. MPS values are post-attempt sampled maxima, not continuous device peaks or whole-machine use. Missing-field coverage, per-seed time, and resource details are in the six-system resource audit (local only: `../results/analysis/development-runtime-resources/report.md`).

Only B’s `alice-p00467-s00-n2`/seed 42 exceeds 30 seconds (31.24 seconds, output_limit). Its BSplit counterpart ends in EOS with ASR errors 24 → 9, but also changes sides of Whisper’s30-second chunk boundary, so it is not independent evidence of a pure TTS-content improvement. The other five systems have no outputs above 30 seconds; BSplit maximum 29.55 seconds. All WER contains ASR error; deletions/insertions do not establish actual omissions/repetition or imply MOS/speaker similarity.

## 7. Evidence Index and Remaining Scope

Raw JSONL, WAVs, and manifests remain intact. Values come from completed artifacts, without regeneration. development-report-evidence.json (local only: `../results/analysis/development-report-evidence.json`) retains full source SHA-256 values, all six-system groupings, nine paired comparisons/intervals, filter booleans, cap cases, and 540-WAV verification coverage, and binds corrected plots, selection, and closing reconciliation. Raw event hashes:

| System | raw.jsonl SHA-256 |
| --- | --- |
| A | `0ccaf495510f58ab6b4ad2fba0f90dd74f294a3f90fdc67c61e4dccabd1c019a` |
| ANoLM | `e24d0639d48ff23ec62ce14480df893bfef71e206d4e96f33501bfb9a1b8144e` |
| ASplit | `aa0cd5c2ee57e0e6630bb4e686c957fe41285a5a0d515838917c45d6fe1e71a5` |
| B | `4e09360e434b4274e9c3ac359ca4e222b8edc7ebd6766e900a08a34cdd0c1cea` |
| BNoLM | `014c3b4a092b83dcadf2fb9281ea4cc47167c900302fcfd72af50fd58889ead2` |
| BSplit | `ac4d9a9ad4c0960e56366cc21e973ed97e4bfb958999f742c6a81d1edf3e25f0` |

Listening entry points: six-system representative samples (local only: `../results/listening/development-comparisons/INDEX.md`), within-B paired strategies (local only: `../results/listening/development-within-B/INDEX.md`). Selection uses actual ASR metrics with saved rules; organizing an index is not human listening.

These development results support further BNoLM testing and show effects/stop-behavior costs from ordinary training, LM prefixes, and sentence splitting. Eligibility comes from selection data, so effect sizes have selection bias. Independent final evaluation after freezing and seed 184 sensitivity remained required; this chapter does not represent their then-unfinished results.
