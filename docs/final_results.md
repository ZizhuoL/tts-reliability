# Independent Final Evaluation Results

**The preselected primary system, B183NoLM, achieved an overall ASR WER of 9.4828% on the independent final set, compared with 25.3979% for starter A; long-text WER was 9.3884% versus 27.6288%, and short-text performance also improved.** The primary system retains the preselected training seed 183 and removal of the LM reference prefix. Ordinary continued training alone also reduced WER, but the combined improvement cannot all be attributed to training. The no-LM result for the second training seed, 184, contains a severe localized outlier; these results do not establish stability across training seeds or general elimination of failures.

## 1. Frozen Identity, Completion Scope, and Verification

The [formal protocol](../results/final-protocol-v1/protocol.json) was frozen at 2026-10-20 14:18:25 UTC, with SHA-256 `1784f64b2ed3ddf1094b5b0795faffa86c99812dfdbfba8924db01be0b59aa38`. The runner recorded frozen-protocol verification at 14:19:06.992572 UTC, followed by verification of the 60 final inputs at 14:19:06.993102. In these records, method and endpoint selection preceded the first read. These are local process records, not trusted timestamps or access controls; they cannot prove that no earlier access occurred outside the records. The primary comparison is A→BNoLM, with overall and long-text micro WER as the two primary outcomes. A→B standard is a required research comparison, interpreted as exploratory under its frozen role; other strategies and seed 184 are exploratory or secondary sensitivity analyses. Final results did not change the selection.

The [completion record](../results/final-evaluation-v1/completion.json) was written at 16:49:20 UTC. All 30 logged stage_exit events had exit code 0, and all 30 stage markers and their artifact hashes matched. Each of the nine conditions had 120/120 complete, scored attempts: 1,080 in total, with 0 exceptions, 0 pending attempts, 0 scoring errors, and 0 missing pairs. SHA and header-duration checks passed for all 1,080 run WAV files; all 180 copied WAV files in the listening bundle matched their source hashes. Here, complete means completion of the synthesis/transcription pipeline, **not successful reading of all content or a human quality pass rate**.

This chapter reaggregated per-attempt S/D/I from the complete artifacts and checked 54 overall, length, and inference-seed aggregate groups. Intervals came from eight completed analyses whose hashes were verified. No audio was regenerated, ASR rerun, or normalization changed during this reporting step, and no human listening was performed. Source hashes, exact summaries, per-audio checks, and selected examples are retained in the report evidence JSON (local only: `../results/analysis/final-report-evidence.json`). A separate independent audit (local only: `../results/analysis/final-independent-audit.md`) renormalized raw ASR and recomputed the overall/length bootstrap results field by field for eight analyses and 17 paired comparisons, obtaining consistent results. The [integrity audit](../results/analysis/final-root-integrity-audit.json) also checked 154 frozen bindings and every stage's file inventory.

## 2. Primary Comparison and Uncertainty

Each system used 60 independent project texts, 20 per length stratum, with inference seeds 17/42. There were 3,016 reference words overall, with 316/836/1,864 in short/medium/long. WER is a word-weighted micro average. Δ is method minus control; negative values indicate lower ASR error.

| A→BNoLM outcome | A | BNoLM | Δ WER, percentage points | 95% CI, percentage points |
| --- | --- | --- | --- | --- |
| overall | 25.3979% | 9.4828% | -15.9151 | [-18.4507, -13.0477] |
| long | 27.6288% | 9.3884% | -18.2403 | [-22.1039, -14.1928] |
| short | 29.4304% | 10.1266% | -19.3038 | [-26.7612, -12.8924] |
| medium | 18.8995% | 9.4498% | -9.4498 | [-12.4695, -6.7567] |

Overall WER decreased by 15.9151 percentage points, a relative reduction of 62.66%; long-text WER decreased by 18.2403 points. The overall gain did not come at the expense of short texts: their point estimate decreased from 29.4304% to 10.1266%, and the paired interval was also below 0. Short and medium results are supporting analyses, not a non-inferiority test with a prespecified equivalence margin. The primary system and A each had 1 output cap, both in the long group. No increase in the primary system's cap count was observed, but this evidence covers only the fixed texts and seeds and does not prove equal underlying failure rates.

The bootstrap resampled text clusters within length strata, retaining both inference seeds for a text together, with 2,000 replicates and random seed 1832026, and recomputed micro WER. The two primary outcomes have **separate 95% percentile intervals without multiple-comparison correction**, not a claim of 95% joint coverage. The intervals condition on these sources, observed models and sampling seeds, and the fixed ASR; they do not incorporate population-level training randomness or ASR-model uncertainty. The 120 attempts are not treated as 120 independent texts. Since every attempt was scored, all 120 pairs were jointly successful, and the sensitivity analysis treating failed attempts as whole-sentence deletions produced the same numbers. That sensitivity analysis is not a mathematical worst-case bound: insertions can make WER exceed 100%.

## 3. Nine Systems: Overall, Length, and Inference-Seed Results

A is the course starter; B is the full-epoch endpoint for seed 183. The three seed-184 conditions are sensitivity analyses only. NoLM removes the LM reference prefix while retaining the same acoustic reference. Split separates actual sentences, synthesizes each, and concatenates the outputs. All systems had 0 exceptions and 0 context caps. In the table, cap means output_limit specifically; EOS count divided by 120 is an operational completion rate requiring available audio and EOS. Normalized exact match means only that ASR and reference text agree.

| System | Overall WER | S/D/I | EOS / cap | Normalized exact match |
| --- | --- | --- | --- | --- |
| A | 25.3979% | 384/229/153 | 119 / 1 | 9.1667% |
| ANoLM | 14.6883% | 261/66/116 | 120 / 0 | 19.1667% |
| ASplit | 21.0875% | 345/150/141 | 120 / 0 | 9.1667% |
| B | 17.7056% | 295/102/137 | 115 / 5 | 23.3333% |
| BNoLM | 9.4828% | 151/40/95 | 119 / 1 | 37.5000% |
| BSplit | 11.6379% | 225/27/99 | 119 / 1 | 22.5000% |
| B184 | 17.0093% | 272/119/122 | 119 / 1 | 18.3333% |
| B184NoLM | 17.0424% | 168/67/279 | 116 / 4 | 28.3333% |
| B184Split | 13.0968% | 215/55/125 | 120 / 0 | 17.5000% |

Each length group has 40 attempts; each seed has 60. The table retains S/D/I to avoid interpreting every WER reduction broadly as “fewer omissions.”

| System | Short WER (S/D/I) | Medium WER (S/D/I) | Long WER (S/D/I) |
| --- | --- | --- | --- |
| A | 29.4304% (50/16/27) | 18.8995% (95/24/39) | 27.6288% (239/189/87) |
| ANoLM | 17.0886% (32/8/14) | 11.4833% (66/2/28) | 15.7189% (163/56/74) |
| ASplit | 29.4304% (50/19/24) | 16.7464% (91/18/31) | 21.6202% (204/113/86) |
| B | 18.0380% (35/0/22) | 9.6890% (47/16/18) | 21.2446% (213/86/97) |
| BNoLM | 10.1266% (22/3/7) | 9.4498% (48/9/22) | 9.3884% (81/28/66) |
| BSplit | 16.7722% (36/0/17) | 11.0048% (53/8/31) | 11.0515% (136/19/51) |
| B184 | 17.4051% (33/4/18) | 11.6029% (59/17/21) | 19.3670% (180/98/83) |
| B184NoLM | 11.7089% (21/6/10) | 35.2871% (53/10/232) | 9.7639% (94/51/37) |
| B184Split | 18.3544% (32/3/23) | 11.3636% (61/9/25) | 12.9828% (122/43/77) |

| System | Inference seed 17 WER (S/D/I) | Inference seed 42 WER (S/D/I) |
| --- | --- | --- |
| A | 25.6631% (189/118/80) | 25.1326% (195/111/73) |
| ANoLM | 14.5889% (123/35/62) | 14.7878% (138/31/54) |
| ASplit | 21.8833% (173/91/66) | 20.2918% (172/59/75) |
| B | 16.3130% (131/53/62) | 19.0981% (164/49/75) |
| BNoLM | 9.2175% (72/27/40) | 9.7480% (79/13/55) |
| BSplit | 12.1353% (109/16/58) | 11.1406% (116/11/41) |
| B184 | 18.5013% (156/48/75) | 15.5172% (116/71/47) |
| B184NoLM | 10.8753% (71/59/34) | 23.2095% (97/8/245) |
| B184Split | 14.9867% (123/34/69) | 11.2069% (92/21/56) |

A's short/medium/long WER values of 29.43%/18.90%/27.63% still do not support simple monotonic deterioration with length. BNoLM is around 9–10% in all three groups, but the natural texts cover only the fixed range of 6–72 text tokens; this does not establish robustness for much longer texts or other domains.

[Final WER-by-length PNG](../results/analysis/final-delivery-plots-v2/overview/wer_by_length.png) / [PDF](../results/analysis/final-delivery-plots-v2/overview/wer_by_length.pdf). Delivery version v2 only corrects layout; the frozen original figures and numbers are retained. Source bindings are in the figure manifest (local only: `../results/analysis/final-delivery-plots-v2/manifest.json`). The reporting review checked all nine systems' legend entries, axes, and bar tops in the revised overview PNG and found them fully visible. The two panels are identical because every attempt was scored, not because the experiment was repeated. Bars show point estimates only; use the table above and the summaries for intervals.

## 4. Separating Training, Prefix Removal, and Sentence Splitting

The following comparisons are exploratory research comparisons under their frozen roles. Differences and intervals are in percentage points, without multiple-comparison correction.

| Exploratory comparison | Overall Δ | Overall 95% CI | Long Δ | Long 95% CI |
| --- | --- | --- | --- | --- |
| A→B, training under standard | -7.6923 | [-10.3539, -4.9519] | -6.3841 | [-10.2990, -2.2777] |
| ANoLM→BNoLM, training under no-lm | -5.2056 | [-7.5490, -2.9504] | -6.3305 | [-9.7312, -2.7143] |
| ASplit→BSplit, training under split | -9.4496 | [-11.5158, -7.3398] | -10.5687 | [-13.4124, -7.7068] |
| B→BNoLM, prefix removal for seed 183 | -8.2228 | [-11.5957, -5.0301] | -11.8562 | [-17.1801, -6.6223] |
| B→BSplit, sentence splitting for seed 183 | -6.0676 | [-9.5942, -2.6463] | -10.1931 | [-15.5140, -5.0142] |
| B184→B184NoLM, prefix removal for seed 184 | +0.0332 | [-9.1573, +16.3636] | -9.6030 | [-14.3016, -5.4039] |
| B184→B184Split, sentence splitting for seed 184 | -3.9125 | [-7.0943, -1.0035] | -6.3841 | [-11.2426, -1.9604] |

A→B decreased from 25.3979% to 17.7056%, supporting a benefit from ordinary continued training under the current automatic evaluation conditions. The matched no-lm training comparison was 14.6883%→9.4828%, and the matched split training comparison was 21.0875%→11.6379%, separating training from strategy changes. The combined 15.9151-point A→BNoLM gain cannot all be attributed to the additional epoch, and these comparisons do not establish the internal causal mechanism behind the gains. B standard also had 5 caps versus A's 1, which must be reported alongside its WER improvement.

B→BSplit reduced overall WER from 17.7056% to 11.6379%. Among the 52 attempts actually split, WER changed from 21.4994% to 10.1611%; count-based metrics were identical for the other 68 attempts. The 120 outputs contained 192 segments, adding 72 boundaries and 10.8 seconds of pauses. Total effective output budget increased from 64,802 to 97,724, while actual generated tokens changed from 30,883 to 29,630. Sentence splitting is not a comparison with equal total generation budgets. Total generation time for the actually split subset was almost unchanged, 346.684→346.767 seconds, whereas the unsplit subset changed from 281.696 to 266.103 seconds. The overall time decrease therefore cannot be attributed directly to acceleration from sentence splitting. See the [within-seed-183 costs](../results/final-evaluation-v1/costs/within-B183/summary.json) for the full decomposition.

## 5. Negative Sensitivity Result for the Second Training Seed

| Same strategy, 183→184 | Seed 183 WER | Seed 184 WER | Δ, percentage points | 95% CI, percentage points |
| --- | --- | --- | --- | --- |
| standard | 17.7056% | 17.0093% | -0.6963 | [-3.3709, +1.8875] |
| no-lm | 9.4828% | 17.0424% | +7.5597 | [-1.4317, +23.8887] |
| split | 11.6379% | 13.0968% | +1.4589 | [-0.7402, +3.5940] |

All overall intervals include 0; this does not establish equivalence between the two training seeds. In particular, B184NoLM's overall WER of 17.0424% is 7.5597 points higher than seed 183's 9.4828%, with a wide interval. Its medium WER is 35.2871%, and inference-seed-42 WER is 23.2095%. Within B184, removing the prefix changed overall WER by only +0.0332 points. Although long-text WER remained lower, **the overall benefit observed for seed 183 did not reproduce stably for seed 184**. A→B184NoLM had an overall Δ of −8.3554 points and a 95% CI of [−17.4834, +8.3625], crossing 0; its long-text Δ of −17.8648 and CI of [−22.1727, −13.5023] remained negative. This does not change the frozen seed-183 primary system.

The specific outlier is `alice-p00741-s00-n2 / seed42`. It was assigned to medium in advance using its actual text-token count, although its normalized reference has only 10 words. B184NoLM produced a 223-word transcript, S/D/I=9/0/213, WER=2220%, and output_limit at 500 generated tokens; its audio lasts 20.00 seconds, and the raw transcript repeatedly contains “a.” This attempt contributes 213 of the system's 279 insertions and must be retained, without removal or clipping to improve the average. It does not exceed 30 seconds and cannot be dismissed as a consequence of Whisper's long-audio chunk window. The evidence establishes severe ASR/reference disagreement and reaching the generation cap; without human listening, it does not establish whether the repetition comes from speech, recognizer hallucination, or both. The matching seed-183 NoLM output reached EOS, lasted 4.96 seconds, and had S/D/I=3/0/2. Two training seeds are insufficient to estimate the population distribution of training randomness, and the wide interval also reflects heavy-tailed errors in this small sample.

## 6. Failure Denominators, Automatic Candidates, and Listening Material

All nine conditions retain 13 output caps: 12 long and 1 medium, the B184NoLM outlier above. All conditions have 0 context caps and 0 exceptions. The primary BNoLM system's only cap is `holmes-p01147-s00-n3 / seed17`; having “only 1” is not equivalent to having no failures. A fixed ASR word-alignment rule automatically selected candidates with at least 3 consecutive unaligned words at the end:

| System | Candidates with ≥3 consecutive trailing deletions / 120 | Trailing deleted words |
| --- | --- | --- |
| A | 6 | 68 |
| ANoLM | 0 | 0 |
| ASplit | 2 | 14 |
| B | 3 | 23 |
| BNoLM | 0 | 0 |
| BSplit | 1 | 6 |
| B184 | 3 | 20 |
| B184NoLM | 1 | 4 |
| B184Split | 0 | 6 |

“Trailing deleted words” sums trailing deletions across all available alignments, including attempts with fewer than 3 words that do not meet the candidate threshold; it is not a sum over candidates alone. B184Split has 4 attempts with 1 trailing word deleted and 1 attempt with 2 deleted, yielding 6 words but still 0 candidates.

The change from A's 6 candidates to BNoLM's 0 supports only that this automatic rule no longer detected candidates; it does not prove the absence of actual omissions. For example, A's `alice-p00741-s00-n2 / seed17` reached EOS normally, but the final three words “behead that dormouse” were unaligned. EOS alone does not establish content completeness. Original candidates and alignments are retained in the failure directory (local only: `../results/final-evaluation-v1/failures/A/tail_candidates.jsonl`).

The listening index (local only: `../results/final-evaluation-v1/listening/INDEX.md`) contains 180 deduplicated audio files and 102 paired presentations, selected by highest/lowest ASR WER within each system/length group and by paired improvement/regression. It preserves text, ASR, stop reason, seeds, metrics, and selection rules. No human listening was performed for this report; the examples below are automatic selections, not perceptual-quality rankings.

| Comparison | ID / seed | Automatic selection | ASR error count | Control WAV | Method WAV |
| --- | --- | --- | --- | --- | --- |
| BNoLM→B184NoLM | alice-p00741-s00-n2/42 | medium/regression | 5→222 / 10 words | BNoLM audio (local only: `../results/final-evaluation-v1/listening/audio/BNoLM/S2f96aa568a24.wav`) | B184NoLM audio (local only: `../results/final-evaluation-v1/listening/audio/B184NoLM/S16e662d617a9.wav`) |
| A→BNoLM | alice-p00575-s00-n1/17 | short/improvement | 8→1 / 6 words | A audio (local only: `../results/final-evaluation-v1/listening/audio/A/Sec42c599ac7e.wav`) | BNoLM audio (local only: `../results/final-evaluation-v1/listening/audio/BNoLM/Sf69f38face25.wav`) |
| A→BNoLM | alice-p00501-s01-n1/17 | medium/improvement | 8→0 / 24 words | A audio (local only: `../results/final-evaluation-v1/listening/audio/A/Scf1f1ac00e1e.wav`) | BNoLM audio (local only: `../results/final-evaluation-v1/listening/audio/BNoLM/S4c5ba25290d8.wav`) |
| A→BNoLM | alice-p00062-s03-n1/42 | long/improvement | 24→4 / 51 words | A audio (local only: `../results/final-evaluation-v1/listening/audio/A/S3bf1c691f014.wav`) | BNoLM audio (local only: `../results/final-evaluation-v1/listening/audio/BNoLM/S4d77207a43ed.wav`) |
| A→BNoLM | alice-p00280-s01-n1/17 | long/regression | 2→4 / 44 words | A audio (local only: `../results/final-evaluation-v1/listening/audio/A/Sb91e70cd04ab.wav`) | BNoLM audio (local only: `../results/final-evaluation-v1/listening/audio/BNoLM/S40a473def35b.wav`) |

The long-text regression example is retained: the primary system did not improve every attempt. The seed-184 outlier and better/worse primary-comparison examples use the same inspectable audio paths, avoiding a showcase of successes alone.

## 7. Measured Time, Resources, and ASR Limits

Every timing column has 120/120 known values. Generation covers the recorded generation process; ASR is listed separately. Attempt covers the whole attempt, so the three columns must not be added together. RTF is total generation seconds divided by the matched attempts' total audio duration; split audio includes inserted pauses.

| System | Generation s | ASR s | Attempt s | Audio s | Generation RTF | Attempts >30 s |
| --- | --- | --- | --- | --- | --- | --- |
| A | 537.165 | 419.397 | 957.217 | 1126.960 | 0.4766 | 0 |
| ANoLM | 502.208 | 414.033 | 916.869 | 1074.760 | 0.4673 | 0 |
| ASplit | 605.687 | 416.735 | 1023.061 | 1148.840 | 0.5272 | 0 |
| B | 628.379 | 434.595 | 1063.675 | 1235.320 | 0.5087 | 1 |
| BNoLM | 516.661 | 426.325 | 943.608 | 1080.560 | 0.4781 | 0 |
| BSplit | 612.870 | 415.765 | 1029.290 | 1196.000 | 0.5124 | 0 |
| B184 | 565.162 | 385.185 | 950.974 | 1181.360 | 0.4784 | 0 |
| B184NoLM | 530.379 | 446.466 | 977.471 | 1116.040 | 0.4752 | 1 |
| B184Split | 603.917 | 394.319 | 998.854 | 1187.120 | 0.5087 | 0 |

Across all nine conditions, generation totaled 5,102.429 seconds, ASR 3,752.819 seconds, attempts 8,861.019 seconds, and audio 10,346.96 seconds. The 30 subprocess stages totaled 9,010.234 seconds of wall time, including 8,974.087 seconds for the nine evaluation subprocesses. The interval from the first frozen-verification event to all_stages_completed was 9,013.277 seconds, approximately 2.504 hours. This is the interval for one successful final pipeline, including subprocess loading/verification and postprocessing. It excludes prior training, development, and preparation before startup, and is not the entire project cost. Accepted/discarded/probe training costs and unknown wall-time components are reported separately in the [training chapter](training_results.md).

B standard and B184NoLM each have one 31.68-second audio file, both `holmes-p01214-s04-n2 / seed17`; the other seven systems have none exceeding 30 seconds. Crossing Whisper's chunk window changes the ASR path, so these individual cases cannot be interpreted solely as TTS-quality gains. The primary BNoLM system's longest audio is 29.04 seconds, which still does not make Whisper ground truth.

| System | Maximum observed RSS high-water, GiB | Maximum sampled MPS current, GiB | Maximum sampled MPS driver, GiB |
| --- | --- | --- | --- |
| A | 8.492 | 1.252 | 12.978 |
| ANoLM | 10.915 | 1.252 | 8.507 |
| ASplit | 9.156 | 1.252 | 11.740 |
| B | 9.682 | 1.252 | 18.524 |
| BNoLM | 8.851 | 1.252 | 14.976 |
| BSplit | 9.781 | 1.252 | 16.217 |
| B184 | 8.560 | 1.252 | 16.108 |
| B184NoLM | 8.675 | 1.252 | 14.249 |
| B184Split | 10.164 | 1.252 | 10.012 |

RSS is the maximum observed process-wide cumulative high-water value, not incremental memory per attempt. MPS current/driver values are discrete samples after attempts, not continuous peaks or whole-machine usage. System order, machine state, audio length, and process lifetime can affect speed and memory. BNoLM's total generation time is slightly lower than A's, but its RTF is slightly higher, 0.4781 versus 0.4766; selecting only favorable timing values would not support a claim of universal acceleration. See the [resource summary](../results/final-evaluation-v1/resources/summary.json) for the original scope definitions.

## 8. Supported Conclusions

With the fixed reference, course starter, two separate full-epoch continuation runs, fixed natural English texts, and Whisper evaluation, the frozen seed-183/BNoLM primary system achieved lower overall and long-text ASR WER on the independent 60-text × 2-inference-seed evaluation while also improving short texts. Training comparisons within the same strategy support a separate contribution from ordinary continued training. This does not establish human-perceived quality, speaker similarity, quality across domains or much longer texts, or stable reproduction of the overall gain with the second training seed.

The original hypothesis of stable overall deterioration with length was not supported. The project instead studied observable prefix sensitivity, training benefits, and failure boundaries, retaining the severe localized seed-184 outlier, caps, worse examples, and all costs. Further study should use a new protocol and data, independent human transcription/listening, and more training replications. Reselecting methods on this final set would invalidate calling it an independent test. See the [technical report](technical_report.md) and [reproduction notes](reproduction_notes.md) for implementation, attribution, training, and reproduction details.
