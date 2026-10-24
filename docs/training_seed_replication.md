# Training Seed 184: Development Sensitivity Check

This check retains the established seed183/BNoLM primary system and does not use seed 184 to reselect checkpoints or strategies. Overall development WER point estimates are close across seeds, while length subgroups vary substantially. Two training seeds provide one sensitivity comparison, not an estimate of population training variance or a guarantee across all repetitions.

Standard uses the fixed LM reference prefix; no-lm removes only that prefix while keeping acoustic conditioning; split synthesizes actual sentence segments and concatenates them. Training seeds183/184 and inference seeds17/42 represent different randomness levels. This report reads completed development outputs only. The lead agent separately audits training exposure; this review neither opened/hashed checkpoints nor read final data.

## Completeness and Provenance

All three replication analyses have complete markers matching the analysis-completion SHAs in B-seed184-execution/completion.json. Verification covered 27 artifact hashes, six raw/manifest identities, and paths, hashes, and header durations for 540 WAVs. Every cell has90/90 complete and scored, 0 exceptions, pending, or missing items: 45 texts × 2 inference seeds, 15 texts per length group. All three paired comparisons match id, seed, original text, and length with no unpaired items.

Within-strategy settings differ only in checkpoint path. Checkpoint identity uses existing manifest SHAs, not rehashing. All S/D/I, group summaries, and three complete paired bootstrap results were recomputed with existing analysis functions and match field by field. Normalization is Whisper EnglishTextNormalizer unicode-quotes v2. All 54 source hashes and per-WAV verification records are in replication-report-evidence.json (local only: `../results/analysis/replication-report-evidence.json`).

## Overall Results and Paired Intervals

Δ = seed 184 − seed 183 in WER percentage points; negative means lower ASR WER in this sample. Each side has 2320 reference words.

| Strategy | 183 WER / S-D-I | 184 WER / S-D-I | Δ points | Paired 95% CI, points | 183 → 184 caps |
| --- | --- | --- | ---: | --- | --- |
| standard | 19.1379% / 232-117-95 | 17.0690% / 231-81-84 | -2.0690 | [-4.3702, +0.1355] | 2 → 1 |
| no-lm | 10.1724% / 135-46-55 | 10.3017% / 144-38-57 | +0.1293 | [-2.0381, +2.1803] | 0 → 0 |
| split | 12.0690% / 174-48-58 | 11.7672% / 169-50-54 | -0.3017 | [-2.2573, +1.7552] | 2 → 0 |

All three overall intervals include zero. They neither establish systematic 184 superiority nor prove equivalence. No-lm has the lowest overall point estimate among the three strategies for both seeds; this checks the established choice rather than selecting anew.

Intervals use 2000 percentile-bootstrap replicates, RNG seed 1832026, resampling normalized text clusters within length strata while retaining both paired inference seeds and recalculating micro-WER. They condition on these 45 texts/sources, the two observed models/inference seeds, and current ASR, and **exclude training randomness and ASR-model uncertainty**. They do not estimate variance from two training-seed samples.

## Length and Inference Seeds

| Strategy | Length | 183 WER / S-D-I | 184 WER / S-D-I | Δ95% CI, points |
| --- | --- | --- | --- | --- |
| standard | short | 13.3858% / 24-6-4 | 17.7165% / 21-12-12 | [-0.747, +10.256] |
| standard | medium | 16.2121% / 75-17-15 | 12.8788% / 53-13-19 | [-7.581, +1.088] |
| standard | long | 21.5505% / 133-94-76 | 18.9189% / 157-56-53 | [-5.672, +0.505] |
| no-lm | short | 7.4803% / 12-4-3 | 7.0866% / 13-3-2 | [-3.880, +2.757] |
| no-lm | medium | 10.1515% / 49-5-13 | 12.4242% / 60-13-9 | [-1.065, +5.033] |
| no-lm | long | 10.6686% / 74-37-39 | 9.8862% / 71-22-46 | [-3.857, +2.086] |
| split | short | 14.1732% / 28-4-4 | 20.8661% / 30-9-14 | [+1.695, +12.500] |
| split | medium | 16.5152% / 75-17-17 | 13.0303% / 56-12-18 | [-7.798, +0.557] |
| split | long | 9.6017% / 71-27-37 | 9.5306% / 83-29-22 | [-2.541, +2.374] |

The clearest local sensitivity is split/short: 14.1732% → 20.8661%,Δ+6.6929 points, CI[+1.6946,+12.5000],S/D/I28/4/4 → 30/9/14. This exploratory subgroup result has no multiplicity correction. It suggests short-text sensitivity to training seed, not stable deterioration for all short texts, and should not be hidden by similar aggregate scores.

| Strategy | Inference seed | 183 WER | 184 WER | 183 → 184 stopping reasons |
| --- | ---: | ---: | ---: | --- |
| standard | 17 | 19.5690% | 16.4655% | {'eos': 44,'output_limit': 1} → {'eos': 45} |
| standard | 42 | 18.7069% | 17.6724% | {'eos': 44,'output_limit': 1} → {'eos': 44,'output_limit': 1} |
| no-lm | 17 | 9.9138% | 10.7759% | {'eos': 45} → {'eos': 45} |
| no-lm | 42 | 10.4310% | 9.8276% | {'eos': 45} → {'eos': 45} |
| split | 17 | 13.0172% | 12.0690% | {'eos': 43,'output_limit': 2} → {'eos': 45} |
| split | 42 | 11.1207% | 11.4655% | {'eos': 45} → {'eos': 45} |

Inference-seed changes are not uniformly directional. Full by_length_and_seed results/S/D/I are retained in original summaries embedded in the evidence JSON. Two draws for one text are not independent texts.

## Caps, ASR Windows, and Evaluation Boundaries

Seed 183 standard/split each have 2 output_limit cases; 184 standard has 1; 184 no-lm/split all end in EOS. No run has context_limit or exceptions. Complete means audio/ASR processing ended, not that every word was spoken.

183/184 standard each contain one 31.24-second output,`alice-p00467-s00-n2`, seed 42, crossing Whisper's 30-second chunk window. The other four conditions have no audio over 30 seconds; 184 no-lm max 26.28 seconds, 184 split 25.55 seconds. Crossing the window flags a transcription-path limitation, not automatic audio failure. ASR deletions/insertions do not establish spoken omissions/repetition. No human listening, MOS, or voice-quality evaluation was performed.

## Measured Evaluation Resources

These sum latest terminal records, not pipeline wall time, and exclude full model loading/prior failed attempts. Generation, ASR, and attempt times are not additive; ASR-only resume can retain earlier generation time while replacing attempt time. RTF includes concatenation pauses in audio duration and uses records with both a numerator and positive duration. Every column is known for90/90 items. Training exposure, time, and recovery belong to separate audits.

| System | Generation s | ASR s | Attempt s | Audio s | Generation RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| B183 | 525.280 | 327.929 | 853.693 | 921.44 | 0.5701 |
| B183NoLM | 387.345 | 318.587 | 706.397 | 827.64 | 0.4680 |
| B183Split | 460.694 | 321.417 | 782.599 | 926.34 | 0.4973 |
| B184 | 419.348 | 316.241 | 736.057 | 923.68 | 0.4540 |
| B184NoLM | 380.467 | 311.515 | 692.425 | 826.72 | 0.4602 |
| B184Split | 436.878 | 312.886 | 750.226 | 917.26 | 0.4763 |

| System | Maximum observed process RSS high-water, GiB | Maximum sampled MPS current, GiB | Maximum sampled MPS driver, GiB |
| --- | ---: | ---: | ---: |
| B183 | 8.023 | 1.252 | 18.894 |
| B183NoLM | 9.891 | 1.252 | 7.089 |
| B183Split | 9.721 | 1.252 | 11.812 |
| B184 | 9.997 | 1.252 | 11.519 |
| B184NoLM | 9.745 | 1.252 | 10.542 |
| B184Split | 10.256 | 1.252 | 6.815 |

RSS is cumulative process high-water, not per-item incremental memory. MPS is sampled after attempts, not continuous peak or whole-machine use. Machine state/process lifetime can affect both time and memory; this table does not establish that one training seed is faster or more memory-efficient. Full records/exact commands: replication-runtime-resources (local only: `../results/analysis/replication-runtime-resources/manifest.json`).

## Plot Checks and Verification Entry Points

Each of three analyses had its PNG and PDF inspected, six visual renderings total; PDFs were rendered using pdftoppm. Titles, legends, axes, and length labels were readable without clipping. “Scored” and “unscored=deletion” panels are identical because every item is scored, not because they are independent replications. Bars are point estimates; consult report/summary intervals rather than inferring significance from heights.

- [Standard plot](../results/analysis/replication-standard/wer_by_length.png) / [analysis](../results/analysis/replication-standard/summary.json)
- [No-lm plot](../results/analysis/replication-no-lm/wer_by_length.png) / [analysis](../results/analysis/replication-no-lm/summary.json)
- [Split plot](../results/analysis/replication-split/wer_by_length.png) / [analysis](../results/analysis/replication-split/summary.json)

Existing analyses/core code remain unchanged. These are development sensitivity results. The independent final evaluation keeps183/BNoLM primary and the three 184 systems as planned secondary sensitivity checks, without reselection.
