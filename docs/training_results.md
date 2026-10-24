# Ordinary Continuation B: Two Full-Epoch Training Seeds and Validation

Evidence cutoff: 2026-10-20, including completed seed 184 training/validation. Both B runs independently start from the same course starter and continue one epoch on the original valid-data distribution; seed 184 does not continue seed 183. Full course-development token cross-entropy decreased from starter **4.54146120** to **4.46616567 (183)** and **4.46599462 (184)**. This supports improved conditional token prediction on that set, not claims about autoregressive omissions, WER, naturalness, or voice quality. This chapter covers training and teacher-forced validation only, excludes generation comparisons, and did not read the then-sealed final set.

## 1. Training Scope and Endpoints

B continues the course speech-token LM from starter weights with `long_weight=1.0`, without long-text oversampling or training all of CosyVoice 2 from scratch. Environment: Apple M2 Max, 64 GiB unified memory, macOS 15.7.9, PyTorch 2.8.0/MPS. Configuration: float32, batch 8, packed sequences, padding to 64, maximum context 2048. AdamW uses learning rate 0.0002, weight decay 0.01, gradient clipping 1.0, and supervised-target-token scheduling with 10% warmup.

| Final exposure | Seed 183 | Seed 184 |
| --- | ---: | ---: |
| Optimizer updates | 44,348 | 44,348 |
| Sample presentations / logged unique samples | 354,780 / 354,780 | 354,780 / 354,780 |
| Complete epochs (summary) | 1 | 1 |
| Supervised targets including speech EOS | 50,450,231 | 50,450,231 |
| Nonpadding input tokens | 58,399,911 | 58,399,911 |
| Input positions including padding | 71,885,568 | 71,901,440 |
| Padding fraction | 18.7599% | 18.7778% |
| Independent sampler/checkpoint exposure audit | Complete | Complete |

Padding is `1 - valid inputs / input positions including padding`, not a fraction of supervised targets. Seed 183's final counts match full deterministic-sampler reconstruction: all 44,348 batches match, every valid sample appears exactly once on the retained branch, and budget overshoot is zero. Raw log rows lack per-sample IDs, so this is sampler/log-based coverage evidence, not a hardware-execution audit independent of the sampler implementation.

Seed 184's lightweight log audit found continuous steps 1–44,348, no rollback, duplicates, or incomplete rows, and finite values throughout. Per-step targets, padding, cumulative targets, declared unique counters, and all summary totals match. There are 44,347 eight-sample batches and a four-sample batch at step 19,897. After the pipeline ended, the lead agent independently reconstructed the full sampler and checked the checkpoint: **zero field mismatches across 44,348 steps; counters, sampler, and seen_indices match; each of 354,780 valid samples appears once; overshoot 0.** Like seed 183, this depends on deterministic sampling, logs, and saved state rather than independent hardware tracing. Seeds change grouping/order, hence padding and per-step targets; total target exposure and optimization rules match.

Seed 183 coverage by exhaustive training-length groups was short 1–17, medium 18–43, long≥44 text tokens: 178,570/142,883/33,327 samples and 11,224,777/26,155,136/13,070,318 targets. These training-wide groups differ from natural-text evaluation boundaries 6–17/18–43/44–72.

**Both comparison endpoints are `latest.pt`, step 44,348. Seed 183 remains primary; 184 only checks randomness and is not selected by results.** Seed 183's 1,024-sample course-dev-selected `best.pt` is step 44,000, subset CE 4.45879289, with 50,052,809 targets—397,422 fewer than the primary endpoint. Seed 184's minimum periodic subset CE is 4.46424575, also step 44,000. Do not substitute best for fixed-budget results or directly compare subset/full-development CE. Any supplementary best result must disclose exposure and validation scope.

Evidence: [183 exposure (path-redacted export)](../results/repository-evidence/analysis/B-final-exposure.json), 183 configuration (local only: `../checkpoints/B-ordinary-seed183/config.json`), 183 summary (local only: `../checkpoints/B-ordinary-seed183/summary.json`), 184 log audit (local only: `../results/analysis/B184-training-log-audit.json`), [184 exposure (path-redacted export)](../results/repository-evidence/analysis/B184-final-exposure.json), 184 configuration (local only: `../checkpoints/B-ordinary-seed184/config.json`), 184 summary (local only: `../checkpoints/B-ordinary-seed184/summary.json`).

## 2. Matched-Batch Full Course-Development Validation

| Endpoint | Token CE ↓ | Token accuracy ↑ | Samples | Supervised targets |
| --- | ---: | ---: | ---: | ---: |
| Starter | 4.5414612033 | 11.1869824% | 9,756 | 1,362,834 |
| B183 latest, step 44,348 | 4.4661656673 | 11.6182895% | 9,756 | 1,362,834 |
| B184 latest, step 44,348 | 4.4659946224 | 11.6315707% | 9,756 | 1,362,834 |
| B183 − Starter | −0.0752955360 | +0.4313071 percentage points | — | — |
| B184 − Starter | −0.0754665809 | +0.4445883 percentage points | — | — |

Relative CE reductions are 1.65796% and 1.66172%. All three validation configurations contain identical actual batch indices: **1,220 batches, 1,219 of eight samples and a final four-sample batch**, covering 9,756 unique indices. Data identity and implementation hashes also match. Canonical batch-list JSON SHA-256: `90a73f5c689b797121a1de841283fc4c021ff57d0283c8a9cfc0a78c8e54e939`. B184−B183 CE is−0.00017104 and accuracy+0.0132812 percentage points. Two point estimates neither establish seed 184 superiority nor adequately estimate training variance.

All use MPS, float32, packed sequences, batch 8, padding 64. Starter/B183/B184 memory fractions are 0.60/0.75/0.75; validation times 77.948/79.819/78.455 seconds. Memory limits/cache states are not tightly matched, so no validation-speed claim follows.

This is course development data, whose fixed128-batch/1,024-sample subsets also monitor checkpoints; it is not independent final testing. Periodic subsets use training seed+1: B183 subset 184 has 143,650 targets; B184 subset 185 has 146,939. **Periodic curves are not pointwise quality comparisons on the same validation set.** Full 9,756-sample validation uses identical batches despite those subset seeds. Describe the starter's historical data use from upstream evidence rather than assuming it never saw these data. Two training seeds provide limited sensitivity evidence, not stable population-wide gains or generation-quality measurement.

Weight SHA-256 values recorded in execution configurations:

- Starter: `4cb40a7140b1055b7e077232a202f5600ce4a33eb955e8ff8ff1825b5fd54463`.
- B183 latest: `6d1b26bdc679ed32e81884d98f61eb621d574107ea6e24b0c57ffa653735b660`.
- B184 latest: `1529b9ab7dcd7c0bfb33a938247635f6acdd4bd0ec58fef30ae4708efeb4df8b`.

Evidence: [starter validation](../results/starter-full-validation/validation.json)/configuration (local only: `../results/starter-full-validation/config.json`), [B183 validation](../results/B-full-validation/validation.json)/configuration (local only: `../results/B-full-validation/config.json`), [B184 validation](../results/B184-full-validation/validation.json)/configuration (local only: `../results/B184-full-validation/config.json`). The lead agent independently streamed the 2,560,137,483-byte B184 weight file and matched its hash to validation configuration. This document records that completed check without reloading/rehashing the large checkpoint.

## 3. Effective Compute, Recovery Cost, and Wall-Time Scope

The following **seed 183-only** accounting independently reconstructs raw logs chronologically. When step numbers roll back, old-branch rows at or beyond that step are removed from effective exposure and counted separately as cost. Of 47,530 training rows, 44,348 consecutive steps remain and 3,182 completed rows were discarded. Recovery probes are summed from their separate log.

| Scope | Completed updates | Sample presentations | Targets | Compute seconds | Within-step total seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final retained branch | 44,348 | 354,780 | 50,450,231 | 11,949.369 | 12,726.450 |
| Completed updates discarded by two rollbacks | 3,182 | 25,456 | 3,681,000 | 853.096 | 899.695 |
| Memory recovery probes | 18 | 144 | 25,568 | 7.772 | 8.311 |
| Total known completed work | 47,548 | 380,380 | 54,156,799 | 12,810.237 | 13,634.455 |

Effective compute/within-step totals are 3.3193/3.5351 hours; including discarded updates/probes gives 3.5584/3.7873 hours. Total sample presentations are not unique samples, and total attempted targets are not retained model exposure.

`compute_seconds` and `total_seconds` have different timer scopes and must not be added. Retained-branch `empty_cache` time 680.120 seconds is already included in total; all known completed cache time 720.575 seconds is also not additive. The 18 probe updates comprise 1 warmup, 8 first visits, 8 steady iterations, and 1 worst-case update; none enters final B weights. The benchmark's 8 steady timings are not total probe cost.

These are **not whole-project wall times**. They do not fully include partial failed steps, loading, saving, validation, outside-step logging, recovery waits, or other project experiments. Summary `wall_seconds_this_run=4092.648` covers only the last resumed session's post-initialization loop/final save, not all of B. No complete wall-time, power, or tariff measurement exists, so total energy and monetary cost are not reported.

The earlier recovery-cost record (local only: `../results/analysis/B-recovery-costs.md`) was frozen before final recovery finished and excludes the then-active resume2. This chapter recomputes the retained branch from complete logs without modifying the old record. Exact values and source hashes are in training-report-evidence.json (local only: `../results/analysis/training-report-evidence.json`).

Seed 184 had no log rollback:

| Seed 184 timer scope | Seconds |
| --- | ---: |
| Synchronized compute for 44,348 updates | 10,751.354 |
| Within-step total | 11,089.577 |
| Cache clearing, already in total | 247.432 |
| 22 periodic validations | 175.500 |
| Post-initialization loop through final save | 13,733.855 |

`wall_seconds_this_run=13733.855238` is 3.814960 hours, from `training.py`'s post-initialization `started_run` through training, logging, periodic validation,periodic/final checkpoint saves, ending before summary writing. It **excludes initialization and subsequent full-development validation/generation**. Subtracting step totals and recorded periodic validation leaves 2,468.778 seconds of unsplit saving, logging, timer-boundary, and other overhead—not exact saving time.

Compute/step-total definitions match across seeds, but summary wall scopes differ: 183 covers its last recovery; 184 covers the uninterrupted full run. Do not compare 4,092.648 versus 13,733.855 seconds as total training speed. Seed 184 starts with memory fraction 0.75 and saving every 500 steps/10 minutes; 183 changed 0.60 → 0.75 after earlier sparser saving and two rollbacks. Cache state/rework differ, so lower 184 step time is not a controlled acceleration result. See 184 cost audit (local only: `../results/analysis/B184-training-log-audit.md`).

## 4. Two OOMs and Resume Reproducibility

The original run and first retry both completed step 35,591 then hit MPS OOM on the next step: first during attention-mask merging, then during cross-entropy. Each resumed from 34,000 and discarded 1,591 completed updates. Original memory fraction was 0.60. After representative/worst-case probes, final recovery used 0.75 and more frequent saving and reached the full endpoint.

This changed the runtime memory allowance, not batch size, data budget, or objective. Observations do not prove caching as the sole cause. Recovery saves/restores optimizer, scheduler, sampler, RNG, and semantic counters. The first 262 replayed updates showed loss, accuracy, and gradient differences: MPS recovery was not bitwise identical to uninterrupted optimization. Final once-per-sample coverage must be reported separately from actual repeated hardware computation.

## 5. Memory: Recorded Snapshots, Not Peaks

For completed steps on seed 183's retained branch, GiB means `2**30` bytes:

| Metric | Maximum observed before empty_cache | Maximum observed after empty_cache |
| --- | ---: | ---: |
| Process RSS | 9.3791 GiB | 9.3791 GiB |
| MPS allocated | 3.9479 GiB | 3.9479 GiB |
| MPS driver | 28.1747 GiB | 20.7547 GiB |

Maxima may occur on different steps and cannot be subtracted as one-step released memory. Within-step forward/backward transients and failed steps are excluded. Minimum observed system-available memory was approximately 7.9602 GiB before clearing; observed host swap ranged 30.5173–33.8354 GiB. These are whole-machine metrics, not all attributable to B. Without continuous sampling, do not call them peak training memory.

Seed 184 snapshot maxima were RSS 8.8921 GiB, MPS allocated 3.6634 GiB, and driver 29.9891/22.1316 GiB before/after clearing. Minimum available memory was 8.0866 GiB; first-to-last host swap changed−0.7278 GiB. These are also not continuous peaks; first recording follows initialization, and host swap changes are not wholly attributable to training. Full ranges/step locations are in the 184 audit JSON.

## 6. Plots, Attribution, and Remaining Evidence Boundaries

Frozen B-final-training-plots-v2 (local only: `../results/analysis/B-final-training-plots-v2/`) contains [training metrics](../results/analysis/B-final-training-plots-v2/training-metrics.png), [optimization/batches](../results/analysis/B-final-training-plots-v2/optimization-and-batches.png), [observed memory](../results/analysis/B-final-training-plots-v2/observed-memory.png), plus PDFs, CSVs, input snapshots, manifest, validation records, and reproduction script. The x-axis is supervised targets with 500,000-target rolling windows. Its 22 validation points use 1,024-sample subsets, not full validation. Two rollbacks share a recovery boundary near 38.601729 million targets; that coordinate is not wall time.

Separate B184-final-training-plots (local only: `../results/analysis/B184-final-training-plots/`) contains [training metrics](../results/analysis/B184-final-training-plots/training-metrics.png), [optimization/batches](../results/analysis/B184-final-training-plots/optimization-and-batches.png), and [observed memory](../results/analysis/B184-final-training-plots/observed-memory.png), leaving 183 plots unchanged. All three PNG/PDF pairs were visually checked; four frozen input hashes match logs/audit sources and CSV count 44,348 passed. Its 22 development points use subset 185, not 183's subset 184. Plots/lightweight audits retain historical “full exposure audit pending” notes; subsequent completed 184 sampler/checkpoint verification is documented separately rather than rewriting frozen artifacts.

The course supplied architecture, data, and starter weights at commit `48e1c407dec5a39df1a50a9069934879f175c373`; CosyVoice supplies frozen tokenization/acoustic generation. Project contributions include local MPS continuation,correctness/resume checks,batch/padding management,logs/rollback-cost auditing, fixed target-budget endpoints, matched full-development validation, and auditable plots. Attribute engineering and training separately; upstream architecture/frozen acoustic models were not independently trained here.

B is an ordinary-continuation control; this chapter does not establish a novel training method's superiority over it. Both seeds' training, full validation, and sampler/checkpoint audits are complete. Generation reliability requires separate actual comparisons with identical reference, settings, texts, paired seeds, and failure denominators. It cannot be inferred from teacher-forced curves. The independent final set remained sealed when this update was written.
