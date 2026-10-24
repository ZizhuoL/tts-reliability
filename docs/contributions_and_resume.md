# Attribution, Contributions, and Resume Statements

This project is an independent extension of course Part C. The distinctions below separate upstream assets from completed work; results come from the completed independent final evaluation after methods were frozen. The complete CosyVoice2 system was not trained from scratch, and MOS or human-perceived audio quality was not measured.

## Attribution

| Component | Source and project contribution |
| --- | --- |
| Causal Transformer architecture and starter | The course supplied architecture specifications and weights trained for two epochs. This project implemented missing model/data paths and loaded weights strictly; pretraining is not claimed as an individual contribution. |
| Paired text/speech training data | Course-released LibriTTS token cache. This project validated all pairs, recomputed length and speaker statistics, and cached tokenizer results; it did not collect the recordings. |
| Frozen speech components | CosyVoice2 speech tokenizer, speaker encoder, flow, and vocoder, plus course loading helpers. This project validated invocation boundaries and local device paths; these components were not trained. |
| Automatic transcription | Pinned Whisper large-v3-turbo. This project implemented consistent normalization, word alignment, S/D/I counts, failure retention, and paired analysis. ASR errors are not human-confirmed audio defects. |
| Model correctness and performance | Input/target shift, EOS, causal/padding masks, gradient updates, and save/resume checks; added KV caching and validated CPU and actual-starter MPS numerical/generation consistency. |
| Local training | Explicit MPS checks, throughput measurement on realistic lengths, packed sequence preparation, padding grids/cache management, token-budget scheduling, resumable optimizer/RNG/sampler state, and actual exposure auditing. Memory observations for a combined configuration are not an isolated ablation of one technique. |
| Experimental design | Natural texts with documented sources, training-text overlap checks, development/final separation, multiple generation seeds, fixed references, and checkpoint/configuration hashes. Final data did not inform selection before freezing. |
| Research diagnostics | Identified and controlled reference-transcript format sensitivity, separated hard caps from EOS, and deferred length weighting after the complete formal baseline lacked aggregate support, while retaining local unaligned-ending candidates. |
| Evaluation and delivery tooling | Raw event/audio records, unified offline rescoring, text-cluster bootstrap, failure profiles, training-exposure analysis, and listening collections with selection rules and hashes. |

See the [README](../README.md), [research notes](research_notes.md), and data/experiment manifests for full versions and provenance. The course checkout remains unchanged; project scripts and results live separately.

## Supported English Resume Bullets

These statements describe completed implementation, two independent continuation runs, and the frozen final primary comparison. Each is independently auditable.

- Implemented and validated a checkpoint-compatible 213M-parameter autoregressive TTS pipeline on Apple Silicon, including causal masking, speech-target alignment, EOS handling, and a KV cache checked against uncached generation, while reusing pretrained course weights and frozen CosyVoice2 acoustic components.
- Completed two independent full-epoch continuation runs from the same starter, each covering 354,780 text–speech pairs and 50.45M supervised targets; built resumable training with optimizer/RNG/sampler state and audited coverage, discarded updates, and memory failures.
- Built a reproducible nine-system TTS evaluation with 1,080 scored outputs; on 60 held-out texts × two generation seeds, the frozen seed-183 combination of continuation training and removal of LM reference prompting reduced ASR WER from 25.40% to 9.48% (paired text-cluster bootstrap 95% CI for the change: −18.45 to −13.05 percentage points).

The third bullet's primary comparison, A → BNoLM, changes both continuation weights and LM reference prompting. The full difference is not a training-only gain. NoLM removes only the LM reference text/speech-token prefix, retaining reference-speaker conditioning in the frozen acoustic path. The long group contains 20 independent texts × 2 seeds: WER 27.63% → 9.39%, change −18.24 percentage points, 95% CI [−22.10, −14.19]. Intervals resample text clusters within length groups while retaining paired generation seeds; they exclude training randomness, ASR bias, and generalization beyond these sources.

Training seed 183 was fixed as primary during development; seed 184 is a sensitivity check, not a replacement selected using final results. 184NoLM overall WER was 17.04%; 213 of its 279 insertions came from one medium-text/seed42 transcription anomaly. Without human listening, acoustic repetition and ASR looping cannot be distinguished. The 9.48% result for seed 183 alone does not establish stable superiority across training seeds. Each run independently added one epoch from the same starter; they are not two additional epochs for one model.

Evidence: [final summary](../results/final-evaluation-v1/analysis/overview/summary.json), [frozen comparison roles](../results/final-evaluation-v1/comparison-roles.json), [training and recovery report](training_results.md), and independent artifact verification (local only: `../results/analysis/final-artifact-review.md`).

## Defining the Work Accurately in Interviews

A supported description is: “I reused the course's pretrained language model and frozen acoustic components, completed the missing implementation, correctness checks, resumable local training, and independent evaluation, then used real development results to freeze the research direction before running an independent final test.”

The initial question concerned long-text completeness, but the complete baseline after correcting reference formatting did not support stable aggregate length degradation. Local failures still warrant analysis. Experiments must examine overall performance, short texts, stopping reasons, and specific alignments rather than selecting bad examples to fit the original question.

For compute, distinguish measured representative-batch estimates, completed training logs, and final elapsed time. For quality, distinguish ASR content consistency from naturalness and speaker similarity. Current automatic evidence covers only content consistency and operational completion metrics.

seed 183 hit OOM twice at batch 35,592. Recovery succeeded, but two discarded replay tails—3,182 updates, 25,456 sample presentations, and 3,681,000 supervised targets—count toward attempted cost, not the final model's effective budget. Memory-threshold and checkpoint-frequency changes are differences in runtime history, not independently proven single-technique OOM solutions. Listening-package hashes and links were checked; no human listening or MOS evaluation was performed.
