# Formal Baseline Review: 90/90 Completed

The results **do not yet support stable aggregate failure specific to long texts and do not justify prioritizing length-weighted C**. Stronger next steps are full-development LM reference-prefix ablation, sentence splitting, and ordinary continuation B. A few long-text ending-omission candidates remain worth tracking; absence of an aggregate trend does not mean absence of problems.

This read-only review used all **45 development texts × seeds 17/42 = 90 complete records** in `results/starter-standard-dev/raw.jsonl`: 15 independent texts and 30 samples per length group. It did not read final data, run GPU work, or perform human listening. Inputs used fixed reference audio, sentence-case reference text, v2 unified normalization, and adaptive new-speech budget `max(500, ceil(11 × target_text_tokens))`. All 18 overlapping diagnostic/formal-baseline ASR transcripts match exactly, allowing direct subset/full-set reconciliation.

| Group | Completed | WER | Deletion rate | Insertion rate | Substitution rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 30 | 26.77% | 6.30% | 4.33% | 16.14% |
| Medium | 30 | 22.12% | 4.85% | 3.33% | 13.94% |
| Long | 30 | 27.52% | 7.40% | 4.05% | 16.07% |
| All | 90 | 25.91% | 6.55% | 3.88% | 15.47% |

Every output ended in EOS; hard output caps, context limits, and exceptions were all zero. Long-text WER exceeded short-text WER by **0.75 percentage points**, and deletion rate by **1.10 points**. Medium texts performed better, but degradation was not monotonic. Overall WER for the two seeds was 25.95% and 25.86%; similar totals do not imply item-level stability: short-group seed WERs were 30.71%/22.83%, and medium-group WERs 19.09%/25.15%.

An exploratory uncertainty check resampled texts with replacement within the 15 long and 15 short texts, retaining both seeds per text and recalculating word-weighted WER over 10,000 replicates (NumPy RNG seed=20261020). The percentile 95% interval for long-minus-short WER was approximately **[−13.10, +14.53] percentage points**. This small-development-set interval neither proves equivalence nor excludes smaller length effects. Two seeds are not 30 independent texts.

Item alignments provide more specific clues:

- Of 104 long-group deletions, 32 were alignment-tail deletions, versus 2/16 for short and 0/32 for medium. At least three consecutive tail words were deleted in 3/30 long outputs and 0/30 short/medium outputs, concentrated in two texts—insufficient to establish an aggregate mechanism.
- `alice-p00032-s01-n1`, seed 17: ASR ended after “under the door,” leaving 16 consecutive normalized reference-tail words unaligned; 58 text tokens, 281 new speech tokens, 11.24 seconds, EOS, budget 638. Seed 42 mainly showed later confusion/repetition and three tail deletions. EOS means voluntary stopping, not complete reading.
- `alice-p00123-s04-n2`, seed 17: the final 12 words, “at least there is no room to grow up any more here,” were unaligned; 53 text tokens, 281 new speech tokens, 11.24 seconds, EOS, budget 583. This is a candidate early-ending case, not hard-cap truncation.
- Repetition/content errors also occurred in shorter groups: ASR repeated “in it for example” for short `holmes-p01618-s00-n1`, seed 17, and “so he did” for medium `alice-p00579-s00-n1`, seed 42. Insertions do not directly establish repetition in audio.
- Median new-speech/text-token ratios were 7.15, 7.67, and 7.46 by group, without pronounced aggregate shortening with text length. Speaking rate, punctuation, and ASR errors make this only an auxiliary signal.

These cases involve quotation, dialogue, multiple clauses, or repetitive constructions, confounding content difficulty with length. Two-sentence examples performed worse than single-sentence examples in all groups, but the short group contained only one two-sentence text; this cannot establish causality or support a new significance claim. Sentence splitting is useful, but the current period-based splitter does not divide semicolons or coordination within one sentence. In particular, do not claim it solves `alice-p00032-s01-n1`.

Reference-format diagnostics on the same 9 development texts × two seeds, with 462 reference words per condition:

| LM reference condition | WER | Deletion rate | Insertion rate | Non-EOS |
| --- | ---: | ---: | ---: | ---: |
| Original uppercase transcript | 65.58% | 14.07% | 27.49% | 0/18 |
| Sentence case + punctuation | 30.95% | 5.19% | 5.19% | 0/18 |
| No LM reference prefix; acoustic reference unchanged | 24.46% | 5.41% | 4.98% | 1/18 |

The uppercase diagnostic primarily reveals reference-format sensitivity and must not serve as a weak baseline for advertising training gains. Removing the LM prefix reduced subset WER by 6.49 percentage points relative to sentence case, but slightly increased deletions and caused one cap stop: `holmes-p00974-s01-n2`, seed 42, reached 737 tokens/29.48 seconds, with ASR repeating “left the country, and finally…”. It is not yet an unconditionally superior method. Retain reasonable budgets and inspect stopping reasons rather than declaring a WER-only winner.

Recommended sequence:

1. Complete **A + no-LM-prefix** on the same 45 texts/two seeds. Fix starter weights, acoustic reference, sampling, and budgets; change only the LM prefix. Compare WER/D/I/S, tail-deletion candidates, non-EOS failures, runtime, and paired item changes. The existing nine-text diagnostic motivates this experiment.
2. Complete **A + sentence split**, fixing formal reference formatting and per-segment budgets. Report concatenation pauses, segments, per-segment stops, total runtime, and audio. Do not select only improved multisentence cases. Any combination with no-LM-prefix must be a separate condition so two changes are not presented as one factor.
3. Retain **B: ordinary-distribution continuation**. Compare with formal starter A and transparently record coverage, effective token budgets, runtime, and checkpoint selection. Use identical synthesis settings for A/B. If feasible, apply selected engineering methods to both to separate training and inference effects. Different A/B training exposure is the experimental treatment, not a fair architecture ablation.
4. **Defer C** unless fuller development analysis shows a stable length/tail-omission association that B and engineering methods leave unresolved. Only then consider a length-sampling branch from B's starting checkpoint with matched effective token budget. The original long-text question alone does not justify C.

Use development data for these decisions, then evaluate final data after methods, prompt format, budgets, and checkpoint are fixed. ASR is not audio ground truth. This review claims no naturalness, voice preservation, human listening, MOS, or confirmed spoken omissions. Retain negative examples for organized later listening.

Evidence: `results/starter-standard-dev/{manifest.json,raw.jsonl,summary.json}` and `results/prompt-diagnosis/{comparison.json,original/raw.jsonl,sentencecase/raw.jsonl,lm-none/raw.jsonl}`.

Formal baseline run_id: `3a74da01588dfbcbfbcfd6a7149b95ac75cd3ee65d370732696d3a426f79fc64`.

Complete raw.jsonl SHA-256 at review time: `0ccaf495510f58ab6b4ad2fba0f90dd74f294a3f90fdc67c61e4dccabd1c019a`.

All three tail-omission candidates were explicitly added to the candidate listening index (local only: `../results/listening/starter-tail-candidates/INDEX.md`), with source/copy hashes matching. It also retains automatically selected low/high-WER examples. Their explicit selection rationale is “ASR candidate with >=3 consecutive unaligned tail words (not human-confirmed).” The directory's existence does not establish completed human listening.
