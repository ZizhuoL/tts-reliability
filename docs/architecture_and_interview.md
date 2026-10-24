# Architecture and Interview Guide

This guide explains implemented mechanisms and interprets the completed, frozen final evaluation with its limitations.

## Accurate Project Scope

The project extends a course-provided causal Transformer with approximately 213 million parameters and a starter trained for two epochs. It completes training and generation logic, adapts execution to local Apple Silicon, and studies spoken-content completeness and generation reliability. Training data pairs text with discrete speech tokens. The CosyVoice2 speech tokenizer, speaker encoder, flow decoder, and vocoder remain frozen. This is not training the complete CosyVoice2 system from scratch.

Qwen2 BPE first encodes input text. The LM maps text, special markers, and historical speech tokens to discrete speech tokens. The frozen flow converts tokens and reference-speaker information to mel features; the frozen vocoder converts mel features to a 24 kHz waveform. Independent Whisper ASR transcribes the waveform, after which WER is calculated with consistent normalization.

## Autoregression, Teacher Forcing, and the Objective

The model predicts one speech token at a time. Training uses preceding ground-truth speech tokens, whereas inference uses previously generated tokens. This difference can accumulate errors, but does not establish that the model must fail on long inputs.

For text `[t0,t1]` and speech `[s0,s1,s2]`:

```text
Input   [SOS, t0, t1, TASK, s0, s1, s2]
Target  [ -1, -1, -1,   s0, s1, s2, EOS]
```

The TASK position predicts the first speech token; the final ground-truth speech input predicts EOS. `-1` excludes a position from supervised loss. Speech EOS is ID 6561. SOS=0 and TASK=1 belong to a separate embedding table and must not be confused with speech IDs.

Training minimizes cross-entropy over all nonignored positions. Validation loss is weighted by valid target tokens so that averaging batches of different lengths does not change the metric's meaning. Lower teacher-forced loss does not guarantee better free generation, which requires actual synthesis and content evaluation.

Let \(V\) be the valid supervised positions in a batch, \(y_j\) the ground-truth speech token at position \(j\), and \(p_\theta(\cdot\mid h_j)\) the model distribution. Then:

\[
L_{\mathrm{batch}}=-\frac{1}{|V|}\sum_{j\in V}\log p_\theta(y_j\mid h_j).
\]

EOS is supervised. Text, unused special-marker positions, and padding are excluded from \(V\). Cross-entropy increases the conditional probability of the true next token, but teacher forcing conditions on the true history. During free generation, an error changes subsequent history and may change EOS timing, so these evaluations measure different behavior.

Each update uses the current batch's mean loss over valid tokens; full validation sums losses and valid-token counts across all batches before dividing. Token-weighted validation does not imply identical training trajectories under different batch partitions: optimizer-update order, batch size, and accumulated state still matter. Token scheduling ties learning-rate progress to cumulative supervised targets, making exposure easier to compare, but equal target budgets do not ensure equal attention FLOPs, runtime, or unique sample coverage.

## Causal and Padding Masks

Attention can access current and past positions, but not future speech inputs. Otherwise it could directly read the next position's true target, leaking supervision. The padding mask blocks padded positions; their targets must also be ignored. These are separate responsibilities.

To limit MPS graph creation for different shapes, training pads sequence lengths to multiples of 64 and groups samples by length. This changes only masked computation positions, preserving valid text/speech pairs, target shifts, and position embeddings. Loss and gradient equivalence tests verify this.

## KV Cache: Purpose and Limits

Naive generation recomputes the entire history at every step. A KV cache stores each layer's previously computed keys and values, so the next step computes queries, keys, and values only for the new position. This is an inference engineering improvement, not a training method.

Checks must cover stepwise logits under identical histories, top-1 tokens, generated-prefix boundaries, and stopping reasons. Small floating-point differences are possible; speed alone does not prove equivalence. The project retains separate CPU and real-starter MPS checks.

Incremental attention still compares each new query with all prior keys, and the cache grows with the sequence. Caching eliminates repeated projections and layer computation for historical positions; it does not make long-sequence decoding independent of history length. Numerical consistency and speed are separate claims, and speed requires measurement.

## Fair Comparisons

First exclude an undersized output cap, incorrect reference boundaries, or scoring normalization errors. Then compare methods from the same checkpoint with optimization settings, learning rate, and effective speech-target token budgets as closely matched as possible. Equal steps do not imply equal computation: long-sample batches often contain more valid tokens and possibly more padding.

Report endpoint supervised tokens, sample coverage, updates, and runtime. If comparing best-validation checkpoints, disclose their respective training exposure rather than hiding budget differences as method gains.

Use development data to identify problems and fix settings; reserve the sealed final set for final evaluation. Generate multiple seeds per text and cluster uncertainty estimates by text. Two seeds for one text are not two independent texts.

For sentence splitting, inspect the subset actually split. Each segment has its own minimum output constraint, output budget, and EOS decision, and concatenation adds fixed pauses. Both content completeness and runtime can change. Summed segment budgets need not equal the whole-text budget, and added pauses change the audio-duration denominator in real-time factors. Preserve these quantities rather than showing WER alone.

## Interpreting Results

WER = `(substitutions + deletions + insertions) / reference words`. Deletions can arise from ASR failures; insertions can reflect number, abbreviation, or punctuation normalization. Interpret word alignments, raw transcripts, EOS/cap/exception records, and actual audio together. Do not label every deletion an omission in speech.

EOS only indicates that generation stopped voluntarily; it does not prove that all text was spoken. Lower WER does not establish naturalness, speaker similarity, or user preference. Those require separate evaluation. Do not claim human listening, MOS, or speaker-similarity tests that were not performed.

## Tradeoffs to Discuss in Interviews

- Reusing the starter directs a limited local budget toward controlled research while clearly attributing pretraining.
- Measurement comes first because initial MPS graph compilation, sequence length, padding, optimizer state, and CPU audio components jointly determine runtime. One short sentence or a default batch size cannot predict a full epoch.
- Sampler and RNG state are saved because model weights alone cannot reproduce the next training batch and random state.
- Sentence splitting remains a baseline because simple engineering can suffice; a lack of additional training benefit must also be reported.
- If a hypothesis fails, report the missing phenomenon, actual findings, limitations, and evidence rather than manufacturing an improvement.

## Answering Follow-up Questions with Project Evidence

**“Why not immediately change the learning rate when validation loss initially worsened?”**

This was fixed-budget continuation from existing weights with a freshly initialized optimizer. Periodic validation worsened before recovering; that trajectory must be preserved, and current evidence does not isolate its cause. Adapting settings to those observations would change the experiment. Interpretation must combine endpoint validation and actual generation. Another optimizer setting should be a separate experiment with explicit configuration and budget.

**“Does saving MPS RNG prove bitwise-identical checkpoint resumption on a Mac?”**

No. The project saves model, optimizer, sampler, and relevant RNG states and strictly validates resume configuration. A short CPU interruption/resume comparison was bitwise identical, and actual MPS checkpoint state completeness was checked. Long training later hit MPS OOM at batch 35,592 and resumed from step 34,000. The first 262 replayed updates matched the old branch's sample/token counts, sampler cursor, and learning rate, but loss, accuracy, and gradient norm differed numerically. This actual recovery was not bitwise reproducible. The check also cannot prove that every state tensor was identical or guarantee equivalence across hardware or versions. See `checkpoints/B-ordinary-seed183/recovery/oom-35592/early-replay-check.json`.

**“What do the final results support?”**

The frozen primary comparison A → BNoLM uses training seed 183, 60 independent final texts, and generation seeds17/42 per text. Overall ASR micro-WER fell from 25.40% to 9.48%; the long group, 20 texts × 2 seeds, fell from 27.63% to 9.39%. The overall change was −15.92 percentage points, with a length-stratified paired text-cluster bootstrap 95% CI of [−18.45, −13.05]. The long-group change was −18.24, CI [−22.10, −14.19]. This comparison combines continuation training with removal of the LM reference prefix. Acoustic reference conditioning remains; neither the entire gain nor reference-free voice generation can be attributed to training/NoLM alone.

Exploratory comparisons were 25.40% → 17.71% for A → B with the same prompt format, 14.69% → 9.48% for ANoLM → BNoLM with the same NoLM format, and 17.71% → 9.48% for B → BNoLM with the same seed 183 weights. They help separate factors but do not replace the frozen primary comparison and have no multiple-comparison correction.

Training seed 184 was an independent sensitivity run and did not replace seed 183. B184NoLM overall WER was 17.04%, versus 17.01% for B184 standard; the overall NoLM advantage did not replicate. Of 184NoLM's 279 insertions, one medium text, `alice-p00741-s00-n2` with generation seed 42, contributed 213. Its output reached the 500-token cap and 20 seconds, and ASR produced repeated “a” tokens. This was not truncation from crossing the 30-second transcription window. Without human listening, the contributions of acoustic repetition and ASR looping remain undetermined. Removing this example cannot justify a stability claim.

All 1,080 generations across nine systems completed and were scored, including capped outputs. A and BNoLM each had 1/120 `output_limit` cases; B had 5/120. Zero exceptions does not mean zero truncations or content errors. Two training seeds, a fixed reference voice, limited natural-text sources, and one ASR scorer do not support broad claims about speaker/domain generalization or subjective naturalness. Intervals capture conditional uncertainty over the current text clusters and observed generation seeds, excluding training randomness and ASR model error.

## Explaining Full Training and Failure Costs

seed 183 and seed 184 each independently continued for one epoch from the same two-epoch course starter: 44,348 effective updates, all 354,780 pairs exactly once, 50,450,231 supervised targets including EOS, and zero budget overshoot. Independent log/sampler/checkpoint exposure audits are complete. These are not two sequential extra epochs for one model or two independent training corpora. Both select full-epoch `latest.pt` endpoints. Full course validation used the same 9,756 samples, with CE 4.466166 and 4.465995, versus starter CE 4.541461. Periodic 1,024-sample subsets use different selection seeds, so their curves are not pointwise comparisons on identical data.

seed 183 hit OOM twice at batch 35,592 and eventually completed after full-checkpoint recovery. Two discarded tails totaled 3,182 updates, 25,456 sample presentations, and 3,681,000 targets; another 18 temporary memory-probe updates also did not enter the final model. Equal effective budgets neither erase this cost nor guarantee bitwise MPS replay. The memory fraction rose from 0.6 to 0.75 and saving changed to every 500 steps/10 minutes; seed 184 used that runtime configuration from the start. Accumulated graph caching is a candidate explanation, not an independently proven sole cause.

For seed 184, `wall_seconds_this_run=13733.855` runs from post-initialization `started_run` through final saving, ending before summary writing. It includes training-time validation and saving, not the whole training/generation pipeline. The same field for seed 183 covers only its final recovery session. Compare explicitly scoped step compute/iteration time, validation, and failed work rather than these two wall fields directly. Observed memory maxima are not continuously monitored system peaks.

Sources: [full training report](training_results.md), [final system summaries](../results/final-evaluation-v1/analysis/overview/summary.json), [NoLM training-seed sensitivity](../results/final-evaluation-v1/analysis/seed-sensitivity-no-lm/summary.json), and delivery plots and reproduction (local only: `../results/analysis/final-delivery-plots-v2/REPRODUCE.md`). The listening collection was checked for files, hashes, selection rules, and links; no human listening or MOS evaluation was performed.
