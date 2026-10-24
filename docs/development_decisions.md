# Development Decisions and Evidence

This record uses development data and component checks only. Final data had not informed method, generation-setting, or checkpoint selection at these decision points. Observations are ASR-alignment results, not human listening judgments.

## Fixing Reasonable Reference-Transcript Formatting

The fixed LibriSpeech reference originally had an uppercase, unpunctuated transcript. Whisper's normalized WER on the real recording was 0; a check sample decoded from real course speech tokens through frozen acoustic components also had WER 0. Generator-boundary checks confirmed that returned tokens exclude the reference speech prefix, and the acoustic components remove reference mel frames. Thus initial high errors cannot simply be attributed to reference audio being directly concatenated into outputs.

A paired diagnostic used the first three predetermined development texts in each length group: 9 texts × 2 generation seeds. Only LM reference conditioning changed; other settings and the acoustic reference were fixed.

| LM reference condition | Overall WER | Short | Medium | Long | Output cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original uppercase, no punctuation | 65.58% | 84.78% | 58.20% | 65.65% | 0/18 |
| Same words, sentence case and punctuation | 30.95% | 21.74% | 9.84% | 41.16% | 0/18 |
| No LM reference prefix; acoustic reference retained | 24.46% | 17.39% | 11.48% | 30.95% | 1/18 |

Source: `results/prompt-diagnosis/comparison.json`; original events/audio remain in each condition directory. Unified normalization and paired statistics are in `results/analysis/prompt-diagnosis-complete/`. Sentence case versus original formatting changed overall WER by −34.63 percentage points; the 2,000-replicate length-stratified text-cluster bootstrap 95% interval was [−45.27, −24.58]. With only nine text clusters, this interval excludes ASR bias, training randomness, and uncertainty from changing the reference speaker.

The earlier `results/analysis/prompt-diagnosis/` contains partial artifacts: CSV export failed because system name `method` collided with the bootstrap-method description field. After repair, analysis reran fully into a new directory, passed a three-system paired CLI regression and all export checks, and wrote `completion.json`. Diagnostic audio was neither regenerated nor changed to alter metrics.

**Decision:** Formal A/B/C comparisons use sentence-case `configs/reference_prompt.json`. This changes neither words nor recording; it repairs the baseline protocol and is not a training contribution. No-LM-prefix remains an engineering candidate; nine texts do not establish broad superiority. The 90 original-format development outputs remain diagnostic evidence, not the denominator for primary training-gain claims.

## Full Ordinary Continuation

The valid training set has 354,780 complete pairs and 50,450,231 supervised speech targets including EOS per epoch. Representative real-batch measurements with batch=8, a 64-position padding grid, packed sequence preparation, and per-step unused MPS-cache clearing estimated approximately 3.55 hours of core training, excluding periodic validation, saving, and full validation. The longest batch passed finite-gradient and parameter-update checks.

On identical batches, packed and original training had maximum absolute loss difference approximately 9.54e−7 and identical accuracy. No substantial speed increase was observed; adoption was motivated by memory stability. Measurements combine packing, cache clearing, and the MPS memory threshold rather than isolating packing. Instantaneous driver allocation is not the full-epoch peak.

**Decision:** B starts from the same course starter and trains one full valid-data epoch with seed=183. The primary comparison uses the budget endpoint; a course-dev-loss-selected best checkpoint is supplementary and must disclose its budget. Actual exposure, time, and memory come from logs, not throughput estimates.

Long training hit the MPS limit twice at step 35592; a successful short benchmark does not prove long-run stability. The second recovery increased the bounded memory allowance and shortened save intervals, passed the offending batch, and completed the epoch without changing the core algorithm or data order. `results/analysis/B-final-exposure.json` independently reconstructed all 44348 steps: all 354780 unique samples exactly once, 50450231 targets, matching log fields and final checkpoint sampler/counters/seen_indices, with zero overshoot. Two old tails totaling 3182 updates were excluded from effective exposure. Failure costs are in `results/analysis/B-recovery-costs.md`. Recovery is not a new training-seed experiment; MPS numerical replay was not bitwise identical.

The accepted trajectory accumulated 11949.37 seconds of synchronized compute and 12726.45 seconds within steps. Neither is project wall time, and step time already includes cache clearing, so they must not be added. Full course validation on 9756 samples/1362834 targets yielded cross-entropy 4.46616567, accuracy 0.11618290, in 79.82 seconds. These are endpoint teacher-forced metrics, not independent-test or generation-quality conclusions. `latest` is step 44348; fixed-subset validation selected `best` at 44000. Primary comparisons retain `latest`.

Starter full validation subsequently yielded cross-entropy 4.54146120 and accuracy 0.11186982 on the same 9756 samples/1362834 targets. Batch, padding, dtype, core source hashes, and complete validation configuration including batches matched. Inference process memory fractions were starter 0.6/B0.75. The 0.07530 CE decrease supports better teacher-forced prediction only, not WER or audio quality. Formal full-training plots are in `results/analysis/B-final-training-plots-v2/`, retaining recovery markers and observed memory ranges.

## Criteria for the Candidate Length-Sampling Route

In the small corrected-format diagnostic, long-group deletion rate was 8.16%, versus 0 for short/medium, but only three long texts were available. The complete 45-text formal development baseline changed this route's priority, as described below.

If full-development evidence supports it, candidate C gives twice the sampling weight to the top quartile of training speech lengths while retaining nonzero probability for every short sample. It targets exposure to longer autoregressive trajectories, without presuming it solves alignment or early stopping. A small real-update pilot must precede a full-budget run. C would start from B's starter with the same optimizer, learning-rate schedule, and 50,450,231 target-token budget, recording final-full-batch overshoot, actual coverage, steps, and padding.

Tradeoffs include reduced short-sample exposure, reduced unique coverage from resampling, and different attention costs even at equal target counts. Report short and overall results together. The sentence-synthesize-concatenate baseline must also actually run; report if it works better.

Budget simulations using actual lengths and the same deterministic sampler are in `results/analysis/exposure-plans/`; both `*-v2.json` files explicitly say **PLAN_ONLY_NOT_EXECUTED**. B's plan has 44,348 updates and 100% unique coverage. Candidate C has 36,260 updates, 54.13% coverage, 50,451,393 targets, and final-batch overshoot 1,162. Samples with speech≥191 account for 25.206% of training due to ties at the boundary; their expected target share rises from 53.11% to 69.40%. These are sampling plans, not completed training or quality evidence.

Sentence-rule checks found18/45 development texts split into multiple segments: short4/15, medium3/15, long11/15. Report both all texts and the actually split subset so unchanged texts do not dilute effects. This check ran text rules only, without synthesis or reading final texts.

## Route Selection After the Complete Formal Baseline

All 90 formal A outputs were scored and ended in EOS, with no output/context truncations. Short/medium/long WER was 26.77%/22.12%/27.52%, deletion rates 6.30%/4.85%/7.40%, overall WER 25.91%. There was no stable aggregate degradation specific to long texts. Two long texts still had consecutive unaligned tail words, retained as candidate failures, without human confirmation of omissions or early endings.

**Decision: defer C; do not force length weighting based on a weak original prompt format or a small subset.** Complete ordinary full-epoch B, then compare standard LM prompting, no-LM-prefix, and sentence splitting for both A and B on the same development data. This 2 × 3 comparison separates continuation and engineering treatments. No-LM-prefix retains identical acoustic conditioning. Report single-factor and combined effects separately; A-standard → B-no-prefix is not entirely a training gain.

`scripts/run_development_comparisons.sh` started after B's full epoch and complete course-dev validation, first running matching starter validation and then five additional development conditions. Starting the script did not establish completion. Freeze the final protocol after selection; final data did not inform this decision. C may remain unrun if B or engineering treatments solve the local issue without additional evidence for C. See [formal baseline review](standard_baseline_review.md).

## Selection Rules Fixed Before Additional Development Evaluation

`configs/development_selection_rule.json` was written before the five new synthesis runs. Each system must complete all 45 texts × 2 seeds. Candidates must lower overall WER versus formal A, not increase short/long WER, and not increase cap-stop or exception rates. If scored coverage differs, report common-success pairs and whole-reference-deletion sensitivity so exclusions cannot manufacture a recommendation. Eligible candidates rank by overall WER, generation time, then configuration complexity. If none qualify, retain A provisionally and report tradeoffs/negative results.

These are conservative development point-estimate filters, not statistical noninferiority tests or guarantees of preserved short-text quality. Report paired text-cluster intervals and per-generation-seed results. This file itself does not freeze the final protocol: selected method, checkpoint hash, and primary comparison must be recorded after development completes and before opening final data. Final results cannot select a new method.

### First Additional Condition: A Without LM Prefix

`results/starter-no-lm-dev/` completed90/90 scores, covering 45 texts × two seeds, with all WAV hashes verified and 0 exceptions. Overall WER was409/2320=17.6293%, versus A-standard 25.9052%. Short/medium/long WER was 11.8110%/17.1212%/18.9189%, overall deletion rate 3.4483%, and generation-seed WER 16.8103%/18.4483%.

It also had1/90 cap stop (long1/30): `holmes-p00974-s01-n2`, seed 42,737 tokens, 29.48 seconds; A-standard had none. It therefore **fails the predefined no-increase-in-cap-rate default-selection rule**, despite lower overall WER. These are complete-condition point estimates and operational tradeoffs; six-system paired analysis/intervals awaited remaining conditions, and no final method was selected. The longest audio was 29.48 seconds, below the configured 30-second ASR window; ASR measurement bias remains.

## Split Subsets and Cost Accounting

`scripts/analyze_strategy_costs.py` derives splitting status from actual output records. Only completely synthesized audio with multiple segments enters the actually split subset; partial segments from interrupted synthesis are unknown. Completed synthesis with failed ASR still counts as an attempt, not zero error. Overall/split/unsplit comparisons pair common successes and report per-seed results and missing counts. Main analysis separately retains whole-reference-deletion sensitivity for missing scores.

Cost tables separate actual tokens, summed configured segment budgets, segment count, generation time, and audio duration. Added pauses are computed from actual boundary counts, saved configuration, sample rate, and implementation sample rounding, not silence detection. Real-time factors use both original duration and duration minus configured added pauses, preventing inserted silence from making speed appear better. Summed segment budgets can exceed whole-text budgets substantially; this is not strictly matched generation exposure.

The tool ran on all 90 existing formal A outputs in `results/analysis/starter-standard-strategy-costs/`: 90 complete segments, 21,390 actual speech tokens, 48,370 configured budget tokens, 0 added pauses, 417.15 seconds generation, 855.60 seconds audio. Generation RTF was approximately 0.488, excluding ASR, model loading, and training. A-split's completed audit follows; B comparisons were still running at that point.

### A-Split Results and Actual-Split Audit

`results/starter-split-dev/` completed90/90 scores with 0 exceptions. Overall WER492/2320=21.2069%; short/medium/long29.1339%/22.5758%/19.1323%; generation-seed WER 21.4655%/20.9483%. Short WER exceeded A-standard 26.7717%, and1/90 output hit its cap, so this condition also failed the predefined filter. In `alice-p00123-s04-n2`, seed 17, the second segment reached 500 tokens. The entire output remains scored; splitting did not eliminate truncation.

The independent `results/analysis/starter-split-audit/` checked hashes/durations for 180 disk WAVs; the lead agent additionally verified six raw/manifest/summary source hashes. All 54 actually unsplit outputs matched pairwise in segment text, tokens, stops, WAV SHA, and normalized ASR. The 36 split outputs from 18 texts improved WER from382/1258=30.3657% to273/1258=21.7011%. These remained development point estimates; the full pipeline separately computed paired intervals.

Splitting added 46 boundaries and 6.9 seconds of configured pauses. Across 90 outputs, summed caps rose 48370 → 68720, actual tokens 21390 → 21651, generation time 417.15 → 439.85 seconds, and audio 855.60 → 872.94 seconds. Generation time on the split subset rose 11.22%. Configured budgets are not actual generated quantities; this is neither equal-total-cap nor punctuation-only ablation. Neither group exceeded 30 seconds; ASplit's longest output was 24.51 seconds.

Short-group S/D/I changed41/16/11 → 37/12/25. Ten of 136 segments fell below the evaluation whole-text lower limit of 6 text tokens, with minimum 3. Training includes 1–5-token examples, so these are not beyond training support. Separating quoted speech from attribution is a possible mechanism, but alignments do not prove that short segments cause insertions or that audio repeats. Detailed ASR/alignment examples remain in the audit; human evaluation was not performed.

### B-Standard Development Results

`results/B-standard-dev/` completed90/90 scores with 0 exceptions; the lead agent verified 45 × 2 coverage, run_id, and every disk WAV hash. Overall WER444/2320=19.1379%, S/D/I232/117/95; short/medium/long13.3858%/16.2121%/21.5505%; generation-seed WER 19.5690%/18.7069%. All length-group point estimates improved over A-standard, but 88 EOS and 2 `output_limit` cases meant it **failed the predefined filter** because cap rates increased. Complete paired intervals awaited six-system export; these are not final-test or stable-generalization claims.

Both caps were long texts: `alice-p00467-s00-n2`, seed 42,781 tokens/31.24 seconds; and `alice-p00032-s01-n1`, seed 17,638 tokens/25.52 seconds. The first was the only output exceeding Whisper's 30-second chunk window, requiring an explicit long-audio limitation. Do not alter scoring or remove it to improve the main result. Summed synthesis was 525.28 seconds, audio 921.44 seconds, ASR 327.93 seconds; interpret costs alongside actual output lengths and resource observations.

### B-NoLM Development Results

`results/B-no-lm-dev/` completed90/90 scores, all EOS, with no exceptions. The lead agent verified 45 × 2 coverage, run_id, and 90 disk WAV hashes. Overall WER236/2320=10.1724%, S/D/I135/46/55; short/medium/long7.4803%/10.1515%/10.6686%; seed17/42 WER 9.9138%/10.4310%. All point-estimate filters versus A-standard passed. B-split/full paired analyses were unfinished then, so this established eligibility, not final selection or independent generalization.

This condition combines ordinary training and LM-prefix removal. Compare A-standard → B-standard, ANoLM → BNoLM, and B-standard → BNoLM separately; do not attribute the entire 25.9052% → 10.1724% A-standard/BNoLM change to training. Acoustic references remain identical; naturalness and speaker similarity were not measured. Synthesis totaled 387.35 seconds, audio 827.64 seconds, ASR 318.59 seconds. Maximum duration was 23.04 seconds, with none above 30 seconds.
