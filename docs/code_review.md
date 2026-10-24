# Independent Code Correctness Review

Date: 2026-10-19 (America/Los_Angeles). Scope: `src/tts_reliability/model.py`, `data.py`, `training.py`, `evaluation.py`, and `scripts/evaluate.py`. This review read the implementation and existing development results, performed no GPU work, and did not read the sealed final texts. These are findings at review time, not final research conclusions; fixes under parallel development require verification by the lead agent.

## Findings Requiring Action

### P2: Sentence Splitting Artificially Weakens the Engineering Baseline

Location at review time: `evaluation.py:95–98`.

The regular expression splits directly on whitespace after a period, separating `Mr. Hatherley` and `Mr. Holmes` into fragments. CPU calls reproduced this on development samples `holmes-p01748-s00-n3` and `holmes-p02149-s01-n3`. Conversely, `“Come here.” The door opened.` is not split because a closing quotation mark separates the period from whitespace.

Protect common abbreviations, recognize closing quotation marks after terminal punctuation, and add regression tests using real development examples. Do not interpret damage caused by incorrect splitting as an inherent limitation of the method. The lead agent assigned the fix to the evaluation implementation agent; this review did not verify it.

### P2: Resume Compatibility Checks Omit Parameters That Change Training

Location at review time: `training.py:388–404`.

Resume checks compare learning rate, weight decay, and other parameters, but omit `grad_clip`. A user can therefore change the clipping threshold while resuming the same checkpoint without a warning. The subsequent trajectory changes and cannot be claimed equivalent to uninterrupted training.

Include `grad_clip`; also check `device` if exact same-device resumption is required. If parameter changes are allowed, record them as a new training phase rather than exact resumption. This finding does not invalidate the current fixed-parameter run.

### P2: Existing Training Outputs Lack Overwrite Protection

Location: output directory creation and configuration, log, and checkpoint writing in `training.py:run`.

A new run without `--resume` can target an existing directory, overwrite `config.json` and `latest.pt`, and append to the old `train.jsonl`. This mixes evidence from two runs and can overwrite recoverable state.

Reject non-resume runs when training artifacts already exist and require a new directory; verify output ownership when resuming. If logs contain steps after the restored checkpoint, mark the abandoned attempt or repair the log tail so reports do not double-count replayed steps. Using unique directories avoids the first problem.

## Evidence Requiring Careful Interpretation

The early `starter-dev` summary, containing only 19 short-text seed samples, had approximately 0.82 WER and 0.44 insertion rate, with every output ending in EOS. Several transcriptions primarily added material at the target opening, including fragments resembling the end of the reference recording. This is a development diagnostic clue, not a causal determination or evidence of long-text omissions.

The lead agent arranged fixed-checkpoint diagnostics of reference-text formatting and removal of the LM reference prefix while keeping the same acoustic reference. These require separate configurations and directories to avoid mixing records with the original baseline. No outcome is predicted here.

## Core Paths Without an Identified Defect

These conclusions come from reading the implementation and existing test code; this review did not rerun the full tests.

- The input `SOS + text + TASK + speech` and targets are correctly shifted: TASK predicts the first speech token, and the final speech input predicts EOS. Text targets and padding use `IGNORE_ID=-1`; cross-entropy is divided by the number of valid speech targets.
- Sequences are concatenated only after slicing to true lengths. Right padding has a key padding mask, and the causal mask blocks future inputs. Padding positions have position embeddings but are neither valid keys nor supervised targets.
- KV caching reuses the original weights under the fixed architecture (post-norm, dropout=0, same activation). Prefix computation uses causal SDPA; a single-step query can attend to every existing key; multistep increments use an explicit allowed mask with an offset. A full-capacity prefix may predict one new token, after which further feedback is blocked.
- Generation returns neither the reference speech prefix nor EOS. Metadata separately records reference length, generated length, and stopping reason.
- RNG state resets for each text/seed, so a sample does not depend on whether its predecessor was restored. LM and acoustic modules share the global RNG: paired seeds do not imply elementwise paired acoustic noise when generation lengths differ, but each configuration remains reproducible.
- Unweighted sampling visits each valid sample exactly once per epoch. Long-sample weighting explicitly samples with replacement and records both total sample presentations and unique coverage.
- Learning-rate progress, stopping budgets, and counters use `speech tokens + EOS`. The final batch may slightly exceed the budget, with the actual token count retained. Equal step counts are not treated as equal training exposure.
- Validation loss and accuracy are weighted by valid target tokens. Sampling seeds are fixed, and validation makes no optimizer updates.
- Evaluation writes PCM16 audio before reading that delivered file for ASR, and verifies WAV hashes on resume. Initial and resumed scoring therefore use the same audio precision.
- ASR and synthesis exceptions retain their original errors. Aggregate WER uses reference words from successfully scored items and explicitly reports `scored_items` and exception rates; exceptions are not silently assigned zero WER.
- Data preparation separates final and development manifests; training does not load final data. The lead agent requires method freezing before final generation.

## Reporting and Follow-up Checks

1. If ASR failure sets differ across configurations, do not compare only their separate successful subsets. Report at least failure-stage counts and the common-success comparison. Current summaries retain errors but do not directly break them down by `error_stage`.
2. Compare B/C using actual consumed tokens. `best.pt` may precede the budget endpoint: disclose its step and token count and retain equal-budget endpoint results.
3. Step-based validation gives long-sample runs a different validation frequency per token. Best-validation-checkpoint comparisons should align candidate token milestones or disclose this selection difference. Fixed-endpoint comparisons avoid it.
4. Different generation seeds for the same text are not independent texts; use paired or clustered uncertainty analysis by text.
5. Report training engineering separately from quality-method contributions. Cache equivalence, throughput measurements, and length-budget corrections cannot replace evidence of training effectiveness.
