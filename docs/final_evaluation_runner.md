# Final Evaluation Runner: Nine Frozen Conditions Completed

The actual protocol was frozen at 2026-10-20 14:18:25 UTC in [final-protocol-v1](../results/final-protocol-v1/protocol.json), SHA `1784f64b2ed3ddf1094b5b0795faffa86c99812dfdbfba8924db01be0b59aa38`. Before freezing, both independent full epochs, matching full course validation, nine development cells, and three-strategy training-seed checks were complete. The primary comparison is **A vs BNoLM**, with primary training seed 183. At 14:19:06 UTC, the runner durably wrote the prerequisite audit event before first reading final data; all 60 texts and the sealed SHA passed verification. All 1,080 scores across nine conditions and all 30 stages completed at 16:49:20 UTC, production exit 0, with 0 runtime exceptions. Records are in `results/final-evaluation-v1/`; see [final results](final_results.md). This approximately 2-hour-30-minute final pipeline excludes earlier training/development costs.

Freeze, execution, and resume commands remain below for reproduction. Templates and candidates retain `TEMPLATE_NOT_FROZEN`; the independent protocol and SHA receipt are the actual freeze evidence. Preparation and artificial-fixture testing did not read real final data. Do not change frozen configuration, code, or other bindings. Selection followed the six primary development cells; seed 184 served only as sensitivity analysis, with actual freezing waiting for its full completion.

## Fixed Scope

- Primary training seed is always **183**. B uses full-epoch `latest.pt`: 44,348 steps, 50,450,231 targets, 354,780 unique samples. Do not substitute `best.pt` or choose the primary seed based on seed 184 results.
- Six required cells: A, ANoLM, ASplit, B, BNoLM, BSplit. Each uses the same 60 final texts × generation seeds17/42 =120 attempts; six cells total 720.
- Default sensitivity cells B184, B184NoLM, B184Split produce nine cells/1,080 attempts. Budget reductions require disabling the corresponding template switch with a specific reason before freezing. Even if seed 184 final generation is omitted, its planned training/development checks must complete before freezing.
- Primary comparison is the development-selected system versus A. Always report A vs B standard. Other comparisons are exploratory; seed 184 is secondary sensitivity analysis, never a basis for reselecting the primary system.
- If no candidate passes the conservative filter, retain A and set `primary_comparison` to `null`, explicitly stating that none passed. Do not invent A-vs-A tests or a winner. Report A's final performance and the planned exploratory A vs B comparison.

Generation matches completed development runs: fixed reference WAV and `configs/reference_prompt.json`, adaptive `max(500, ceil(11 * target_text_tokens))`, temperature 1, top-k 50, minimum ratio 2, KV cache, split pause 0.15 seconds, LM/flow on MPS, vocoder/ASR on CPU. Segment budgets add together; splitting does not match whole-text actual tokens or compute. Cost analysis records actual segments, caps, generated tokens, and pauses.

## 1. Fill the Candidate Configuration

From the project root:

```sh
.venv/bin/python scripts/run_final_evaluation.py template-check
cp -n configs/final_evaluation.template.json configs/final_evaluation.candidate.json
```

Fill the candidate from the six primary development cells; actual freezing must wait for seed 184 checks. Required fields:

- `selection.selected_system`: one of the six development systems, excluding B184 systems.
- `selection.rationale`: pre-freeze evidence for filter checks, ranking, or A fallback.
- `selection.missingness_review`: failure denominators, common-success pairing, and whole-reference-deletion sensitivity; explicitly record when all 90 items are scored.
- `selection.failure_robust_eligible_systems`: keep `null` with no missing scores. If any cell has unscored attempts, list candidates still supported by common-success and deletion-sensitivity analyses; an empty list means none are supported.

The script checks strict overall-WER improvement, no short/long WER increase, and no cap/exception-rate increase. It ranks by overall WER, total generation time, and standard → no-LM → split complexity, using fixed cell order for exact ties. The lead agent must still review substantive missingness interpretation; a written assertion does not prove robustness.

`template-check` validates structure and fixed parameters only, not development completion or candidate support. It does not inspect final text bytes or hashes. Output must be a new, nonexistent directory.

## 2. Create the Actual Frozen Protocol and Hash Receipt

```sh
.venv/bin/python scripts/run_final_evaluation.py freeze \
  --candidate configs/final_evaluation.candidate.json \
  --output results/final-protocol-v1
```

Hard prerequisites:

1. All nine development runs contain matching 45 × 2 terminal `complete`/`error` events, no `synthesized` rows pending ASR, and valid saved WAV hashes.
2. Both183/184 runs reach full-budget endpoints. Full validation uses 9,756 samples/1,362,834 targets with identical actual batches. Different full-validation `subset_seed` metadata does not imply different full batches.
3. Selection uses only the six-cell primary development overview. Analysis completion/artifact hashes are valid, raw input hashes match complete current evidence, and all three strategy replication completions exist.
4. Checkpoints, references, code, acoustic/ASR assets, and packages are bound to development runs. Large files are rehashed by streaming CPU reads without model deserialization.

After passing, exclusive creation writes `protocol.json`, flushes/fsyncs it, then writes `protocol.sha256.json`; overwrites are rejected. The final-text SHA **only references the already sealed declaration in `data/eval/manifest.json`**. Freezing does not read/hash final text contents.

This is auditable procedural freezing, not a tamper-proof timestamp or OS access control. Preserve protocol, receipt, candidate, and binding evidence. Startup rechecks source hashes; changes to bound code/configuration prevent launch/resume. Necessary repairs must preserve original evidence and explicitly document deviations, never reselect based on final results.

## 3. Inspect Commands, Then Execute Serially

```sh
.venv/bin/python scripts/run_final_evaluation.py plan \
  --protocol-dir results/final-protocol-v1
```

`plan` validates the receipt and prints generation, analysis, cost, failure-profile, and listening commands without reading final texts or invoking models.

Only after the lead agent confirms prior GPU sessions have ended and the sole GPU is available:

```sh
caffeinate -i .venv/bin/python -u scripts/run_final_evaluation.py run \
  --protocol-dir results/final-protocol-v1 \
  --confirm-gpu-idle
```

The runner revalidates all frozen bindings/package versions, checks known Python training/evaluation processes, and acquires its own lock. Subprocesses run serially. Older scripts do not share the lock, and arbitrary external GPU use cannot be identified completely; `--confirm-gpu-idle` requires actual scheduling confirmation, not just lock acquisition.

Before first reading final data, it durably writes a run record bound to the frozen SHA and the `frozen_protocol_verified_before_final_read` event. It then reads contents and verifies the sealed SHA, 60 unique IDs, and 20 texts per length group. Any mismatch stops execution without regenerating/replacing texts.

## 4. Outputs and Integrity

Outputs use the candidate's predetermined new directory, default `results/final-evaluation-v1/`:

| Path | Contents |
| --- | --- |
| `runs/<cell>/` | Raw JSONL events, WAVs, manifest, runtime, original summary |
| `analysis/overview/` | Unified rescoring across cells, S/D/I, length/seed results, A-baseline pairing, CSVs, plots |
| `analysis/within-B183/` and related paths | Within-B strategies, fixed-strategy training comparisons, 183/184 sensitivity, plots |
| `costs/` | Completed/failed coverage, actual segments, budgets, generated tokens, pauses, generation/ASR time |
| `resources/` | Generation/ASR/attempt time, RSS/MPS observations, audio over 30 seconds, scope caveats |
| `failures/<cell>/` | ASR-alignment failure profiles, tail-deletion candidates, length analysis; not human-confirmed defects |
| `listening/` | Automatically selected better/worse short/medium/long and paired improvement/regression examples, copied audio and rationales |
| `logs/` | Separate subprocess logs, failure/resume events, measured subprocess wall time |
| `stage-markers/` | Artifact SHA lists for completed stages |
| `comparison-roles.json` | Frozen primary/research/exploratory roles, unchanged by final results |
| `completion.json` | Created only after every required stage succeeds |

Each cell requires 120 unique id/seed terminal attempts. Exit 0 does not turn errors into successes. Successful rows require real WAVs; every recorded WAV hash is checked. Analysis retains original metrics and applies one normalizer, separately reporting scored WER, common-success pairs, and missing-output whole-reference-deletion sensitivity. Length-stratified bootstrap retains both seeds in each text cluster rather than inflating the independent sample count.

Subprocess wall time includes loading, hashing, and execution; per-item generation/ASR/attempt times are separate measures. Failed-process wall time remains in audit logs; successful-stage sums alone are not total cost. Interpret RSS high-water and between-step MPS observations separately; MPS snapshots are not peak measurements. A listening collection organizes samples but does not establish human listening, MOS, or speaker-similarity evaluation.

## 5. Resume and Overwrite Protection

```sh
caffeinate -i .venv/bin/python -u scripts/run_final_evaluation.py run \
  --protocol-dir results/final-protocol-v1 \
  --confirm-gpu-idle --resume
```

Explicit `--resume` accepts only the same frozen SHA. Completed stages are skipped only after all artifact hashes pass. Existing manifests/events allow valid saved WAVs to resume at ASR. Completed/error records remain; errors are not automatically retried. A truncated JSONL row requires explicit repair and retained evidence, never silent deletion.

If interrupted after `comparison-roles.json` but before `completion.json`, resume compares existing roles exactly with protocol-derived values and reuses the file without changing bytes/timestamps. Different content is rejected rather than overwritten or reselected.

Incomplete offline-analysis directories block resume to avoid overwriting partial evidence. Preserve and move the partial directory explicitly, then resume from the same protocol; do not delete/fabricate completion. Fully completed runs reject reexecution. Protocol changes, error retries, or new generation settings require a clearly labeled supplementary experiment rather than overwriting the primary comparison.

## Preparation Checks Actually Run

```sh
.venv/bin/python scripts/run_final_evaluation.py template-check
.venv/bin/python -m py_compile scripts/run_final_evaluation.py scripts/test_final_evaluation_runner.py
.venv/bin/python scripts/test_final_evaluation_runner.py
```

Temporary artificial-fixture regressions cover unfilled-template rejection, final-read/hash blocking, illegal-open detection during freezing, binding/protocol tamper rejection, A fallback, incomplete-development rejection, incorrect selection/unreviewed missing-score rejection, idle-GPU confirmation, nine-cell argument compatibility/fixed seeds, and interrupted role-file recovery/tamper rejection. Closing-recovery tests read only 60 artificial fixture texts and prohibit subprocess execution. Temporary freezes are not the project's real protocol; tests load no models and read no real final data.
