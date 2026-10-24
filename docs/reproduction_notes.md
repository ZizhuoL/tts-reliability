# Reproduction Inputs and Resume Boundaries

The reproduction entry point is the [README](../README.md). This page documents inputs needed for exact reconstruction and resuming existing runs. No environment reinstallation or repeated training was performed here; file verification is not full-pipeline reproduction.

## Preparing a New Environment

Use the validated Python 3.12 environment and install dependencies after pinning the course commit as described in the README. `requirements.txt` lists direct dependencies; `requirements.lock.txt` records versions installed locally. Versions or checksums for course data, starter weights, and acoustic/ASR components are recorded by upstream `course_download`, `model_manifest.json`, and individual run manifests. `scripts/download_assets.py --models` handles course data, starter weights, extraction, and frozen components.

`prepare_eval_data.py` prepares both the fixed reference recording and evaluation texts by default. To reproduce this project's length groups, explicitly use the README's 6/17/43/72 boundaries and seed 1832026; script defaults are insufficient. Run only in a new preparation directory. Existing evaluation manifests are protected; do not regenerate data or reference assets during active experiments.

## Exact Evaluation Texts Depend on Download Snapshots

Gutenberg download URLs are not version-pinned endpoints. Even with the same random seed, future changes to text or paragraph boundaries can change evaluation IDs and selected texts. Preserve these existing snapshots and `data/eval/manifest.json` for exact reconstruction:

| File | Recorded SHA-256 |
| --- | --- |
| `data/eval/sources/pg11.txt` | `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e` |
| `data/eval/sources/pg1661.txt` | `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0` |

The preparation script reuses existing literary source files and records their actual hashes in its manifest; it does not hard-code the literary snapshot hashes above as download checks. Compare recorded source hashes after any new download, and do not claim an identical evaluation set if a source changed. Preserve development/final JSONL files together with source snapshots. Use final data only after methods and checkpoints are fixed under the protocol.

The reference recording is different: its LibriSpeech parquet download uses a pinned revision and expected SHA, selects a fixed utterance, and retains original FLAC and PCM16 WAV files. `data/reference/prompt.json` preserves the original uppercase transcript. Formal evaluation must keep passing `--prompt-metadata configs/reference_prompt.json`, which uses sentence case and punctuation with the same word order, rather than accidentally reverting to the early diagnostic format. Both literary snapshots, the parquet, original FLAC, and WAV were individually verified in the [input audit](../results/analysis/reproduction-input-audit.json). That verification did not open the final text file.

## Resuming Existing Training Versus Starting a New Run

Training checkpoints save optimizer, token scheduler, sampler, RNG, and cumulative coverage; resume through `scripts/train.py --resume <checkpoint>`. Resume checks validate absolute data paths, sizes, mtimes, actual validation batches, sampling/optimization configuration, and core source hashes. Moving the workspace, reserializing identical samples, or changing bound source files can make existing resume checks reject the run. Automatic migration to arbitrary paths is not implemented.

Preserving the current workspace and source inputs allows resumption using the existing scripts and saved configuration. Reproduction in a new workspace should start from the starter and fixed inputs, use a new output directory, and retain a new manifest. Do not edit old records to pretend they belong to a new path. Neither completed nor failed run identities may be reused by overwriting files.

Small CPU resume tests were bitwise identical, but actual MPS replay showed numerical differences. Saving RNG, data order, and optimizer state does not guarantee bitwise equivalence across MPS runs; quality checks should report seeds, effective budgets, coverage, and statistics. Keep shell/Python files unchanged during execution, including comments. The development pipeline's closing error is retained in the [reconciliation record](../results/analysis/development-execution-reconciliation/incident.json).

Actual final-evaluation freezing and resume behavior are documented in the [runner guide](final_evaluation_runner.md). Running reproduction commands cannot turn unfinished training or evaluation into completed evidence.
