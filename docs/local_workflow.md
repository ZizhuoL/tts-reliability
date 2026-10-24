# Local workflow and experiment history

An independent extension of EECS 183/283A Homework 4 Part C. Two ordinary
continuation seeds each completed a full epoch over 354,780 valid examples,
complete course-dev validation and three development strategies. The frozen
independent final evaluation completed all nine cells: 60 texts × two generation
seeds × nine systems = 1,080 scored attempts, with no runtime exceptions.

The development-selected combination (seed183 continuation with the LM reference
prefix removed) reduced final ASR WER from 25.40% to 9.48%; the paired difference
was −15.92 percentage points (95% text-cluster bootstrap CI −18.45 to −13.05).
Long-text WER fell from 27.63% to 9.39%. This combines training and a decoding
change: ordinary continuation alone scored 17.71%. Seed184 with the prefix
removed scored 17.04%, including one severe repeated-ASR result, so the selected
system's overall gain is not established as stable across training seeds.

Start with the [technical report](technical_report.md),
[complete final results](final_results.md), and
paired listening index (local only: `../results/final-evaluation-v1/listening/INDEX.md`).
The [delivery audit](delivery_audit_final.md) maps the deliverables to evidence.
These are automatic intelligibility results; human listening evaluation is
not completed.

The trainable model is the course's 213,191,842-parameter causal Transformer,
initialized from its two-epoch starter. CosyVoice2's speech tokenizer, speaker
encoder, flow model and vocoder are frozen components, not models trained from
scratch by this project. Whisper supplies an automatic intelligibility proxy,
not a human naturalness or speaker-similarity score.

## Local environment

Validated machine: Apple M2 Max, 64 GB unified memory, macOS 15.7.9,
Python 3.12.3, PyTorch/torchaudio 2.8.0. All inference and training run locally.
Data, public model weights, Python packages and upstream code require downloads.

From this directory:

```sh
mkdir -p upstream
git clone https://github.com/Louis0324/eecs183-hw4.git upstream/eecs183-hw4
git -C upstream/eecs183-hw4 checkout 48e1c407dec5a39df1a50a9069934879f175c373
python3 -m venv .venv
source scripts/env.sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_assets.py --models
.venv/bin/python scripts/prepare_training_data.py
.venv/bin/python -m pytest -q
```

`requirements.lock.txt` records the installed versions. `scripts/env.sh` keeps
download/cache directories inside this project. The unmodified course checkout
is pinned to `48e1c407dec5a39df1a50a9069934879f175c373` under
`upstream/eecs183-hw4`. Its manifests pin model revisions and course checksums.

MPS must be available to the Python process. The app's restricted shell sandbox
cannot access it on this machine; authorized local execution can. Training
explicitly fails if MPS is unavailable. Prompt ONNX/STFT, the vocoder's complex
iSTFT, and Whisper currently run on CPU, while the LM and flow model use MPS.

The initial training attempt and one fresh-process retry both hit the configured
0.6 MPS memory ceiling at update 35,592. Their logs and the saved update-34,000
checkpoint are preserved under `checkpoints/B-ordinary-seed183/recovery/`.
An isolated bounded 0.75-ceiling probe passed a batch padded to `[8,1216]`,
including backward and parameter updates. It used different samples and a
shorter process history, so this does not prove the ceiling change fixes the
original failure. Its temporary updates never enter the formal checkpoint.
`scripts/resume_B_memory075.sh` completed formal training with a 0.75 ceiling
and saves every 500 steps, followed by full validation. Batch size, data order,
optimizer, learning-rate schedule and core source remain fixed. The
original failing batch has now succeeded in formal recovery (step 35,592:
loss 4.31045, finite gradient norm 1.29619), with subsequent updates observed.
The full epoch and full course-dev validation subsequently completed. The first
retry's audited replay had matching sample, token, cursor and learning-rate
records, but MPS numerical results were not bitwise identical. Discarded work,
the probe and recovery overhead are retained for cost accounting.
These device choices are recorded in each evaluation manifest.

## Data and evaluation preparation

```sh
.venv/bin/python scripts/prepare_eval_data.py \
  --min-tokens 6 --short-max 17 --medium-max 43 --max-tokens 72 --seed 1832026
```

Run preparation only for a fresh data directory; existing evaluation manifests
are protected from replacement. The explicit length arguments reproduce this
project's protocol rather than the script's exploratory defaults.

The fixed reference is the LibriSpeech CC BY 4.0 recording
`1272-128104-0000`. `data/reference/prompt.json` retains its source, transcript,
transformation and hashes. `data/eval/manifest.json` records the public-domain
natural-text sources, preprocessing, duplicate checks and development/final
partition. The sealed final set is not used to choose methods or checkpoints.

For exact source snapshots, fixed-reference formatting and the path-bound
resume policy, see [reproduction notes](reproduction_notes.md). Gutenberg
URLs are mutable; retaining this run's source files is necessary for exact
evaluation-set reconstruction.

Training and course validation comprise 354,780 and 9,756 valid paired examples.
Both text and speech lengths are checked without independently truncating
paired modalities. Derived distributions are in `data/processed/statistics.json`.
The original cache's speaker map and `num_samples` metadata describe an older
subset; actual counts are recomputed from sample records.

`results/starter-provenance.json` records metadata read directly from the starter:
two completed epochs with 354,780 examples per epoch, a reported validation loss,
and no optimizer state. That upstream validation value is not a local remeasurement
or necessarily the same aggregation as this project's token-weighted validation.
The metadata alone does not establish how validation influenced upstream selection.
The course validation set is therefore never labeled an independent final test.

## Executable entry points

```sh
# End-to-end starter/decoder/ASR check (writes results/smoke).
.venv/bin/python scripts/smoke_audio.py

# Representative batches, cold and repeated measurements, padding and memory.
.venv/bin/python scripts/train.py --mode benchmark \
  --data data/processed/train.pt --output results/benchmark-new \
  --batch-size 8 --pad-multiple 64 --steps 8 --benchmark-repeats 3 \
  --sequence-implementation packed --empty-cache-every 1 --mps-memory-fraction 0.6

# Development diagnosis. Unique output directories protect experiment identity.
.venv/bin/python scripts/evaluate.py \
  --checkpoint models/hw4-tts-starter.pt --texts data/eval/dev.jsonl \
  --output results/starter-standard-dev-new --seeds 17 42 \
  --prompt-metadata configs/reference_prompt.json

# Same evaluation with the simple sentence-splitting engineering control.
.venv/bin/python scripts/evaluate.py \
  --checkpoint models/hw4-tts-starter.pt --texts data/eval/dev.jsonl \
  --output results/starter-split-dev-new --seeds 17 42 --split-sentences \
  --prompt-metadata configs/reference_prompt.json

.venv/bin/python scripts/train.py --help
.venv/bin/python scripts/evaluate.py --help
```

The ordinary continuation budget is one full valid epoch: 50,450,231 supervised
speech targets including EOS. `scripts/run_baseline_and_training.sh` serializes
the standardized development baseline, ordinary training and full course-dev
validation. That initial process failed; the explicit recovery script above
completed successfully. `scripts/run_development_comparisons.sh` completed
the starter full-validation reference, the remaining five development
conditions and their analyses, but the outer shell later exited 127
on a trailing command fragment. The original log is retained; full artifact and
prerequisite checks passed before starting seed184. See
`results/analysis/development-execution-reconciliation/incident.json` and STATUS.

Training supports atomic optimizer/scheduler/RNG/sampler checkpoints, validation,
and explicit resume; exact commands for selected experiments belong in STATUS
and their saved `config.json`. A partial epoch is never reported as a full one.

The completed seed183 epoch contains 44,348 accepted updates, 354,780 unique
examples and 50,450,231 targets, with no repeated examples or budget overshoot.
`results/analysis/B-final-exposure.json` independently reconstructs every accepted
batch and matches the final checkpoint's sampler and counters. The full
9,756-example course-dev cross entropy is 4.46617; this is teacher-forced token
prediction, not generated-speech WER or independent final-test performance.
See [the training results chapter](training_results.md) for the paired
full-validation reference, recovery costs and observed-memory limits.
The [development results chapter](development_results.md) reports all six
systems, paired uncertainty, one-factor contrasts, stopping behavior and resource
costs. The selected combination's development WER is 10.17% versus 25.91% for
the standardized starter; this is not an independent final-test estimate.

`scripts/run_B_seed184_replication.sh` completed a second ordinary-training
seed after the first epoch, full validation and six-cell development comparison.
Its default `--check-only` verifies prerequisites without loading models or
creating a run; `--run` performs the serial training/validation/development
pipeline. Its 44,348 updates cover every valid example once, as checked in
`results/analysis/B184-final-exposure.json`; all 270 development attempts were
scored. Both seeds use full-epoch endpoints, and seed183
remains primary regardless of which scores better. Periodic validation subsets
differ with the training seed; the complete 9,756-example validation is shared.

The [final evaluation runner](final_evaluation_runner.md) froze the actual
protocol at 2026-10-20 14:18:25 UTC, before opening the final set.
`results/final-protocol-v1/` preserves the protocol and SHA receipt; the primary
comparison is BNoLM versus A. Nine cells, each with 60 texts and two generation
seeds, completed in `results/final-evaluation-v1/` at 16:49:20 UTC. The first
final-file read passed its recorded gate at 14:19:06 UTC. All 30 stages completed;
the original records, graphs, failure profiles, audio and completion markers
are retained. Frozen configurations and bound sources remain unchanged.
`configs/protocol.json` is a historical pre-final snapshot; current completion
is recorded in `results/final-evaluation-v1/completion.json` and STATUS.
The two primary outcome CIs (overall and long WER) are individually 95%,
without multiplicity adjustment or a joint-coverage claim.

The original reference transcript remains unchanged in `data/reference`.
Formal comparisons use `configs/reference_prompt.json`, which preserves the
same words and audio while adding sentence case and punctuation. The controlled
diagnostic and rationale are recorded in
[docs/development_decisions.md](development_decisions.md); this correction
is not counted as a training improvement.

Evaluation writes original text/transcripts, word alignments, WER components,
generation boundaries, timing, hashes and audio per text/seed. The default new
speech-token budget is `max(500, ceil(11 × target_text_tokens))`; context capacity
is a separately recorded limit. The ratio is based on the training distribution
(speech/text ratio's 99th percentile is 10.5), not final-test tuning.

## Attribution and scope

- Course architecture, token cache, starter and frozen-component helpers:
  [Louis0324/eecs183-hw4](https://github.com/Louis0324/eecs183-hw4).
- Frozen speech components:
  [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B).
- Automatic evaluator:
  [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo).
- Training recordings underlying the course cache:
  [LibriTTS](https://www.openslr.org/60/), CC BY 4.0.
- Research context and independent text/reference provenance:
  [docs/research_notes.md](research_notes.md).

Project additions include the missing model/data implementation,
tested KV cache, local MPS adaptation, shape-bounded training padding,
recoverable token-budget training, independent text preparation and detailed
evaluation records. No homework submission archive is created.

See [contributions and resume wording](contributions_and_resume.md) for
precise ownership boundaries and evidence-based English bullets, and
[the architecture and interview guide](architecture_and_interview.md) for
teacher forcing, causal attention, decoding and experiment-design explanations.
