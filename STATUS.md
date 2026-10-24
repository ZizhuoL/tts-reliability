# Project Status

Last substantive experiment update: October 22, 2026 (America/Los_Angeles).
Repository presentation update: October 23, 2026.

The original research implementation, two complete continuation-training runs,
development comparisons, frozen final evaluation, and delivery audit were
completed on October 20. A separate personal-voice demonstration was added on
October 22. This document summarizes recorded outcomes; it is not a live process
monitor or a substitute for the application's Goal state. No new training was
started as part of the repository cleanup.

The complete original handoff journal is retained locally at
`notes/STATUS.history.zh.md` and excluded from the repository. It preserves the
historical session IDs, intermediate observations, recovery details, and original
Chinese notes. Historical descriptions of active processes are not current jobs.
Personal recordings, transcripts, feedback, and generated personal-voice audio
remain local only and are excluded from the repository.

## Completed Research

- All computation ran locally on an Apple M2 Max with 64 GB of memory. The LM and
  acoustic flow used MPS; the vocoder and Whisper ASR used CPU. No paid API or cloud
  GPU was used.
- The course source was pinned to commit
  `48e1c407dec5a39df1a50a9069934879f175c373`. The 213,191,842-parameter LM began from
  the supplied checkpoint; this project did not train the full CosyVoice2 system
  from scratch.
- Both continuation-training seeds, 183 and 184, completed one full epoch:
  44,348 accepted updates, 354,780 unique training examples, and 50,450,231
  supervised target tokens each. Main comparisons use the complete-budget
  `latest.pt`, not the earlier `best.pt`.
- Full teacher-forced validation used the same 9,756 course examples for all three
  checkpoints. Cross-entropy was 4.54146 for the starter, 4.46617 for seed 183,
  and 4.46599 for seed 184. These are course-validation results, not independent
  generation-quality scores.
- Development used 45 texts and generation seeds 17/42. Standard, no-LM-prefix,
  and sentence-split strategies were evaluated for the starter and both trained
  checkpoints. BNoLM with training seed 183 was selected under the development
  rule; seed 184 was reserved for sensitivity analysis.
- The final protocol was frozen before the final text file was opened. All nine
  conditions completed 60 texts × two generation seeds: 1,080 scored attempts,
  zero runtime exceptions or missing attempts, and 30 completed pipeline stages.
  Output-limit terminations still occurred in some conditions; runtime success
  does not mean error-free speech.

## Main Results and Limits

| Final condition | Overall ASR WER |
| --- | ---: |
| A: starter, standard reference prefix | 25.3979% |
| B: continued training, standard reference prefix | 17.7056% |
| BNoLM: continued training, LM prefix removed | 9.4828% |
| B184NoLM: second training seed, LM prefix removed | 17.0424% |

The primary A → BNoLM difference was −15.9151 percentage points, with a 95%
text-cluster bootstrap interval of [−18.4507, −13.0477]. Long-text WER changed
from 27.6288% to 9.3884%. This combines training with a prompting change; the
training-only comparison is A → B. The reported primary intervals were not
adjusted for multiple comparisons.

The second training seed exposed a substantial failure: one medium-text attempt
had 213 ASR insertions and reached its output limit. The attempt remains in the
results. The BNoLM seed-183 → seed-184 difference was +7.5597 points, with a 95%
interval of [−1.4317, +23.8887]. Stable improvement across training seeds has not
been established. ASR text alone does not prove acoustic repetition.

The corrected baseline did not establish a clean failure specific to long
inputs, so length-weighted training was deferred rather than presented as an
executed experiment. ASR WER does not measure naturalness or speaker similarity.
The research evaluation did not include systematic human listening, MOS, speaker
similarity scores, or a clean-environment rerun. Some outputs crossed Whisper's
30-second chunk boundary; that limitation is retained in the reports.

## Key Milestones

| Local date | Recorded milestone |
| --- | --- |
| October 19 | Verified model/data correctness and real MPS execution; prepared fixed public reference audio and disjoint development/final text sets; corrected reference transcript formatting; started full continuation training. |
| October 20, early morning | Recovered seed 183 from two MPS out-of-memory failures, completed its full epoch and validation, and finished six main development conditions. |
| October 20, morning | Completed seed 184's full epoch, validation, and development sensitivity analysis; froze the final protocol at 07:18 PDT; finished the final evaluation at 09:49 PDT and delivery audit at approximately 10:00 PDT. |
| October 22 | Generated a separate local personal-voice demo and then a single-example repair in response to user listening feedback. |

Seed 183 failed twice at update 35,592 and resumed from the saved update-34,000
state. The successful recovery increased the bounded MPS memory fraction from
0.60 to 0.75 and saved more frequently. Data order, optimizer, effective training
budget, and core training implementation were retained. The two discarded tails
contain 3,182 updates and 3,681,000 target tokens: they count as spent compute,
but not as accepted training exposure. Replay was not bitwise identical on MPS.
The successful recovery does not isolate the causal effect of the memory change.

The main development pipeline produced its complete outputs but exited with a
closing shell error. Its nonzero exit and reconciliation evidence were retained;
outputs were independently checked rather than silently rerun. Do not edit a
running shell script, including its comments.

## Personal-voice extension: local only

A separate personal-reference demonstration and a single-example sentence-splicing
repair were completed on October 22. They are outside the frozen research
comparison and do not establish general voice-cloning quality. The complete
operational account is retained in the local historical journal.

Personal recordings, transcripts, feedback, demo scripts, and generated outputs
are excluded from Git. Their local entry points remain
`results/personalized-voice-v1/index.html` and
`results/personalized-voice-v2/repair.json`. This extension did not change the
frozen research experiment.

## Evidence and Recovery

Start with the [technical report](docs/technical_report.md),
[final results](docs/final_results.md),
[delivery audit](docs/delivery_audit_final.md), and
[reproduction notes](docs/reproduction_notes.md).

The original workspace retains these large/local evidence paths:

- Checkpoints: `checkpoints/B-ordinary-seed183/latest.pt` and
  `checkpoints/B-ordinary-seed184/latest.pt`.
- Recovery evidence: `checkpoints/B-ordinary-seed183/recovery/` and
  `results/analysis/B-recovery-costs.json`.
- Frozen protocol: `results/final-protocol-v1/protocol.json`.
- Completed runs: `results/final-evaluation-v1/`.
- Independent audits: `results/analysis/final-independent-audit.json`,
  `results/analysis/final-root-integrity-audit.json`, and
  `results/analysis/project-delivery-completion.json`.

These paths describe retained local artifacts; a source-only checkout does not
contain downloaded assets, checkpoints, or the full generated run archive.
Follow the README to prepare a new environment. From the project root, inspect
available checks and command options with:

```sh
source scripts/env.sh
.venv/bin/python -m pytest -q
.venv/bin/python scripts/train.py --help
.venv/bin/python scripts/run_final_evaluation.py --help
```

For an actually interrupted run, use the recorded configuration and
`train.py --resume <checkpoint>`, or the explicit final-runner resume mechanism
in the [runner guide](docs/final_evaluation_runner.md). Completed runs must not be
resumed or overwritten. Existing manifests bind source hashes, asset identity,
and sometimes absolute paths or mtimes; moving the workspace or changing bound
files may invalidate historical resume checks. Do not rewrite old evidence to
make it appear compatible with a new environment. New experiments need new
output directories and explicit records of any changed protocol.

## Repository preparation (2026-10-23)

The project now has an independent local Git repository on `main`, rooted only
in `tts-reliability/`, with `origin` set to
`https://github.com/ZizhuoL/tts-reliability.git`. The remote was confirmed private.
English documentation, selected result evidence, and path-redacted metadata
exports are prepared for the initial commit. Personal assets and bulk local
artifacts are excluded according to `docs/repository_contents.md`.

The cleanup passed all 47 existing CPU tests. Fourteen frozen source/configuration
hashes and the protocol receipt still match; all 1,080 final attempt records are
complete. This work did not start training or rerun synthesis/evaluation.

The initial upload is pending the owner's GitHub CLI authentication, initiated
from the owner's own terminal. The earlier agent-initiated device request was
denied; no project files were uploaded by that request.
