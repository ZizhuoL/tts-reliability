# Local TTS Reliability Project

This document is the complete project brief for handoff to a new Codex chat. Saving these instructions does not mean the project has started or finished. The user must explicitly request the research launch.

> **Repository authorization, 2026-10-23:** The user authorized consistent English project files, an independent local Git repository in `tts-reliability/` rather than the parent workspace, and creation of and upload to a **private** repository on the user's GitHub account. Curate repository contents for presentation and reproducibility; exclude private personal artifacts and bulk assets, including datasets, model weights, environments, and generated audio. This supersedes the earlier upload restriction only for that curated private repository. It does not authorize starting or resuming research or training, public sharing, or coursework submission. Keep chat communication in Chinese by default; written project deliverables are now in English.

## 1. Background and Objectives

Course repository: https://github.com/Louis0324/eecs183-hw4

The user is studying NLP and has experience with mathematics, optimization, and a handwritten MLP project. They want an independent project that includes model training and can be explained on a resume and in interviews. They are more interested in TTS than ASR or chart VLMs.

Use Homework 4 Part C as the starting point for a reproducible TTS project on an Apple M2 Max (initially estimated to have 64 GB of memory; verify this). Perform training, inference, audio synthesis, and evaluation locally. Necessary public code, dependencies, data, and models may be downloaded; do not use paid APIs or cloud GPUs.

The original target was a first version in approximately 1–2 days. Plan a work budget of roughly 24–48 hours based on measured throughput. This is neither an unmeasured completion guarantee nor a promise to occupy all available compute continuously.

Candidate research question:

> Under a limited local training budget, can a small autoregressive TTS model make fewer omissions, repetitions, and premature stops when reading longer text while preserving short-text performance?

First establish from the literature, code, and baseline experiments that this question is worth investigating. Do not assume that the course model fails or guarantee an effective improvement. Do not manufacture failures with inputs beyond the model's capacity or deliberately weak baselines.

Completion means more than training for more epochs than the assignment: a reliable implementation, a real problem, a justified modification, fair experiments, independent evaluation, and reproducible deliverables.

## 2. Questions to Resolve at Launch

First read the repository README, Part C notebook, data documentation, model manifest, checkpoint-loading code, and synthesis and evaluation helpers. Perform the necessary read-only environment checks.

Ask together only questions that require the user's decision and cannot reasonably be inferred. Cover the working directory, necessary reference audio, disk space and downloads, runtime budget, and permissions that must be arranged early when relevant. Recommend choices, do not repeat known questions, and do not invent questions merely to satisfy a process.

The user may want the project to run overnight. After receiving necessary answers, create and start the Goal; if there are no required questions, start directly. Decide ordinary technical details independently. Progress updates do not require a user reply.

While the user is still available during launch, complete necessary downloads, one real training batch, one complete audio synthesis, and one ASR evaluation as early as possible to expose blockers.

If the user's own recording is unavailable, prefer a fixed public reference recording with clear licensing and an accurate transcript. Record its source and use it consistently across conditions. Do not block the project over a replaceable recording.

## 3. Directory, Resources, and Collaboration

The project root is the `tts-reliability/` subdirectory of the current workspace. If the user moves the workspace, use the project's actual location.

Use an isolated dependency environment and separate configuration, checkpoint, and output directories. Protect existing files, inspect applicable project instructions, and do not overwrite original course files or neighboring projects.

The user may be developing another lightweight project concurrently; this alone is not a reason to reduce training scale. Use available resources while the machine remains stable. Adjust batch size, loading order, or concurrency if significant memory pressure, sustained heavy swapping, failures, or throughput degradation occurs.

Multiple agents may inspect code, research methods, analyze results, or prepare documentation. The lead agent coordinates responsibilities, resources, integration, and verification to avoid editing conflicts, duplicate downloads, and duplicate training.

Do not change system-level settings, disable safeguards, or delete unrelated content. The original research authorization did not permit automatic repository publication or uploads of data, recordings, or models; the dated authorization above permits only the curated private repository. Do not create or submit a coursework package: this is an independent extension based on the course.

## 4. Local Feasibility and Full-Data Training Priority

Verify the actual chip, memory, free disk space, runtime environment, and model/checkpoint compatibility. Check local execution paths for training, audio synthesis, and ASR evaluation separately.

The course notebook may choose only between CUDA and CPU. Explicitly check MPS support. Do not silently start long training on CPU or claim GPU use without verification. Record any operators that require CPU fallback and their performance impact.

Prioritize compatibility with the course checkpoint. Do not default to rewriting the framework, but make necessary, verifiable adaptations.

Training scale:

1. Do not reduce the project to toy data merely because it runs on a Mac. The user wants, where feasible, one continued-training epoch over the complete valid training set.
2. Warm up and time representative batches reflecting the real length distribution. Measure memory, data loading, and padding overhead. Do not estimate from short samples alone.
3. Estimate the combined time for a full epoch, validation, audio generation, control experiments, and reporting.
4. Run full-data training if it fits the budget. Otherwise reduce scope independently and explain the evidence; ordinary budget allocation does not require another question.
5. Reserve time for fair comparisons and final evaluation rather than spending the entire budget on training.
6. Report full epochs, partial datasets, fixed step counts, and actual valid-sample coverage distinctly.

The course documentation previously listed approximately 354,780 training examples. Recheck the actual valid count, version, and filtering. A default batch size is no substitute for measured throughput.

## 5. Correct Implementation and Baseline

Reuse or implement the necessary Part C functionality and verify:

- Input/target shifts for text and speech tokens.
- Causal masks, padding masks, and ignored positions.
- The speech EOS token ID.
- Gradients, parameter updates, and train/eval behavior.
- Checkpoint saving and restoration.
- The boundary between the reference-audio prefix and newly generated speech.

Pass small, meaningful correctness checks before long runs. Falling loss is not sufficient evidence of correctness.

Clearly attribute the course architecture, code, data, and checkpoint; CosyVoice2's frozen components; and this project's additions. Do not claim to have trained the full CosyVoice2 system from scratch.

The following observations from preliminary code reading are leads to recheck against the version actually used, not established defects:

- Allowed training speech-token lengths differ from the default inference `max_length`.
- The minimum-output-length heuristic.
- Total context capacity and reference-prefix usage.
- Whether generation recomputes the complete history.
- Whether token validation loss agrees with actual generation quality.

First establish a baseline with a reasonable generation-length budget. Distinguish truncation at a hard limit from the model choosing to stop. Do not present simply increasing `max_length` as a training contribution.

For every generation, save the text and its length, model and checkpoint, reference-audio identifier, random seed, sampling settings, token count, stop reason (EOS, output limit, context limit, or exception), elapsed time, audio duration, audio, ASR transcript, and error information.

## 6. Data and Evaluation

Measure training text lengths, speech lengths, their ratio, invalid and over-limit examples, and speaker distribution when relevant. Define short, medium, and longer groups from the actual distribution.

Use natural, understandable inputs with a real reading purpose and within the model's reasonable context capacity. Establish separate diagnostic/development and final test sets, checking overlap with training text. Do not use the final test set to choose hyperparameters, methods, or checkpoints.

Record how the starter uses training and validation data, and state uncertainty explicitly. Do not treat course validation data already used for checkpoint selection as a fully independent test set. Do not repeatedly tune against the course's final test questions.

Start with approximately 30–60 diagnostic texts, then determine formal evaluation size from measured synthesis speed. Include naturally occurring texts with explicit sources, not only self-written difficult sentences.

At minimum, report:

- WER and deletion, insertion, and substitution components.
- Results by length group.
- Failure rates, including limit truncation and exceptions.
- Analysis of omissions, repetition, and premature endings in representative samples.
- Training time, memory, and generation time.

Apply consistent text normalization and preserve raw transcripts so differences in number or punctuation formatting do not distort conclusions. WER deletions do not automatically establish omitted speech, and insertions do not automatically establish repetition; examine alignments, stopping reasons, and audio together.

ASR is not ground truth for audio quality. Do not infer naturalness, speaker similarity, or complete content correctness from transcripts alone. Do not invent human listening, MOS, or tests that were not conducted.

If user listening feedback is unavailable, complete automated evaluation and organize representative audio. State that human evaluation remains incomplete without blocking other deliverables.

Use multiple seeds for key generation comparisons. Recheck training randomness when the budget permits. State uncertainty for small samples and do not treat chance improvements as stable gains.

## 7. Fixed Baseline, Open Improvement Route

Part C is already a causal Transformer. Retain it as a reliable starting point, but other models are allowed and length-balanced training is not mandatory.

Depending on the evidence, consider:

- Changes to the training-data distribution or objective.
- Generation-strategy improvements.
- Local architecture changes.
- Other models suitable for the machine and time budget.

Before choosing a route, state the observed problem, supporting evidence, whether weights and data can be reused, runtime cost, and design of a fair comparison.

An existing TTS model may serve as an external reference, but explain differences in pretraining data, size, and budget. Do not describe that comparison as a strict architecture ablation.

Prioritize thorough verification of one justified route, starting with small trials before long training. Do not add complexity merely to appear sophisticated or reject a better method just to preserve the original plan. Make these decisions independently within the project's confirmed scope.

Candidate training experiments, only if supported by data and failure analysis:

- A: Original starter.
- B: Ordinary continued training with the original data distribution.
- C: Continued training with a higher sampling proportion of longer examples while retaining short examples.

Start B and C from the same checkpoint. Keep objectives and optimization settings aligned where possible, match effective training-token budgets, and report steps, sample coverage, padding, elapsed time, and memory. Equal step counts do not automatically mean equal compute.

If a shared full-data training stage precedes B/C branches, start both branches from the same shared checkpoint and record shared and branch training separately. Do not rely solely on comparisons between models receiving different amounts of training.

Use only training and development data to choose sampling strategies. Do not create long examples by arbitrarily concatenating mismatched text and speech. Report whether improvements on long text sacrifice short-text performance.

## 8. Practical Engineering Comparison

Implement a simple sentence-splitting, separate-synthesis, and concatenation comparison. Compare completeness, elapsed time, boundary pauses, and representative audio.

If sentence splitting solves the problem and training provides no extra benefit, say so directly. If splitting harms continuity, report that as well. Do not assume a winner.

Performance optimizations such as a KV cache are allowed when they meaningfully reduce experiment cost. First verify numerical and generation correctness, report their contribution separately, and do not present them as training improvements.

## 9. Autonomous Overnight Work

After launch is confirmed, independently handle dependency conflicts, code errors, ordinary parameter choices, download retries, reliable public substitute resources, local adaptations, experiment scaling, and abandonment of unproductive trials. Record reasons without repeatedly seeking permission.

Use reasonable timeouts and bounded retries for downloads, installation, and evaluation. Judge training stalls using measured step times and logs. Avoid both premature termination from overly short timeouts and indefinite idle waiting.

Periodically save configurations, logs, random states, checkpoints, and completed evaluations, clearly identifying recovery boundaries. Monitor finite loss, valid updates, and usable output. If one component is unavailable, continue independent work.

Do not bypass approvals or permissions. Only irreplaceable authorization, credentials, or missing information may require the user's involvement; first complete unaffected work and explain the blocker.

If the candidate failure cannot be reproduced, deliver the evidence and state that the hypothesis was not supported. Do not manufacture a problem. Report negative results when a method fails. Propose major changes beyond the confirmed TTS scope before undertaking them.

Use the original 1–2 day budget to control scope and preserve evaluation and reporting time. Do not mark unfinished experiments complete simply because time has elapsed. Follow the actual Goal's state and budget rules; document status does not replace tool state.

## 10. Deliverables and Completion Criteria

Priority: correct baseline → justified modification → fair comparison → independent evaluation → report. Do not begin with a complex UI, large-scale pretraining from scratch, or unsupported architecture rewrites.

Deliver:

1. Clear standalone code with environment and dependency versions.
2. Data/model sources, versions, and preparation instructions for reproduction.
3. Recoverable training workflows, checkpoints, and configurations.
4. Working commands for baseline execution, training, synthesis, and evaluation.
5. Raw records, metrics, charts, and failure cases.
6. Comparative audio and a simple listening page or organized listening directory.
7. A technical report covering the evidence for the problem, method, controls, results, uncertainty, and limitations.
8. A list distinguishing course/upstream material from this project's contributions.
9. Two or three truthful English resume bullets without unverified improvements.
10. An English project explanation and interview guide covering architecture, causal attention, autoregression, teacher forcing, the training objective, decision rationale, failures, and costs. This follows the 2026-10-23 English-file request; the original plan specified Chinese written explanations. Chat discussion remains Chinese by default.

Actually execute the core commands and checks. Negative conclusions can constitute research outcomes; experiments that were not run cannot. Mark the Goal complete only when implementation, experiments, and reporting within the agreed scope are actually finished. No particular method is required to produce an improvement.

## 11. References and Existing Evidence

- [Course repository](https://github.com/Louis0324/eecs183-hw4)
- [Part C](https://github.com/Louis0324/eecs183-hw4/blob/main/hw4-c.ipynb)
- [Data documentation](https://github.com/Louis0324/eecs183-hw4/blob/main/data/README.md)
- [Synthesis helpers](https://github.com/Louis0324/eecs183-hw4/blob/main/hw4_util.py)
- [CosyVoice 2 technical report](https://arxiv.org/abs/2412.10117)
- [NAACL 2025: Robust and Unbounded Length Generalization in Autoregressive Transformer-Based Text-to-Speech](https://aclanthology.org/2025.naacl-long.591/)

Preliminary read-only inspection used course commit `48e1c407dec5a39df1a50a9069934879f175c373`. Pin and record the version actually used at formal launch. Findings in the literature do not establish that the course checkpoint reproduces the same failures.

## 12. Launch Prompt for a New Chat

The user may send the following in a new chat in the same workspace:

> Read `tts-reliability/AGENTS.md` and `tts-reliability/PROJECT_BRIEF.md`. Start the TTS project according to those instructions: first ask together any questions that require my decision; after confirmation, create and start a Goal and independently carry out implementation, training, experiments, and delivery. Start directly if there are no required questions. Do not provide only a plan.
