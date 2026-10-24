# TTS Reliability: Research Basis and Data Candidates

Verification date: 2026-10-19 (America/Los_Angeles). This document distinguishes literature findings, reviewed course documentation, and hypotheses to test. It does not represent completed project training or evaluation results.

## 1. What the Literature Supports

### CosyVoice 2

[Original technical report v3](https://arxiv.org/html/2412.10117v3), 2024-12-25; see especially §2.2–2.4.

The report describes a 25 tokens/s speech tokenizer, an autoregressive text-to-speech LM, and downstream flow matching and vocoder waveform reconstruction. Nonstreaming ICL uses reference text, target text, and a fixed reference speech-token prefix, then generates until EOS. This motivates checking prefix boundaries, target shifts, EOS, and token duration. Attribute reused frozen components explicitly: the course's custom small model is not the complete CosyVoice 2 system, and the project does not train the entire system from scratch.

The report neither establishes a length-robustness defect in the course checkpoint nor guarantees that course-generated audio duration exactly equals token count divided by 25. Measure actual audio duration.

### Very Attentive Tacotron

[Original NAACL 2025 paper](https://aclanthology.org/2025.naacl-long.591.pdf), Battenberg et al.; see §3, §4.2, §5.3, and Limitations.

The paper observes repetition and omissions on long inputs in its T5-style encoder-decoder TTS model and improves generalization with learnable monotonic alignment positions and relative cross-attention positional information. Training audio is at most 9.6 seconds; length diagnostics use 1,034 texts of 100–1,500 characters. Naturalness is evaluated separately. The alignment module incurs a training-speed cost.

This motivates research; it is not a finding about the course model. The method depends on encoder-decoder cross-attention and cannot be treated as a checkpoint-compatible patch for the course's decoder-only model. Its beyond-training-length stress range should not be copied simply to force the course model to fail.

## 2. Reviewed Course Information and Open Questions

Local `upstream/eecs183-hw4/data/README.md` describes the public Part C release as `hw4-2026-v4`, with 354,780 training records and 9,756 validation records; the starter contains two-epoch weights for the original architecture. It states that validation excludes 201 records corresponding to 200 course final questions. These are course descriptions: actual valid counts, filtering, and checkpoint history still require verification.

The independent project needs its own final set. Course validation cannot be presented as a final test never used for model selection. If the starter's full selection history and frozen upstream components' pretraining coverage are unknown, retain those limitations.

## 3. Fixed Reference Speech: A Small Available Download

Recommended candidate: LibriSpeech `dev-clean` utterance `1272-128104-0000`, mirrored in the [Hugging Face internal-testing dummy dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy). Its corpus transcript is:

> MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL

The provenance and license source is [OpenSLR SLR12](https://www.openslr.org/12), which lists LibriSpeech under CC BY 4.0. Attribute Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, LibriSpeech (ICASSP 2015). The historical HF script points directly to that corpus. Do not confuse the download script's Apache 2.0 code license with the recording license.

Pinned download metadata from the [HF revision](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/commit/5be91486e11a2d616f4ec5db8d3fd248585ac07a):

| Field | Value |
| --- | --- |
| repo_id | `hf-internal-testing/librispeech_asr_dummy` |
| repo_type | `dataset` |
| revision | `5be91486e11a2d616f4ec5db8d3fd248585ac07a` |
| File | `clean/validation-00000-of-00001.parquet` |
| Size | 9,192,059 bytes |
| SHA256 | `4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603` |
| Records | 73 |
| Original sample rate | 16,000 Hz |

[Pinned download](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/5be91486e11a2d616f4ec5db8d3fd248585ac07a/clean/validation-00000-of-00001.parquet).

Download with `hf_hub_download`, read by `id` with `pyarrow.parquet`, extract `audio.bytes`, and decode with `soundfile`. Avoid adding datasets audio-decoding dependencies solely for this reference. Preserve the download, original transcript, sample ID, and original audio hash; record explicit resampling configuration if needed. Initial selection should depend only on availability, intact audio, suitable duration, and transcript consistency, never final-result-based voice selection.

This is a fixed reference condition, not a speaker-generalization benchmark. HF example records include local-path strings that are historical metadata and should not be accessed; use embedded audio bytes. Upstream models may have seen public recordings, so do not claim a zero-shot unseen speaker.

## 4. Natural Development Texts and an Independent Final Set

Small available text sources:

| Source | Record | Plain text |
| --- | --- | --- |
| Lewis Carroll, Alice's Adventures in Wonderland | [Gutenberg #11](https://www.gutenberg.org/ebooks/11) | [pg11.txt](https://www.gutenberg.org/cache/epub/11/pg11.txt) |
| Arthur Conan Doyle, The Adventures of Sherlock Holmes | [Gutenberg #1661](https://www.gutenberg.org/ebooks/1661) | [pg1661.txt](https://www.gutenberg.org/cache/epub/1661/pg1661.txt) |

Both pages identify their texts as public domain in the United States. Retain downloaded originals, header/footer license text, retrieval dates, SHA256, and provenance. Select natural prose sentences or consecutive original sentences, excluding contents lists, poetry layouts, illustration labels, and webpage summaries. These works may appear in LibriSpeech/LibriTTS or LM pretraining; independence claims apply only to verifiable course training data.

Before viewing any generation, fix nonoverlapping development/final regions by work or chapter and sample each by length. Cover the same works or similar text types in both sets so that a development set entirely from Alice and a final set entirely from Holmes do not confound domain and method effects. Keep final generations unopened until methods, thresholds, and checkpoints are frozen.

Deduplication should include whole-sentence exact matching after case/punctuation/whitespace normalization, constituent-sentence checks for multisentence examples, and word n-gram overlap review for long texts. Hashing whole paragraphs alone cannot detect paragraphs assembled from training sentences. Retain excluded IDs, reasons, similarity rules, and sources; do not imply coverage of inaccessible upstream pretraining data.

If too few literary examples remain, add other public natural-text sources. A few original everyday reading examples can broaden coverage but should not dominate formal testing. Do not repeat sentences or join nonconsecutive source passages merely to increase length.

## 5. Proposed Preregistered Diagnostics

These are project-design recommendations, not course-model conclusions established by the papers above.

1. Measure valid training text tokens, speech tokens, total context, and token ratios. Report p10/p50/p75/p90/p95/p99 and actual limits. Character counts aid readability; model budgets require actual tokenizer counts.
2. Start with 36–48 development texts, 12–16 in each short/medium/long group. Define disjoint groups from training-length quantiles; use the upper supported range before extreme outliers. Save thresholds before generation.
3. Estimate output caps from training-ratio distributions and development data while reserving context for reference prefixes, text, and special tokens. Record predicted budgets and actual lengths. Report cap stops separately from voluntary EOS. Correcting a cap is a baseline repair, not a training contribution.
4. Save raw/normalized ASR, WER S/D/I, word alignments, stopping reasons, durations, and token counts. Elevated deletion rates are candidate evidence; attribute omissions only with stopping positions, alignments, and verifiable audio. ASR cannot replace naturalness or speaker-similarity evaluation.
5. Use identical texts, reference audio, sampling settings, and paired seeds for key comparisons. Report each length group's sample count, word-weighted WER, error components, and uncertainty. Bootstrap by text cluster across multiple seeds rather than treating each seed as an independent text.
6. Retain failures, exceptions, and truncations instead of calculating success rates on successful generations only. Count ASR and synthesis failures separately. Publish explicit rules governing which references enter the WER denominator.

## 6. Conditional Method Priorities

| Observed evidence | Candidate to test | Fair comparison and stopping rule |
| --- | --- | --- |
| Most failures come from hard output caps | Reasonable length budgets and context guards | Correct all baselines first; acknowledge an engineering limitation if the problem disappears. |
| Longer supported texts perform worse and receive less training coverage | Increase long-sample probability while retaining short samples | Branch B (original distribution) and C (long-sample weighting) from one checkpoint; match valid speech-target token budgets and report coverage/compute. |
| A reliable baseline has cross-sentence completeness problems | Synthesize sentences separately and concatenate | Same texts, reference, seeds, and ASR; report runtime/boundary pauses and provide whole/split audio. |
| EOS occurs early after excluding shift/mask errors | Analyze EOS probability and target statistics by length, then consider objective changes | Suppressing EOS can cause repetition or cap failures; distinguish this from better length rules. |
| Recomputing full history dominates inference | Independent KV-cache optimization | Verify logits and generation equivalence for fixed inputs; do not call it a quality-training contribution. |

Prefer a weight-reusing data-sampling approach only if diagnostics support its mechanism. Literature alone does not justify a complex alignment architecture; VAT is related research, not an available project ablation. If no reproducible length problem emerges, or continuation has no stable benefit, deliver the negative result with the original hypothesis, raw data, and uncertainty intact.
