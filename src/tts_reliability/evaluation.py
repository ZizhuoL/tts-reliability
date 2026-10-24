"""Reproducible local synthesis and ASR evaluation, with resumable evidence.

WER measures recognition disagreement, not perceptual quality. Deletions and
insertions alone do not establish omitted or repeated speech.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata

NORMALIZER_VERSION = "transformers-whisper-EnglishTextNormalizer-unicode-quotes-empty-spelling-map-v2"


@lru_cache(maxsize=1)
def _normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    return EnglishTextNormalizer({})


def normalize_text(text):
    """Use Whisper's English number/contraction normalization for both strings."""
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({
        "’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"',
    }))
    return " ".join(_normalizer()(text).split())


def word_errors(reference, hypothesis):
    """Deterministic minimum edit alignment, retaining individual edit evidence."""
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    cost = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        cost[i][0] = i
    for j in range(len(hyp) + 1):
        cost[0][j] = j
    for i, r in enumerate(ref, 1):
        for j, h in enumerate(hyp, 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (r != h),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    edits, counts = [], Counter()
    i, j = len(ref), len(hyp)
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            operation = "equal" if ref[i - 1] == hyp[j - 1] else "substitution"
            i, j = i - 1, j - 1
            edit = dict(operation=operation, reference=ref[i], hypothesis=hyp[j], reference_index=i, hypothesis_index=j)
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            i -= 1
            operation = "deletion"
            edit = dict(operation=operation, reference=ref[i], hypothesis=None, reference_index=i, hypothesis_index=None)
        else:
            j -= 1
            operation = "insertion"
            edit = dict(operation=operation, reference=None, hypothesis=hyp[j], reference_index=None, hypothesis_index=j)
        counts[operation] += 1
        edits.append(edit)
    errors = sum(counts[k] for k in ("substitution", "deletion", "insertion"))
    return dict(reference_normalized=" ".join(ref), hypothesis_normalized=" ".join(hyp),
                reference_words=len(ref), hypothesis_words=len(hyp),
                substitutions=counts["substitution"], deletions=counts["deletion"],
                insertions=counts["insertion"], correct=counts["equal"], errors=errors,
                wer=errors / len(ref) if ref else None, alignment=list(reversed(edits)))


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def generation_budget(text_tokens, prefix_tokens, context_capacity, *, mode="adaptive", max_length=500, tokens_per_text=11.0):
    """Output allowance and context allowance are separate failure mechanisms.

    The model may predict one last token from a full-capacity prefix. Pass the
    requested allowance into generate, so a binding context cap remains labeled
    context_limit instead of being mislabeled as output_limit.
    """
    if mode not in ("adaptive", "fixed"):
        raise ValueError("Unknown length budget mode")
    requested = max(max_length, math.ceil(tokens_per_text * text_tokens)) if mode == "adaptive" else max_length
    context_available = max(0, context_capacity - prefix_tokens + 1)
    return dict(length_budget_mode=mode, requested_output_budget=requested,
                context_output_budget=context_available,
                effective_output_budget=min(requested, context_available),
                tokens_per_text=tokens_per_text)


def split_sentences(text):
    """Conservative English punctuation heuristic, preserving common titles.

    Abbreviations and initials are ambiguous; this is a documented engineering
    baseline rather than a learned linguistic sentence segmenter.
    """
    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "rev",
                     "hon", "capt", "gen", "lt", "col", "sgt", "vs", "etc", "e.g", "i.e"}
    text = text.strip()
    output, start = [], 0
    for match in re.finditer(r"[.!?]+[\"'”’）)\]]*(?=\s+|$)", text):
        if match.group().startswith("."):
            token = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:match.start()])
            word = token.group(1).lower() if token else ""
            if word in abbreviations or len(word) == 1 or ("." in word and all(len(p) == 1 for p in word.split(".") if p)):
                continue
        output.append(text[start:match.end()].strip())
        start = match.end()
    if text[start:].strip():
        output.append(text[start:].strip())
    return output


def read_texts(path, limit=None):
    texts, identifiers = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{path}:{number}: requires a nonempty text field")
        row = dict(row, id=str(row.get("id", number)))
        if row["id"] in identifiers:
            raise ValueError(f"Duplicate text id: {row['id']}")
        identifiers.add(row["id"])
        texts.append(row)
    return texts if limit is None else texts[:limit]


def append_record(path, record):
    import os
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_records(path, run_id, repair_tail=True):
    """Last event per item wins; only a torn final JSON line may be repaired."""
    path = Path(path)
    if not path.exists():
        return {}
    rows = {}
    data, offset = path.read_bytes(), 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError(f"Corrupt result line {index + 1} in {path}")
        if row.get("run_id") != run_id:
            raise ValueError("Result manifest mismatch; refusing to reuse outputs")
        rows[row["key"]] = row
        offset += len(line)
        if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return rows


def summarize(records):
    rows = list(records)
    scores = [r["metrics"] for r in rows if r.get("status") == "complete"]
    words = sum(s["reference_words"] for s in scores)
    totals = {k: sum(s[k] for s in scores) for k in
              ("errors", "substitutions", "deletions", "insertions", "reference_words")}
    stops = Counter(r.get("stop_reason", "exception") for r in rows)
    totals.update(items=len(rows), scored_items=len(scores),
                  exceptional_items=sum(r.get("status") == "error" for r in rows),
                  wer=totals["errors"] / words if words else None,
                  deletion_rate=totals["deletions"] / words if words else None,
                  insertion_rate=totals["insertions"] / words if words else None,
                  substitution_rate=totals["substitutions"] / words if words else None,
                  stop_reasons=dict(stops),
                  truncation_rate=sum(stops[k] for k in ("output_limit", "context_limit")) / len(rows) if rows else None,
                  exception_rate=sum(r.get("status") == "error" for r in rows) / len(rows) if rows else None,
                  generation_seconds=sum(r.get("generation_seconds", 0) for r in rows),
                  audio_seconds=sum(r.get("audio_seconds", 0) for r in rows),
                  asr_seconds=sum(r.get("asr_seconds", 0) for r in rows))
    return totals


def grouped_summary(records):
    rows = list(records)
    groups, seeds = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row.get("length_group", "unspecified")].append(row)
        seeds[str(row["seed"])].append(row)
    return dict(overall=summarize(rows), by_length={k: summarize(v) for k, v in groups.items()},
                by_seed={k: summarize(v) for k, v in seeds.items()},
                interpretation="WER is ASR disagreement; edits are not human-verified omissions/repetitions. Human listening and MOS have not been performed.")


class LocalEvaluator:
    """LM on chosen device; frozen acoustic stack explicitly defaults to CPU."""
    sample_rate = 24000

    def __init__(self, checkpoint, cosy_dir, whisper_dir, prompt_path, prompt_text,
                 lm_device="mps", audio_device="cpu", asr_device="cpu", threads=8,
                 flow_device=None, vocoder_device=None):
        import torch
        from .model import TextToSpeechLM
        from hw4_util import (load_text_tokenizer, load_speech_tokenizer,
                              load_flow_model, load_vocoder, prompt_features)
        from hw4_runtime import load_audio
        from transformers import pipeline
        torch.set_num_threads(threads)
        self.torch, self.lm_device = torch, lm_device
        self.flow_device = flow_device or audio_device
        self.vocoder_device = vocoder_device or audio_device
        if any(d == "mps" for d in (lm_device, self.flow_device, self.vocoder_device, asr_device)) and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable; explicitly select --lm-device cpu if intended")
        self.tokenizer = load_text_tokenizer(str(cosy_dir))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = state.get("model_state_dict", state.get("state_dict", state))
        config = state.get("model_config") or {}
        config = {k: v for k, v in config.items() if k in ("d_model", "n_heads", "n_layers", "max_seq_len")}
        self.model = TextToSpeechLM(self.tokenizer.vocab_size, 6561, **config)
        self.model.load_state_dict(weights, strict=True)
        self.model = self.model.to(lm_device).eval()
        del state, weights
        self.flow = load_flow_model(str(cosy_dir), device=self.flow_device)
        self.vocoder = load_vocoder(str(cosy_dir), device=self.vocoder_device)
        speech_tokenizer = load_speech_tokenizer(str(cosy_dir), device="cpu")
        audio, _ = load_audio(prompt_path, 16000)
        # Prompt STFT/ONNX stay on CPU even when LM uses MPS.
        features = prompt_features(torch.as_tensor(audio)[None], speech_tokenizer, str(cosy_dir), "cpu")
        self.prompt_tokens, self.prompt_mel, self.embedding = [x.to(self.flow_device) for x in features]
        self.lm_prompt_tokens = self.prompt_tokens.to(lm_device)
        self.prompt_text = torch.tensor([self.tokenizer.encode(prompt_text)], dtype=torch.long, device=lm_device)
        self.asr = pipeline("automatic-speech-recognition", model=str(whisper_dir),
                            tokenizer=str(whisper_dir), feature_extractor=str(whisper_dir),
                            device=asr_device, torch_dtype=torch.float32)
        self.devices = dict(lm=lm_device, flow=self.flow_device, vocoder=self.vocoder_device,
                            prompt_features="cpu", speech_tokenizer="CPUExecutionProvider",
                            speaker_encoder="CPUExecutionProvider", asr=asr_device,
                            reason="Explicit per-component devices: CPU vocoder avoids unsupported MPS iSTFT/complex operators; prompt features use CPU ONNX/STFT. No silent runtime device fallback.")

    def synchronize(self):
        if "mps" in (self.lm_device, self.flow_device, self.vocoder_device):
            self.torch.mps.synchronize()

    def synthesize(self, text, seed, *, max_length, temperature, top_k, min_length_ratio,
                   use_cache, sentence_split=False, pause_seconds=0.15,
                   length_budget="adaptive", tokens_per_text=11.0):
        import numpy as np
        import random
        torch = self.torch
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        if self.lm_device == "mps":
            torch.mps.manual_seed(seed)
        parts = split_sentences(text) if sentence_split else [text]
        chunks, metadata = [], []
        self.last_generation_metadata = metadata
        self.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for index, part in enumerate(parts):
                text_ids = torch.tensor([self.tokenizer.encode(part)], dtype=torch.long, device=self.lm_device)
                budget = generation_budget(text_ids.shape[1],
                    2 + text_ids.shape[1] + self.prompt_text.shape[1] + self.lm_prompt_tokens.shape[1],
                    self.model.position_embedding.num_embeddings, mode=length_budget,
                    max_length=max_length, tokens_per_text=tokens_per_text)
                speech, detail = self.model.generate_with_metadata(
                    text_ids, max_length=budget["requested_output_budget"], temperature=temperature, top_k=top_k,
                    min_length=min(budget["requested_output_budget"], int(text_ids.shape[1] * min_length_ratio)),
                    prompt_speech_token=self.lm_prompt_tokens, prompt_text_tokens=self.prompt_text,
                    use_cache=use_cache)
                self.synchronize()
                detail.update(budget, text=part, segment_index=index)
                metadata.append(detail)
                speech = speech.to(self.flow_device).reshape(1, -1)
                if speech.numel() == 0 or speech.min() < 0 or speech.max() >= 6561:
                    error = ValueError("LM generated empty or invalid speech tokens")
                    error.generation_metadata = metadata
                    raise error
                def length(n):
                    return torch.tensor([n], dtype=torch.int32, device=self.flow_device)
                acoustic_started = time.perf_counter()
                mel, _ = self.flow.inference(token=speech, token_len=length(speech.shape[1]),
                    prompt_token=self.prompt_tokens, prompt_token_len=length(self.prompt_tokens.shape[1]),
                    prompt_feat=self.prompt_mel, prompt_feat_len=length(self.prompt_mel.shape[1]),
                    embedding=self.embedding, streaming=False, finalize=True)
                audio, _ = self.vocoder.inference(speech_feat=mel.to(self.vocoder_device),
                    cache_source=torch.zeros(1, 1, 0, device=self.vocoder_device))
                self.synchronize()
                detail["acoustic_seconds"] = time.perf_counter() - acoustic_started
                audio = audio.flatten().cpu().numpy()
                if not len(audio) or not np.isfinite(audio).all():
                    raise ValueError("Acoustic stack produced invalid audio")
                if chunks and pause_seconds:
                    chunks.append(np.zeros(round(pause_seconds * self.sample_rate), dtype=np.float32))
                chunks.append(audio)
        elapsed = time.perf_counter() - started
        reasons = {m["stop_reason"] for m in metadata}
        reason = next((r for r in ("context_limit", "output_limit", "exception") if r in reasons), "eos")
        return np.concatenate(chunks), dict(segments=metadata, stop_reason=reason,
            generated_tokens=sum(m["generated_tokens"] for m in metadata), generation_seconds=elapsed,
            text_tokens=sum(m["target_text_tokens"] for m in metadata))

    def transcribe(self, audio):
        import torchaudio
        samples = torchaudio.functional.resample(self.torch.as_tensor(audio), self.sample_rate, 16000)
        return self.asr(dict(raw=samples.numpy(), sampling_rate=16000),
                        chunk_length_s=30, return_timestamps=False,
                        generate_kwargs={"language": "en", "task": "transcribe"})["text"]
