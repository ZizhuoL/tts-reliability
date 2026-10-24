#!/usr/bin/env python3
"""Evaluate local TTS; all model assets must already exist on disk."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import resource
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "upstream/eecs183-hw4")]
# Never trigger implicit network model fetches during evaluation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tts_reliability.evaluation import (NORMALIZER_VERSION, LocalEvaluator, append_record,
    content_hash, grouped_summary, read_records, read_texts, sha256_file, word_errors)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Starter or locally saved checkpoint")
    parser.add_argument("--texts", type=Path, required=True, help="JSONL: text, optional id, length_group and provenance fields")
    parser.add_argument("--output", type=Path, required=True, help="Unique directory per configuration; identical runs resume")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--split-sentences", action="store_true", help="Synthesize punctuation-separated sentences with fixed silence gaps")
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N texts for smoke checks")
    parser.add_argument("--cosy-dir", type=Path, default=ROOT / "models/cosy")
    parser.add_argument("--whisper-dir", type=Path, default=ROOT / "models/whisper")
    parser.add_argument("--prompt", type=Path, default=ROOT / "data/reference/prompt.wav")
    parser.add_argument("--prompt-metadata", type=Path, default=ROOT / "data/reference/prompt.json", help="JSON containing exact text and source/license provenance")
    parser.add_argument("--lm-device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--audio-device", choices=["mps", "cpu"], help="Override BOTH frozen flow/vocoder devices")
    parser.add_argument("--flow-device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--vocoder-device", choices=["mps", "cpu"], default="cpu", help="CPU avoids unsupported MPS complex iSTFT")
    parser.add_argument("--asr-device", choices=["mps", "cpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--length-budget", choices=["adaptive", "fixed"], default="adaptive", help="adaptive = max(max-length, ceil(tokens-per-text * target text tokens))")
    parser.add_argument("--max-length", type=int, default=500, help="Fixed NEW-token limit, or adaptive minimum allowance per segment; context capacity separately enforced")
    parser.add_argument("--tokens-per-text", type=float, default=11.0, help="Adaptive allowance ratio; train-data speech/text p99 is about 10.5")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--min-length-ratio", type=float, default=2.0, help="Course minimum speech length heuristic per target text token")
    parser.add_argument("--no-lm-prompt", action="store_true", help="Diagnostic: omit LM reference text/speech prefix while retaining identical acoustic reference features")
    parser.add_argument("--no-cache", action="store_true", help="Disable validated LM KV cache")
    parser.add_argument("--retry-errors", action="store_true", help="Retry previously failed items; keep successful evidence")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.tokens_per_text <= 0 or args.max_length <= 0 or args.top_k < 0 or args.temperature <= 0 or args.min_length_ratio < 0 or args.pause_seconds < 0 or args.threads <= 0:
        parser.error("Invalid generation/threads bounds")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    return args


def asset_fingerprint(folder):
    # Hash the actual assets, not names or mutable download timestamps.
    paths = [p for p in folder.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(folder).parts)]
    if not paths:
        raise ValueError(f"No assets found: {folder}")
    return {str(p.relative_to(folder)): sha256_file(p) for p in sorted(paths)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another evaluation owns this output directory")
        return evaluate(args)


def evaluate(args):
    import soundfile as sf
    import numpy as np
    prompt_info = json.loads(args.prompt_metadata.read_text())
    if not isinstance(prompt_info.get("text"), str) or not prompt_info["text"].strip():
        raise ValueError("Prompt metadata must contain exact nonempty text")
    texts = read_texts(args.texts, args.limit)
    if not texts:
        raise ValueError("No evaluation texts")
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("output", "retry_errors")}
    sources = [Path(__file__), ROOT / "src/tts_reliability/evaluation.py", ROOT / "src/tts_reliability/model.py",
               ROOT / "upstream/eecs183-hw4/hw4_util.py", ROOT / "upstream/eecs183-hw4/hw4_runtime.py"]
    print("Hashing checkpoints/assets to bind resumable results to exact inputs...", flush=True)
    manifest = dict(format_version=1, settings=settings, normalizer=NORMALIZER_VERSION,
        checkpoint_sha256=sha256_file(args.checkpoint), texts_sha256=sha256_file(args.texts),
        selected_texts_sha256=content_hash(texts), prompt_sha256=sha256_file(args.prompt),
        prompt_metadata_sha256=sha256_file(args.prompt_metadata), prompt_metadata=prompt_info,
        cosy_assets=asset_fingerprint(args.cosy_dir), whisper_assets=asset_fingerprint(args.whisper_dir),
        implementation={str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
        packages={name: importlib.metadata.version(name) for name in
                  ("torch", "torchaudio", "transformers", "numpy", "soundfile", "onnxruntime")})
    run_id = content_hash(manifest)
    manifest["run_id"] = run_id
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Output configuration/assets/code differ. Choose a new --output directory; refusing stale-result reuse.")
    else:
        if (args.output / "raw.jsonl").exists():
            raise ValueError("raw.jsonl exists without a manifest; refusing ambiguous reuse")
        write_json(manifest_path, manifest)
    raw_path = args.output / "raw.jsonl"
    records = read_records(raw_path, run_id)
    jobs = [(row, seed, content_hash(dict(id=row["id"], seed=seed))[:24]) for row in texts for seed in args.seeds]
    pending = [(row, seed, key) for row, seed, key in jobs if key not in records or
               records[key].get("status") == "synthesized" or
               (records[key].get("status") == "error" and args.retry_errors)]
    if not pending:
        write_json(args.output / "summary.json", grouped_summary(records.values()))
        print(f"Already finished: {len(records)} records; {args.output / 'summary.json'}", flush=True)
        return
    evaluator = LocalEvaluator(args.checkpoint, args.cosy_dir, args.whisper_dir, args.prompt,
        prompt_info["text"], args.lm_device, args.audio_device or "cpu", args.asr_device, args.threads,
        flow_device=args.audio_device or args.flow_device, vocoder_device=args.audio_device or args.vocoder_device)
    if args.no_lm_prompt:
        evaluator.prompt_text = evaluator.prompt_text[:, :0]
        evaluator.lm_prompt_tokens = evaluator.lm_prompt_tokens[:, :0]
        evaluator.devices["lm_prompt_enabled"] = False
    write_json(args.output / "runtime.json", dict(devices=evaluator.devices, run_id=run_id))
    (args.output / "wav").mkdir(exist_ok=True)
    for index, (row, seed, key) in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {row['id']} seed={seed}", flush=True)
        started = time.perf_counter()
        previous = records.get(key, {})
        record = dict(run_id=run_id, key=key, id=row["id"], text=row["text"], text_sha256=content_hash(row["text"]),
                      text_metadata=row, length_group=row.get("length_group", "unspecified"), seed=seed,
                      checkpoint_sha256=manifest["checkpoint_sha256"], prompt_sha256=manifest["prompt_sha256"],
                      sampling=dict(max_length=args.max_length, temperature=args.temperature, top_k=args.top_k,
                                    min_length_ratio=args.min_length_ratio, use_cache=not args.no_cache,
                                    length_budget=args.length_budget, tokens_per_text=args.tokens_per_text),
                      split_sentences=args.split_sentences, lm_prompt_enabled=not args.no_lm_prompt, devices=evaluator.devices)
        wav_path = args.output / "wav" / f"{key}.wav"
        stage = "synthesis"
        try:
            if previous.get("wav_sha256") and wav_path.exists():
                if sha256_file(wav_path) != previous["wav_sha256"]:
                    raise ValueError("Saved waveform changed; refusing ASR-only resume")
                record = {k: v for k, v in previous.items() if k not in ("error", "traceback", "error_stage")}
                audio, sample_rate = sf.read(wav_path, dtype="float32")
                if sample_rate != evaluator.sample_rate:
                    raise ValueError("Saved waveform sample rate mismatch")
            else:
                audio, detail = evaluator.synthesize(row["text"], seed,
                    max_length=args.max_length, temperature=args.temperature, top_k=args.top_k,
                    min_length_ratio=args.min_length_ratio, use_cache=not args.no_cache,
                    sentence_split=args.split_sentences, pause_seconds=args.pause_seconds,
                    length_budget=args.length_budget, tokens_per_text=args.tokens_per_text)
                record.update(detail)
                temporary = wav_path.with_suffix(".tmp.wav")
                sf.write(temporary, audio, evaluator.sample_rate, subtype="PCM_16")
                temporary.replace(wav_path)
                record.update(wav=str(wav_path.relative_to(args.output)), wav_sha256=sha256_file(wav_path),
                              audio_seconds=len(audio) / evaluator.sample_rate,
                              peak_amplitude=float(np.abs(audio).max()),
                              clipped_sample_fraction=float((np.abs(audio) >= 1).mean()), status="synthesized")
                append_record(raw_path, record)
                # Score the delivered PCM file, exactly as on recovery.
                audio, _ = sf.read(wav_path, dtype="float32")
            stage = "asr"
            asr_started = time.perf_counter()
            transcript = evaluator.transcribe(audio)
            record.update(asr_seconds=time.perf_counter() - asr_started, asr_transcript=transcript,
                          metrics=word_errors(row["text"], transcript), status="complete")
        except Exception as error:
            record.update(status="error", error_stage=stage, error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc())
            generation_metadata = getattr(error, "generation_metadata",
                getattr(evaluator, "last_generation_metadata", []) if stage == "synthesis" else [])
            if generation_metadata:
                record["segments"] = generation_metadata
                record["stop_reason"] = generation_metadata[-1]["stop_reason"]
                record["generated_tokens"] = sum(m["generated_tokens"] for m in generation_metadata)
            record.setdefault("stop_reason", "exception")
            print(record["error"], flush=True)
        record["attempt_seconds"] = time.perf_counter() - started
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["process_peak_rss_bytes"] = max_rss if sys.platform == "darwin" else max_rss * 1024
        if args.lm_device == "mps" or args.flow_device == "mps":
            import torch
            if torch.backends.mps.is_available():
                record["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
                record["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        append_record(raw_path, record)
        records[key] = record
        write_json(args.output / "summary.json", grouped_summary(records.values()))
    print(f"Saved {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
