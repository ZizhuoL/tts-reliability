#!/usr/bin/env python3
"""Check reference ASR and paired ground-truth tokens through frozen acoustics.

This does not load or score a text-to-speech language model. The paired tokens
come only from the development cache, never from the sealed final text set.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "upstream/eecs183-hw4")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tts_reliability.evaluation import (NORMALIZER_VERSION, append_record, content_hash,
    read_records, sha256_file, word_errors)
from evaluate import asset_fingerprint, write_json


def select_sample(samples, minimum_text_tokens=10):
    """Deterministic closest-to-global-median selection; ties preserve file order."""
    import statistics
    median = statistics.median(len(row["speech_token"]) for row in samples)
    eligible = [(index, row) for index, row in enumerate(samples)
                if len(row["text_token"]) >= minimum_text_tokens and len(row["speech_token"]) > 0]
    if not eligible:
        raise ValueError("No paired dev sample satisfies text-token minimum")
    index, row = min(eligible, key=lambda item: (abs(len(item[1]["speech_token"]) - median), item[0]))
    return index, row, median


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/dev.pt", help="Paired development token cache only")
    parser.add_argument("--output", type=Path, default=ROOT / "results/audio_component_checks")
    parser.add_argument("--prompt", type=Path, default=ROOT / "data/reference/prompt.wav")
    parser.add_argument("--prompt-metadata", type=Path, default=ROOT / "data/reference/prompt.json")
    parser.add_argument("--cosy-dir", type=Path, default=ROOT / "models/cosy")
    parser.add_argument("--whisper-dir", type=Path, default=ROOT / "models/whisper")
    parser.add_argument("--flow-device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--vocoder-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--asr-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if "dev" not in args.data.name.lower() or "final" in str(args.data).lower():
        parser.error("--data must name a development cache, not training/final data")
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


def run(args):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import pipeline
    from hw4_runtime import load_audio
    from hw4_util import load_flow_model, load_vocoder, load_speech_tokenizer, prompt_features
    torch.set_num_threads(args.threads)
    if "mps" in (args.flow_device, args.vocoder_device, args.asr_device) and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; select --flow-device cpu explicitly")
    prompt_info = json.loads(args.prompt_metadata.read_text())
    if not prompt_info.get("text"):
        raise ValueError("Exact prompt transcript is required")
    print("Loading paired development cache and hashing assets...", flush=True)
    cache = torch.load(args.data, map_location="cpu", weights_only=True)
    index, sample, median = select_sample(cache["samples"])
    speech = sample["speech_token"].long().reshape(1, -1)
    if (speech < 0).any() or (speech >= 6561).any():
        raise ValueError("Ground-truth speech tokens outside the frozen decoder vocabulary")
    selection = dict(processed_index=index, original_index=sample.get("original_index"),
        utt=sample["utt"], text=sample["text"], speaker_id=sample["speaker_id"],
        text_tokens=len(sample["text_token"]), speech_tokens=speech.numel(),
        speech_token_ids=speech.flatten().tolist(),
        global_dev_speech_length_median=median,
        selection="Closest global dev speech-length median among text_token length >= 10; first file index breaks ties")
    manifest = dict(format_version=1, purpose="Reference ASR and frozen token-to-audio component checks; NOT LM quality",
        normalizer=NORMALIZER_VERSION, seed=args.seed, selection=selection,
        sourcefile=str(args.data.resolve()), sourcefile_sha256=sha256_file(args.data),
        prompt_sha256=sha256_file(args.prompt), prompt_metadata=prompt_info,
        prompt_metadata_sha256=sha256_file(args.prompt_metadata),
        cosy_assets=asset_fingerprint(args.cosy_dir), whisper_assets=asset_fingerprint(args.whisper_dir),
        devices=dict(flow=args.flow_device, vocoder=args.vocoder_device, asr=args.asr_device,
                     prompt_features="cpu", speech_tokenizer="CPUExecutionProvider"),
        implementation_sha256=sha256_file(Path(__file__)))
    run_id = content_hash(manifest)
    manifest["run_id"] = run_id
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Existing output belongs to another component-check configuration; choose --output")
    write_json(path, manifest)
    del cache
    records = read_records(args.output / "raw.jsonl", run_id)
    if all(records.get(key, {}).get("status") == "complete" for key in ("prompt_asr", "groundtruth_decode")):
        print("Both component checks already complete", flush=True)
        return
    asr = pipeline("automatic-speech-recognition", model=str(args.whisper_dir), tokenizer=str(args.whisper_dir),
        feature_extractor=str(args.whisper_dir), device=args.asr_device, torch_dtype=torch.float32)

    def transcribe(audio, rate):
        if rate != 16000:
            audio = torchaudio.functional.resample(torch.as_tensor(audio), rate, 16000).numpy()
        return asr(dict(raw=audio, sampling_rate=16000), chunk_length_s=30,
                   generate_kwargs={"language": "en", "task": "transcribe"})["text"]

    def save(record):
        record.update(run_id=run_id, seed=args.seed, sourcefile_sha256=manifest["sourcefile_sha256"],
                      prompt_sha256=manifest["prompt_sha256"], devices=manifest["devices"], normalizer=NORMALIZER_VERSION)
        append_record(args.output / "raw.jsonl", record)
        records[record["key"]] = record
        write_json(args.output / "results.json", dict(checks=records,
            interpretation="Ground-truth speech-token reconstruction and prompt ASR are component checks, not LM performance or human listening."))

    if records.get("prompt_asr", {}).get("status") != "complete":
        prompt_audio, rate = load_audio(args.prompt, 16000)
        started = time.perf_counter()
        hypothesis = transcribe(prompt_audio, rate)
        save(dict(key="prompt_asr", status="complete", text=prompt_info["text"], asr_transcript=hypothesis,
            metrics=word_errors(prompt_info["text"], hypothesis), asr_seconds=time.perf_counter() - started,
            audio_seconds=len(prompt_audio) / rate, wav=str(args.prompt.resolve()), wav_sha256=sha256_file(args.prompt),
            source=prompt_info.get("source_url"), speaker_id=prompt_info.get("id")))
        print("Prompt transcription:", hypothesis, flush=True)
    if records.get("groundtruth_decode", {}).get("status") == "complete":
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    random.seed(args.seed)
    if "mps" in (args.flow_device, args.vocoder_device):
        torch.mps.manual_seed(args.seed)
    wav = args.output / "groundtruth_decode.wav"
    existing = records.get("groundtruth_decode", {})
    if existing.get("wav_sha256") and wav.exists():
        if sha256_file(wav) != existing["wav_sha256"]:
            raise ValueError("Saved ground-truth waveform changed; choose another output directory")
        record = dict(existing)
        audio, rate = sf.read(wav, dtype="float32")
    else:
        print("Loading frozen flow and vocoder (no LM)...", flush=True)
        flow = load_flow_model(str(args.cosy_dir), args.flow_device)
        vocoder = load_vocoder(str(args.cosy_dir), args.vocoder_device)
        tokenizer = load_speech_tokenizer(str(args.cosy_dir), "cpu")
        reference, _ = load_audio(args.prompt, 16000)
        features = prompt_features(torch.as_tensor(reference)[None], tokenizer, str(args.cosy_dir), "cpu")
        prompt_tokens, prompt_mel, embedding = [x.to(args.flow_device) for x in features]
        speech = speech.to(args.flow_device)
        def length(n):
            return torch.tensor([n], dtype=torch.int32, device=args.flow_device)
        if "mps" in (args.flow_device, args.vocoder_device):
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            mel, _ = flow.inference(token=speech, token_len=length(speech.shape[1]), prompt_token=prompt_tokens,
                prompt_token_len=length(prompt_tokens.shape[1]), prompt_feat=prompt_mel,
                prompt_feat_len=length(prompt_mel.shape[1]), embedding=embedding, streaming=False, finalize=True)
            waveform, _ = vocoder.inference(speech_feat=mel.to(args.vocoder_device),
                cache_source=torch.zeros(1, 1, 0, device=args.vocoder_device))
        if "mps" in (args.flow_device, args.vocoder_device):
            torch.mps.synchronize()
        acoustic_seconds = time.perf_counter() - started
        audio, rate = waveform.flatten().cpu().numpy(), 24000
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Frozen decoder returned invalid waveform")
        temporary = wav.with_suffix(".tmp.wav")
        sf.write(temporary, audio, rate, subtype="PCM_16")
        temporary.replace(wav)
        record = dict(key="groundtruth_decode", status="synthesized", **selection,
            acoustic_seconds=acoustic_seconds, audio_seconds=len(audio) / rate,
            wav=str(wav.resolve()), wav_sha256=sha256_file(wav),
            prompt_speech_tokens=prompt_tokens.shape[1], prompt_mel_frames=prompt_mel.shape[1],
            conditioning="Fixed public reference acoustics; original paired dev speech tokens retained exactly; no LM generation")
        save(record)
        audio, rate = sf.read(wav, dtype="float32")
    started = time.perf_counter()
    hypothesis = transcribe(audio, rate)
    record.update(status="complete", asr_transcript=hypothesis, asr_seconds=time.perf_counter() - started,
                  metrics=word_errors(selection["text"], hypothesis))
    save(record)
    print("Ground-truth token transcription:", hypothesis, flush=True)


if __name__ == "__main__":
    main()
