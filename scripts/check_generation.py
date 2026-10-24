#!/usr/bin/env python3
"""Compare cached and full-history logits under identical prefix/token histories.

This is an LM numerical diagnostic. The fixed 30-token greedy continuations are
not complete utterances, audio synthesis, or an audio-quality evaluation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "upstream/eecs183-hw4")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from tts_reliability.training import (
    file_sha256, load_model, memory_record, synchronize, write_json)


def compare_logits(reference, actual, atol, rtol):
    """Compare on CPU in float64, retaining absolute and scale-aware errors."""
    reference = reference.detach().cpu().double()
    actual = actual.detach().cpu().double()
    error = (reference - actual).abs()
    relative = error / reference.abs().clamp_min(1e-6)
    reference_top = reference.argmax(-1)
    actual_top = actual.argmax(-1)
    reference_values = reference.topk(2, dim=-1).values
    return dict(
        finite=bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()),
        allclose=bool(torch.allclose(reference, actual, atol=atol, rtol=rtol)),
        max_abs_error=error.max().item(),
        mean_abs_error=error.mean().item(),
        max_relative_error_floor_1e6=relative.max().item(),
        relative_l2_error=(torch.linalg.vector_norm(error)
                           / torch.linalg.vector_norm(reference).clamp_min(1e-12)).item(),
        top1_matches=int((reference_top == actual_top).sum()),
        top1_positions=reference_top.numel(),
        reference_top1=reference_top.flatten().tolist(),
        cached_top1=actual_top.flatten().tolist(),
        reference_min_top1_margin=(reference_values[..., 0]
                                   - reference_values[..., 1]).min().item())


@torch.inference_mode()
def compare_case(model, target_text, prompt_text, prompt_speech, forced_tokens,
                 greedy_tokens=30, seed=183, atol=2e-4, rtol=2e-4):
    device = target_text.device
    prefix = torch.cat((
        model.special_embedding.weight[0:1][None],
        model.text_embedding(torch.cat((prompt_text, target_text), dim=1)),
        model.special_embedding.weight[1:2][None],
        model.speech_embedding(prompt_speech)), dim=1)
    if prefix.shape[1] + max(len(forced_tokens), greedy_tokens) > \
            model.position_embedding.num_embeddings:
        raise ValueError("diagnostic prefix and continuation exceed context capacity")
    started = time.perf_counter()
    complete = prefix
    incremental = prefix
    cache = None
    comparisons = []
    prefill = None
    # Step zero predicts before any forced token; remaining steps compare after
    # feeding the same externally fixed tokens into both paths.
    for step in range(len(forced_tokens) + 1):
        reference = model._logits(complete)
        cached, cache = model._cached_logits(incremental, cache)
        synchronize(device)
        if step == 0:
            prefill = compare_logits(reference, cached, atol, rtol)
            # Avoid hundreds of prefill argmax IDs obscuring next-token checks.
            prefill.pop("reference_top1")
            prefill.pop("cached_top1")
        comparison = compare_logits(reference[:, -1], cached[:, -1], atol, rtol)
        comparison.update(step=step, prefix_and_forced_length=complete.shape[1],
                          previous_forced_token=None if step == 0 else forced_tokens[step - 1])
        comparisons.append(comparison)
        if step < len(forced_tokens):
            next_token = torch.tensor([[forced_tokens[step]]], device=device,
                                      dtype=torch.long)
            incremental = model.speech_embedding(next_token)
            complete = torch.cat((complete, incremental), dim=1)
    forced_seconds = time.perf_counter() - started
    del cache, reference, cached, complete, incremental
    greedy = {}
    for use_cache in (False, True):
        torch.manual_seed(seed)
        if device.type == "mps":
            torch.mps.manual_seed(seed)
        output, metadata = model.generate_with_metadata(
            target_text, prompt_text_tokens=prompt_text,
            prompt_speech_token=prompt_speech, max_length=greedy_tokens,
            min_length=greedy_tokens, top_k=1, temperature=1.0,
            use_cache=use_cache)
        synchronize(device)
        greedy["cached" if use_cache else "full_history"] = dict(
            tokens=output.cpu().flatten().tolist(), metadata=metadata)
    full_tokens = greedy["full_history"]["tokens"]
    cached_tokens = greedy["cached"]["tokens"]
    first_difference = next((index for index, (a, b) in enumerate(
        zip(full_tokens, cached_tokens)) if a != b), None)
    if first_difference is None and len(full_tokens) != len(cached_tokens):
        first_difference = min(len(full_tokens), len(cached_tokens))
    greedy.update(exact_match=full_tokens == cached_tokens,
                  first_different_position=first_difference,
                  interpretation="Fixed-length numerical check; EOS deliberately suppressed. Not a complete utterance.")
    return dict(
        prefix_tokens=prefix.shape[1], target_text_tokens=target_text.shape[1],
        prompt_text_tokens=prompt_text.shape[1], prompt_speech_tokens=prompt_speech.shape[1],
        forced_tokens=forced_tokens, forced_seconds=forced_seconds,
        prefill=prefill, per_step=comparisons, greedy=greedy,
        all_logits_close=prefill["allclose"] and all(row["allclose"] for row in comparisons),
        all_next_token_top1_match=all(row["top1_matches"] == row["top1_positions"]
                                     for row in comparisons),
        max_next_token_abs_error=max(row["max_abs_error"] for row in comparisons),
        **memory_record(device))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/hw4-tts-starter.pt")
    parser.add_argument("--texts", type=Path, default=ROOT / "data/eval/dev.jsonl")
    parser.add_argument("--cosy-dir", type=Path, default=ROOT / "models/cosy")
    parser.add_argument("--prompt", type=Path, default=ROOT / "data/reference/prompt.wav")
    parser.add_argument("--prompt-metadata", type=Path,
                        default=ROOT / "data/reference/prompt.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/generation_checks.json")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--forced-steps", type=int, default=20)
    parser.add_argument("--greedy-tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=183)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    args = parser.parse_args()
    if min(args.forced_steps, args.greedy_tokens, args.threads) < 1:
        parser.error("step/token/thread counts must be positive")
    return args


def main():
    args = arguments()
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; no automatic CPU fallback")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    prompt_metadata = json.loads(args.prompt_metadata.read_text())
    rows = [json.loads(line) for line in args.texts.read_text().splitlines() if line.strip()]
    # First text in each existing dev length group: fixed, no outcome selection.
    selected = []
    for group in ("short", "medium", "long"):
        row = next((row for row in rows if row.get("length_group") == group), None)
        if row is None:
            raise ValueError(f"no dev text for length group {group}")
        selected.append(row)
    from hw4_util import load_text_tokenizer, load_speech_tokenizer, prompt_features
    from hw4_runtime import load_audio
    tokenizer = load_text_tokenizer(str(args.cosy_dir))
    # Use the exact same CPU reference-feature path as LocalEvaluator, including
    # its speech-token/mel-frame alignment trim. No flow/vocoder/ASR is loaded.
    speech_tokenizer = load_speech_tokenizer(str(args.cosy_dir), "cpu")
    waveform, _ = load_audio(args.prompt, 16000)
    features = prompt_features(torch.as_tensor(waveform)[None], speech_tokenizer,
                               str(args.cosy_dir), "cpu")
    prompt_speech = features[0].long().to(device)
    prompt_text = torch.tensor([tokenizer.encode(prompt_metadata["text"])],
                               dtype=torch.long, device=device)
    # A priori fixed valid speech history from the reference, independent of
    # either decoder's outputs, prevents divergence from changing the comparison.
    reference_ids = prompt_speech.cpu().flatten().tolist()
    forced = [reference_ids[index % len(reference_ids)] for index in range(args.forced_steps)]
    del features, speech_tokenizer, waveform
    gc.collect()
    model, state = load_model(args.checkpoint, device, args.cosy_dir)
    model.eval()
    del state
    report = dict(
        kind="cache_numerical_diagnostic", status="running",
        settings={key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()},
        torch_version=str(torch.__version__), checkpoint_sha256=file_sha256(args.checkpoint),
        model_source_sha256=file_sha256(ROOT / "src/tts_reliability/model.py"),
        script_sha256=file_sha256(Path(__file__)), texts_sha256=file_sha256(args.texts),
        prompt_sha256=file_sha256(args.prompt),
        prompt_metadata_sha256=file_sha256(args.prompt_metadata),
        prompt_metadata=prompt_metadata,
        forced_token_policy="First reference speech tokens, repeated only if needed; fixed before any LM comparisons.",
        prompt_token_sha256=hashlib.sha256(json.dumps(reference_ids).encode()).hexdigest(),
        samples=[])
    write_json(args.output, report)
    for row in selected:
        print(f"Checking {row['length_group']}: {row['id']}", flush=True)
        target = torch.tensor([tokenizer.encode(row["text"])], dtype=torch.long, device=device)
        result = compare_case(
            model, target, prompt_text, prompt_speech, forced,
            args.greedy_tokens, args.seed, args.atol, args.rtol)
        result.update(id=row["id"], text=row["text"], length_group=row["length_group"])
        report["samples"].append(result)
        write_json(args.output, report)
        print(json.dumps(dict(id=row["id"], max_abs_error=result["max_next_token_abs_error"],
                              all_logits_close=result["all_logits_close"],
                              all_top1_match=result["all_next_token_top1_match"],
                              greedy_exact_match=result["greedy"]["exact_match"])), flush=True)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    report["status"] = "complete"
    report["all_logits_close"] = all(row["all_logits_close"] for row in report["samples"])
    report["all_greedy_exact_match"] = all(row["greedy"]["exact_match"] for row in report["samples"])
    report["interpretation"] = (
        "These checks address cache numerical correctness for the tested contexts. "
        "They do not establish audio quality, full-utterance equality, or explain WER by themselves.")
    write_json(args.output, report)
    print(f"Saved {args.output}", flush=True)
    return 0 if report["all_logits_close"] and report["all_greedy_exact_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
