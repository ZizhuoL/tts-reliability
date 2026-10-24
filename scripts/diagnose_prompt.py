#!/usr/bin/env python3
"""Controlled prompt-format diagnosis and offline v2 rescoring.

Default synthesis: first three short, medium, long development texts, two seeds,
three prompt conditions. Never use sealed final texts for this diagnosis.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tts_reliability.evaluation import (NORMALIZER_VERSION, content_hash, grouped_summary,
    read_texts, sha256_file, word_errors)
from evaluate import write_json, asset_fingerprint

SENTENCE_CASE = "Mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."


def snapshot_results(folder):
    """Read a consistent byte snapshot, tolerating only an incomplete final event."""
    data = (folder / "raw.jsonl").read_bytes()
    rows = {}
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise
        rows[row["key"]] = row
    return rows, hashlib.sha256(data).hexdigest()


def rescore(source, destination):
    """Retain raw text/audio evidence; change only explicitly versioned scoring."""
    rows, digest = snapshot_results(source)
    manifest = json.loads((source / "manifest.json").read_text())
    provenance = dict(source_run_id=manifest["run_id"], source_raw_sha256=digest,
                      normalizer=NORMALIZER_VERSION, source_directory=str(source.resolve()))
    run_id = content_hash(provenance)
    rescored = []
    for row in rows.values():
        row = dict(row, source_run_id=row["run_id"], run_id=run_id, normalizer=NORMALIZER_VERSION)
        if row.get("status") == "complete":
            row["previous_metrics"] = row["metrics"]
            row["metrics"] = word_errors(row["text"], row["asr_transcript"])
        rescored.append(row)
    destination.mkdir(parents=True, exist_ok=True)
    # Separate filenames ensure original raw.jsonl and metrics remain untouched.
    raw_path = destination / "raw-v2.jsonl"
    temporary = raw_path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rescored))
    temporary.replace(raw_path)
    write_json(destination / "manifest-v2.json", dict(provenance, run_id=run_id))
    write_json(destination / "summary-v2.json", grouped_summary(rescored))
    print(f"Rescored {len(rescored)} latest events: {raw_path}", flush=True)
    return rescored


def import_original(source, output, selected, seeds, checkpoint, prompt, prompt_metadata):
    """Reuse only complete original-condition rows from matching model/config."""
    if not (source / "manifest.json").exists() or not (source / "raw.jsonl").exists():
        return False
    source_manifest = json.loads((source / "manifest.json").read_text())
    config = source_manifest["settings"]
    required = dict(length_budget="adaptive", max_length=500, tokens_per_text=11.0,
                    temperature=1.0, top_k=50, min_length_ratio=2.0,
                    split_sentences=False, no_cache=False, lm_device="mps", flow_device="mps",
                    vocoder_device="cpu", asr_device="cpu")
    if any(config.get(k) != v for k, v in required.items()) or config.get("no_lm_prompt", False) or config.get("audio_device") is not None:
        raise ValueError("Original result generation configuration does not match controlled diagnosis")
    if source_manifest["checkpoint_sha256"] != sha256_file(checkpoint) or source_manifest["prompt_sha256"] != sha256_file(prompt):
        raise ValueError("Original result checkpoint/reference audio mismatch")
    if source_manifest["prompt_metadata"]["text"] != json.loads(prompt_metadata.read_text())["text"]:
        raise ValueError("Original prompt transcript differs")
    if source_manifest["implementation"]["src/tts_reliability/model.py"] != sha256_file(ROOT / "src/tts_reliability/model.py"):
        raise ValueError("Original result LM implementation differs")
    for component in ("cosy", "whisper"):
        if source_manifest[f"{component}_assets"] != asset_fingerprint(ROOT / "models" / component):
            raise ValueError(f"Original {component} assets differ")
    for package, version in source_manifest["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Original package version differs: {package}")
    for relative in ("upstream/eecs183-hw4/hw4_util.py", "upstream/eecs183-hw4/hw4_runtime.py"):
        if source_manifest["implementation"][relative] != sha256_file(ROOT / relative):
            raise ValueError(f"Original acoustic/helper implementation differs: {relative}")
    rows, source_hash = snapshot_results(source)
    if any(row["run_id"] != source_manifest["run_id"] for row in rows.values()):
        raise ValueError("Original raw results do not match their manifest")
    by_id = {(r["id"], r["seed"]): r for r in rows.values()}
    wanted = [(text, seed, by_id.get((text["id"], seed))) for text in selected for seed in seeds]
    if any(row is None or row.get("status") != "complete" for _, _, row in wanted):
        print("Original source has not completed the selected subset; original condition will be synthesized independently.", flush=True)
        return False
    manifest = dict(condition="original", imported=True, normalizer=NORMALIZER_VERSION,
        source_run_id=source_manifest["run_id"], source_raw_sha256=source_hash,
        source_manifest=source_manifest, selected_texts_sha256=content_hash(selected), seeds=seeds)
    manifest["run_id"] = content_hash(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "wav").mkdir(exist_ok=True)
    selected_rows = []
    for text, seed, row in wanted:
        if row["text"] != text["text"]:
            raise ValueError("Reused original text differs")
        wav = source / row["wav"]
        if sha256_file(wav) != row["wav_sha256"]:
            raise ValueError("Original WAV hash mismatch")
        target = output / "wav" / wav.name
        if target.exists() and sha256_file(target) != row["wav_sha256"]:
            raise ValueError("Destination waveform differs")
        if not target.exists():
            shutil.copyfile(wav, target)
        row = dict(row, source_run_id=row["run_id"], run_id=manifest["run_id"],
                   source_result_directory=str(source.resolve()), condition="original", normalizer=NORMALIZER_VERSION,
                   previous_metrics=row["metrics"], metrics=word_errors(row["text"], row["asr_transcript"]),
                   wav=str(target.relative_to(output)))
        selected_rows.append(row)
    write_json(output / "manifest.json", manifest)
    temporary = output / "raw.jsonl.tmp"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected_rows))
    temporary.replace(output / "raw.jsonl")
    write_json(output / "summary.json", grouped_summary(selected_rows))
    print(f"Reused {len(selected_rows)} original-condition waveforms with v2 rescoring", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--texts", type=Path, default=ROOT / "data/eval/dev.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/hw4-tts-starter.pt")
    parser.add_argument("--prompt", type=Path, default=ROOT / "data/reference/prompt.wav")
    parser.add_argument("--prompt-metadata", type=Path, default=ROOT / "data/reference/prompt.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/prompt-diagnosis")
    parser.add_argument("--reuse-original", type=Path, default=ROOT / "results/starter-dev")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42])
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true", help="Write controlled selection/config only; never load a model")
    parser.add_argument("--rescore-results", type=Path, help="Only rescore this existing result directory; no model/device use")
    args = parser.parse_args()
    if args.rescore_results:
        rescore(args.rescore_results, args.rescore_results)
        return
    if args.per_group <= 0 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Positive group size and unique seeds required")
    if "sealed" in args.texts.name or "final" in args.texts.name:
        parser.error("Prompt format selection must use development data, not final evaluation")
    texts = read_texts(args.texts)
    selected = []
    for group in ("short", "medium", "long"):
        group_rows = [row for row in texts if row.get("length_group") == group]
        if len(group_rows) < args.per_group:
            raise ValueError(f"Need {args.per_group} texts in length_group={group}")
        selected.extend(group_rows[:args.per_group])
    args.output.mkdir(parents=True, exist_ok=True)
    text_path = args.output / "selected-dev.jsonl"
    text_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    prompt_info = json.loads(args.prompt_metadata.read_text())
    sentence_prompt = dict(prompt_info, text=SENTENCE_CASE,
        original_transcript=prompt_info["text"], diagnostic_transformation="Sentence case and punctuation only; identical recorded waveform")
    sentence_path = args.output / "prompt-sentencecase.json"
    write_json(sentence_path, sentence_prompt)
    write_json(args.output / "design.json", dict(selection="First per-group development texts in saved file order",
        texts_sha256=sha256_file(args.texts), selected_ids=[r["id"] for r in selected], seeds=args.seeds,
        per_group=args.per_group, normalizer=NORMALIZER_VERSION,
        conditions={"original": "Unmodified uppercase prompt text and reference speech in LM; original acoustic prompt",
                    "sentencecase": SENTENCE_CASE,
                    "lm-none": "No LM reference text/speech prefix; original acoustic tokens, mel, and speaker embedding"},
        interpretation="Development diagnosis only; choose settings before final evaluation. This is not a training intervention."))
    if args.prepare_only:
        print(f"Prepared {len(selected)} development texts and three conditions in {args.output}; no models loaded")
        return
    summaries = {}
    for condition in ("original", "sentencecase", "lm-none"):
        out = args.output / condition
        reused = False
        if condition == "original":
            try:
                reused = import_original(args.reuse_original, out, selected, args.seeds,
                    args.checkpoint, args.prompt, args.prompt_metadata)
            except ValueError as error:
                print(f"Original reuse declined: {error}; synthesizing an independent condition.", flush=True)
        if not reused:
            command = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--checkpoint", str(args.checkpoint),
                "--texts", str(text_path), "--output", str(out), "--prompt", str(args.prompt),
                "--prompt-metadata", str(sentence_path if condition == "sentencecase" else args.prompt_metadata),
                "--seeds", *map(str, args.seeds)]
            if condition == "lm-none":
                command.append("--no-lm-prompt")
            subprocess.run(command, check=True, cwd=ROOT)
        summaries[condition] = json.loads((out / "summary.json").read_text())
        write_json(args.output / "comparison.json", dict(conditions=summaries,
            decision="Pending review of controlled development metrics and representative audio; no automatic claim of superiority."))


if __name__ == "__main__":
    main()
