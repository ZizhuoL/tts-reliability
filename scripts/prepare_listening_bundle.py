#!/usr/bin/env python3
"""Build a local listening index from existing offline analysis artifacts.

Input examples:
  --system OriginalPrompt=results/analysis/prompt-diagnosis
  --system SentenceCase=results/analysis/prompt-diagnosis
  --pair OriginalPrompt:SentenceCase=path/to/paired.examples.jsonl

NAME=DIRECTORY resolves DIRECTORY/NAME.rescored.jsonl. Existing
BASE_vs_METHOD.examples.jsonl files are discovered between named systems.
No source text manifests, models, synthesis or ASR are opened or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    if path.name == "final.sealed.jsonl" or not (
            path.name.endswith(".rescored.jsonl") or path.name.endswith(".examples.jsonl")):
        raise ValueError("Only analysis *.rescored.jsonl / *.examples.jsonl inputs are supported.")
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    return rows, dict(path=str(path.resolve()), sha256=digest(data), rows=len(rows))


def identity(row):
    return str(row["id"]), row["seed"]


def score(row):
    value = row.get("metrics") or {}
    wer = value.get("wer")
    return wer if isinstance(wer, (int, float)) and math.isfinite(wer) else None


def sort_key(row, reverse=False):
    return ((-1 if reverse else 1)*score(row), str(row["id"]), str(row["seed"]))


def diverse_top(rows, count, excluded_texts=()):
    selected, seen = [], set(excluded_texts)
    for row in rows:
        # Avoid presenting two seeds of the same sentence as distinct examples.
        if row["text"] in seen:
            continue
        selected.append(row)
        seen.add(row["text"])
        if len(selected) == count:
            break
    return selected


def audio_path(row):
    if row.get("wav_path"):
        return Path(row["wav_path"]).resolve()
    if row.get("source_directory") and row.get("wav"):
        return (Path(row["source_directory"])/row["wav"]).resolve()
    return None


def escape(text):
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def pct(value):
    return "未评分" if value is None else f"{100*value:.2f}%"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--pair", action="append", default=[], metavar="BASE:METHOD=FILE")
    parser.add_argument("--include", action="append", default=[], metavar="NAME:ID:SEED",
                        help="Explicit diagnostic example, in addition to unchanged automatic selections")
    parser.add_argument("--include-reason",
                        help="Shared reason recorded for explicitly included examples; not a listening verdict")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=1)
    parser.add_argument("--audio-mode", choices=("copy", "reference"), default="copy")
    args = parser.parse_args(argv)
    if args.per_category < 1:
        parser.error("--per-category must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be new or empty; existing bundles are not overwritten")
    systems, folders, provenance, unavailable = {}, {}, [], []
    normalizers = set()
    for specification in args.system:
        if "=" not in specification:
            parser.error("--system requires NAME=DIRECTORY")
        name, value = specification.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in systems:
            parser.error("System names must be unique letters/digits/'_'/'-'")
        folder = Path(value).resolve()
        rows, source = read_jsonl(folder / f"{name}.rescored.jsonl")
        provenance.append(source)
        mapping = {}
        for row in rows:
            key = identity(row)
            if key in mapping:
                raise ValueError(f"Duplicate id/seed in rescored input: {name}/{key}")
            if row.get("system", name) != name:
                raise ValueError(f"System label mismatch: {name}")
            normalizers.add(row.get("analysis_normalizer", "unknown"))
            path = audio_path(row)
            if path is None or not path.is_file():
                unavailable.append(dict(system=name, id=key[0], seed=key[1],
                                        reason="No existing WAV", path=str(path)))
            mapping[key] = row
        systems[name], folders[name] = mapping, folder
    if len(normalizers) > 1:
        raise ValueError("Analysis normalizers differ; rescore all systems uniformly first.")
    pairs = {}
    for specification in args.pair:
        try:
            names, value = specification.split("=", 1)
            base, method = names.split(":", 1)
        except ValueError:
            parser.error("--pair requires BASE:METHOD=FILE")
        if base not in systems or method not in systems or base == method:
            parser.error("Pair names must be two distinct supplied systems")
        pairs[(base, method)] = Path(value).resolve()
    for base in systems:
        for method in systems:
            if base == method or (base, method) in pairs:
                continue
            found = {folder/f"{base}_vs_{method}.examples.jsonl" for folder in folders.values()}
            found = {path for path in found if path.is_file()}
            if len(found) > 1:
                raise ValueError(f"Ambiguous paired files for {base}/{method}; specify --pair.")
            if found:
                pairs[(base, method)] = next(iter(found))

    selections, categories, paired_selections = {}, [], []
    def register(name, row, reason):
        key = (name, *identity(row))
        entry = selections.setdefault(key, dict(
            system=name, row=row, selection_reasons=[],
            sample_id="S"+digest(json.dumps(key, ensure_ascii=False).encode())[:12],
        ))
        if reason not in entry["selection_reasons"]:
            entry["selection_reasons"].append(reason)
        return entry["sample_id"]

    length_order = lambda length: ({"short": 0, "medium": 1, "long": 2}.get(length, 3), length)
    for name, mapping in systems.items():
        groups = defaultdict(list)
        for row in mapping.values():
            if score(row) is not None and (path := audio_path(row)) and path.is_file():
                groups[row.get("length_group", "unspecified")].append(row)
        for length, rows in sorted(groups.items(), key=lambda item: length_order(item[0])):
            best = diverse_top(sorted(rows, key=sort_key), args.per_category)
            worst = diverse_top(sorted(rows, key=lambda row: sort_key(row, True)),
                                args.per_category, [row["text"] for row in best])
            for label, chosen in (("lower_WER", best), ("higher_WER", worst)):
                reason = f"{name}/{length}/{label}"
                categories.append(dict(
                    category=reason, candidates=len(rows),
                    sample_ids=[register(name, row, reason) for row in chosen],
                    rule="Rank WER ascending" if label == "lower_WER" else
                    "Rank WER descending, excluding texts already chosen as lower_WER",
                ))

    for (base, method), path in pairs.items():
        rows, source = read_jsonl(path)
        provenance.append(source)
        groups = defaultdict(list)
        seen_pairs = set()
        for example in rows:
            key = identity(example)
            if key in seen_pairs:
                raise ValueError(f"Duplicate paired id/seed: {path}/{key}")
            seen_pairs.add(key)
            if key not in systems[base] or key not in systems[method]:
                raise ValueError(f"Paired example absent from supplied rescored data: {key}")
            left, right = systems[base][key], systems[method][key]
            if left["text"] != right["text"] or left["text"] != example["text"]:
                raise ValueError(f"Paired text mismatch: {key}")
            if score(left) is None or score(right) is None:
                continue
            if any(audio_path(row) is None or not audio_path(row).is_file() for row in (left, right)):
                continue
            delta = score(right)-score(left)
            stored_delta = example.get("delta_wer")
            if stored_delta is None or not math.isclose(delta, stored_delta, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Paired scores differ from current rescored input: {key}")
            length = left.get("length_group", "unspecified")
            if right.get("length_group", "unspecified") != length:
                raise ValueError(f"Paired length mismatch: {key}")
            groups[length].append(dict(left=left, right=right, delta=delta,
                                       text=left["text"], id=key[0], seed=key[1]))
        for length, candidates in sorted(groups.items(), key=lambda item: length_order(item[0])):
            for label, sign in (("improvement", -1), ("regression", 1)):
                directional = [row for row in candidates if row["delta"]*sign > 1e-12]
                directional.sort(key=lambda row: (-sign*row["delta"], row["id"], str(row["seed"])))
                chosen = diverse_top(directional, args.per_category)
                ids = []
                for row in chosen:
                    reason = f"{base}_vs_{method}/{length}/{label}"
                    left_id, right_id = register(base, row["left"], reason), register(method, row["right"], reason)
                    pair_id = "P"+digest(f"{reason}:{row['id']}:{row['seed']}".encode())[:12]
                    ids.append(pair_id)
                    paired_selections.append(dict(
                        pair_id=pair_id, baseline=base, method=method,
                        id=row["id"], seed=row["seed"], text=row["text"],
                        length_group=length, direction=label, delta_wer=row["delta"],
                        baseline_sample_id=left_id, method_sample_id=right_id,
                        source_examples=str(path), selection_rule=reason,
                    ))
                categories.append(dict(
                    category=f"{base}_vs_{method}/{length}/{label}",
                    candidates=len(directional), pair_ids=ids,
                    rule="Largest signed WER decrease/increase among scored pairs with existing audio; no substitute if the direction is absent.",
                ))

    explicit_requests = []
    explicit_seen = set()
    include_reason = args.include_reason or "Explicit diagnostic candidate; not manually listening-verified."
    for specification in args.include:
        try:
            name, remainder = specification.split(":", 1)
            identifier, seed_string = remainder.rsplit(":", 1)
            seed = int(seed_string)
        except ValueError:
            parser.error("--include requires NAME:ID:SEED with an integer seed")
        if name not in systems:
            parser.error(f"Unknown --include system: {name}")
        key = (identifier, seed)
        if key not in systems[name]:
            parser.error(f"--include id/seed does not exist in {name}: {identifier}:{seed}")
        row = systems[name][key]
        path = audio_path(row)
        if path is None or not path.is_file():
            parser.error(f"--include requires an existing WAV: {specification}")
        if (name, *key) in explicit_seen:
            continue
        explicit_seen.add((name, *key))
        sample_id = register(name, row, f"explicit: {include_reason}")
        explicit_requests.append(dict(system=name, id=identifier, seed=seed,
                                      sample_id=sample_id, reason=include_reason))
    if explicit_requests:
        categories.append(dict(
            category="explicit_inclusions", candidates=len(explicit_requests),
            sample_ids=[row["sample_id"] for row in explicit_requests],
            rule=include_reason,
        ))

    # Verify every chosen file before creating output.
    import soundfile as sf
    for entry in selections.values():
        row, path = entry["row"], audio_path(entry["row"])
        actual_sha = sha_file(path)
        if row.get("wav_sha256") and row["wav_sha256"] != actual_sha:
            raise ValueError(f"Source WAV hash changed: {path}")
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Invalid WAV header: {path}")
        entry.update(source_wav=str(path), wav_sha256=actual_sha,
                     sample_rate=info.samplerate, frames=info.frames,
                     channels=info.channels, actual_duration_seconds=info.frames/info.samplerate)
    output.mkdir(parents=True, exist_ok=True)
    for entry in selections.values():
        source = Path(entry["source_wav"])
        if args.audio_mode == "copy":
            destination = output / "audio" / entry["system"] / f"{entry['sample_id']}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha_file(destination) != entry["wav_sha256"]:
                raise IOError(f"Copied audio checksum mismatch: {destination}")
        else:
            destination = source
        entry["audio_relative_path"] = os.path.relpath(destination, output)
        entry["audio_absolute_path"] = str(destination.resolve())

    entries = sorted(selections.values(), key=lambda entry: (
        entry["system"], length_order(entry["row"].get("length_group", "unspecified")), entry["sample_id"]))
    manifest = dict(
        format_version=1, human_listening_performed=False, purpose="Automatically selected listening candidates, not a perceptual quality ranking.",
        arguments={**vars(args), "output": str(output)}, normalizers=sorted(normalizers),
        sources=provenance, script_sha256=sha_file(Path(__file__)),
        selection_rules=[
            "For each system/length, select lowest and highest scored ASR WER with an existing WAV.",
            "Select distinct texts, not multiple seeds of one text; higher-WER picks exclude lower-WER texts.",
            "For each available system pair/length, select largest strictly negative and positive WER delta; do not invent missing directions.",
            "Tie-break deterministically by text id then seed; selections are illustrative extremes, not representative estimates.",
            "A lower WER does not establish better naturalness, timbre, prosody or word completeness; listen before judging.",
        ],
        categories=categories, samples=entries, pairs=paired_selections,
        explicit_inclusions=explicit_requests,
        missing_audio=unavailable, available_input_items={name: len(rows) for name, rows in systems.items()},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    lines = [
        "# TTS 诊断试听索引", "",
        "以下样本由 ASR 指标自动选择，尚未进行人工听音。低/高 WER 只表示转写误差较少/较多，不表示感知音质好坏。",
        "每个系统、长度组挑选低 WER 与高 WER 的不同文本；配对挑选同文本同 seed 的最大改善和退步。若不存在某方向样本，不替代或虚构。",
        "这些是有意选择的诊断案例，不能用于估计总体平均效果。完整指标、原指标、对齐和来源见 `manifest.json`。",
        "", "## 样本总览", "",
        "| 样本 | 系统 | 长度 | seed | WER | S/D/I | stop | 音频 |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for entry in entries:
        row, metrics = entry["row"], entry["row"]["metrics"] or {}
        link = quote(entry["audio_relative_path"], safe="/.")
        lines.append(f"| [{entry['sample_id']}](#{entry['sample_id'].lower()}) | {escape(entry['system'])} | "
                     f"{escape(row.get('length_group'))} | {row['seed']} | {pct(metrics.get('wer'))} | "
                     f"{metrics.get('substitutions', '—')}/{metrics.get('deletions', '—')}/{metrics.get('insertions', '—')} | "
                     f"{escape(row.get('stop_reason'))} | [播放 WAV]({link}) |")
    if explicit_requests:
        lines += ["", "## 显式指定候选", "",
                  "以下条目按指定诊断原因额外纳入，未经人工确认；自动选例规则保持不变。", ""]
        for request in explicit_requests:
            lines.append(f"- [{request['system']} / {request['id']} / seed {request['seed']}]"
                         f"(#{request['sample_id'].lower()})：{escape(request['reason'])}")
    lines += ["", "## 配对试听", ""]
    if not paired_selections:
        lines.append("没有可用的严格改善或退步配对。")
    for pair in paired_selections:
        lines += [
            f"- {pair['pair_id']} · {pair['length_group']} · {pair['direction']} · seed {pair['seed']} · "
            f"ΔWER（方法 − 基线）{100*pair['delta_wer']:+.2f} 个百分点："
            f"[{pair['baseline']}](#{pair['baseline_sample_id'].lower()}) → "
            f"[{pair['method']}](#{pair['method_sample_id'].lower()})",
        ]
    empty = [category["category"] for category in categories
             if not category.get("sample_ids", category.get("pair_ids"))]
    if empty:
        lines += ["", "以下选择类别没有满足条件的不同文本/改善/退步样本：", ""]
        lines.extend(f"- `{category}`" for category in empty)
    for entry in entries:
        row = entry["row"]
        metrics = row["metrics"] or {}
        lines += [
            "", f"## {entry['sample_id']}", "",
            f"系统：{entry['system']}；文本 ID：`{row['id']}`；seed：{row['seed']}；"
            f"长度组：{row.get('length_group')}；stop：{row.get('stop_reason')}。",
            "", f"选择原因：`{'`, `'.join(entry['selection_reasons'])}`", "",
            f"原文本：{escape(row['text'])}", "",
            f"ASR：{escape(row.get('asr_transcript', '未评分'))}", "",
            f"WER：{pct(metrics.get('wer'))}；S/D/I："
            f"{metrics.get('substitutions', '—')}/{metrics.get('deletions', '—')}/{metrics.get('insertions', '—')}；"
            f"参考词数：{metrics.get('reference_words', '未评分')}；"
            f"音频：{entry['actual_duration_seconds']:.3f} 秒，{entry['sample_rate']} Hz。",
            "", f"[播放 WAV]({quote(entry['audio_relative_path'], safe='/.')})", "",
            "人工听音备注：未填写。",
        ]
    (output / "INDEX.md").write_text("\n".join(lines)+"\n")
    result = dict(output=str(output), unique_audio_files=len(entries), paired_examples=len(paired_selections),
                  missing_input_audio=len(unavailable), audio_mode=args.audio_mode,
                  explicit_inclusions=len(explicit_requests))
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
