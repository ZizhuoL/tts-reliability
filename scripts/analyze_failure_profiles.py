#!/usr/bin/env python3
"""Offline ASR failure profiles and exploratory independent-length bootstrap.

Reads only supplied result JSONL and an optional adjacent manifest. Never loads
TTS/ASR models or follows references to final text files. Empty/error ASR is
missing evidence, not a zero-tail-deletion success.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tts_reliability.evaluation import NORMALIZER_VERSION, content_hash, normalize_text, word_errors


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_latest(path):
    data = path.read_bytes()
    rows, ignored_tail = {}, False
    lines = data.splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if number == len(lines) and not line.endswith(b"\n"):
                ignored_tail = True
                break
            raise ValueError(f"Corrupt complete JSON line {number}: {path}")
        if "id" not in row or "seed" not in row:
            raise ValueError(f"Missing id/seed on line {number}")
        rows[row.get("key", content_hash([row["id"], row["seed"]]))] = row
    identities = [(str(row["id"]), row["seed"]) for row in rows.values()]
    if len(set(identities)) != len(identities):
        raise ValueError("Distinct result keys have duplicate id/seed identities")
    run_ids = {row.get("run_id") for row in rows.values()}
    if len(run_ids) > 1:
        raise ValueError("Input mixes different run IDs")
    return list(rows.values()), dict(raw_sha256=hashlib.sha256(data).hexdigest(),
        raw_bytes=len(data), event_lines=len(lines), latest_items=len(rows),
        ignored_torn_final_line=ignored_tail, source_run_ids=list(run_ids))


def profile(row, tail_minimum=3):
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Nonempty reference text required: {row['id']}")
    normalized = normalize_text(text)
    text_tokens = row.get("text_tokens", row.get("text_metadata", {}).get("text_tokens"))
    if text_tokens is None and row.get("segments"):
        text_tokens = sum(part.get("target_text_tokens", 0) for part in row["segments"])
    generated = row.get("generated_tokens")
    result = dict(id=str(row["id"]), seed=row["seed"], key=row.get("key"),
        length_group=row.get("length_group", "unspecified"), status=row.get("status"),
        stop_reason=row.get("stop_reason", "unknown"), text=text,
        reference_normalized=normalized, reference_final_words=normalized.split()[-20:],
        asr_transcript=row.get("asr_transcript"), text_cluster=content_hash(normalized),
        text_tokens=text_tokens, generated_tokens=generated,
        generated_per_text_token=generated / text_tokens if generated is not None and text_tokens else None,
        text_per_generated_token=text_tokens / generated if text_tokens is not None and generated else None,
        audio_seconds=row.get("audio_seconds"), generation_seconds=row.get("generation_seconds"),
        wav=row.get("wav_path", row.get("wav")), error=row.get("error"),
        profile_available=False, missing_reason=None, metrics=None,
        trailing_deletion_count=None, trailing_deleted_words=None, tail_deletion_candidate=None,
        prefix_deletion_count=None, prefix_insertion_count=None)
    hypothesis = row.get("asr_transcript")
    if row.get("status") != "complete":
        result["missing_reason"] = "ASR/generation error" if row.get("status") == "error" else "incomplete result"
        return result
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        result["missing_reason"] = "empty or missing ASR transcript"
        return result
    if not normalized or not normalize_text(hypothesis):
        result["missing_reason"] = "reference or ASR becomes empty after normalization"
        return result
    score = word_errors(text, hypothesis)
    alignment = score["alignment"]
    tail, cursor = [], len(alignment) - 1
    while cursor >= 0 and alignment[cursor]["operation"] == "deletion":
        tail.append(alignment[cursor]["reference"])
        cursor -= 1
    prefix = []
    for edit in alignment:
        if edit["operation"] == "equal":
            break
        prefix.append(edit)
    result.update(profile_available=True, metrics=score, trailing_deletion_count=len(tail),
        trailing_deleted_words=list(reversed(tail)), tail_deletion_candidate=len(tail) >= tail_minimum,
        prefix_deletion_count=sum(edit["operation"] == "deletion" for edit in prefix),
        prefix_insertion_count=sum(edit["operation"] == "insertion" for edit in prefix))
    return result


def aggregate(rows):
    valid = [row for row in rows if row["profile_available"]]
    totals = {key: sum(row["metrics"][key] for row in valid)
              for key in ("reference_words", "errors", "substitutions", "deletions", "insertions")}
    words = totals["reference_words"]
    ratios = [row["generated_per_text_token"] for row in rows if row["generated_per_text_token"] is not None]
    candidates = [row for row in valid if row["tail_deletion_candidate"]]
    return dict(items=len(rows), profile_available_items=len(valid), missing_profile_items=len(rows)-len(valid),
        missing_reasons=dict(Counter(row["missing_reason"] for row in rows if not row["profile_available"])),
        text_clusters=len({row["text_cluster"] for row in rows}),
        stop_reasons=dict(Counter(row["stop_reason"] for row in rows)), **totals,
        wer=totals["errors"]/words if words else None,
        deletion_rate=totals["deletions"]/words if words else None,
        insertion_rate=totals["insertions"]/words if words else None,
        substitution_rate=totals["substitutions"]/words if words else None,
        trailing_deletions=sum(row["trailing_deletion_count"] for row in valid),
        prefix_deletions=sum(row["prefix_deletion_count"] for row in valid),
        prefix_insertions=sum(row["prefix_insertion_count"] for row in valid),
        tail_candidate_items=len(candidates), tail_candidate_text_clusters=len({row["text_cluster"] for row in candidates}),
        tail_candidate_rate_among_available=len(candidates)/len(valid) if valid else None,
        median_generated_per_text_token=statistics.median(ratios) if ratios else None,
        min_generated_per_text_token=min(ratios) if ratios else None,
        max_generated_per_text_token=max(ratios) if ratios else None)


def independent_length_bootstrap(rows, repeats=10000, seed=20261020):
    import numpy as np
    groups = {"short": {}, "long": {}}
    missing_clusters = set()
    membership = {}
    for row in rows:
        group = row["length_group"]
        if group not in groups:
            continue
        cluster = row["text_cluster"]
        if cluster in membership and membership[cluster] != group:
            raise ValueError("Identical normalized text spans both compared length groups")
        membership[cluster] = group
        if not row["profile_available"]:
            missing_clusters.add(cluster)
        values = groups[group].setdefault(cluster, {"errors": 0, "words": 0, "seeds": []})
        values["seeds"].append(row["seed"])
        if row["profile_available"]:
            values["errors"] += row["metrics"]["errors"]
            values["words"] += row["metrics"]["reference_words"]
    # Exclude the entire text cluster if any observed seed lacks ASR evidence;
    # never count that missing event as error-free or keep only its better seed.
    groups = {group: {key: values for key, values in clusters.items() if key not in missing_clusters}
              for group, clusters in groups.items()}
    seed_sets = {tuple(sorted(v["seeds"])) for clusters in groups.values() for v in clusters.values()}
    result = dict(method="Independent length-group percentile bootstrap; resample normalized-text clusters within short and long, retaining every observed seed for each selected text; recompute micro WER.",
        interpretation="Exploratory, non-causal length comparison; text content/source and difficulty are confounded with length. Does not establish equivalence or include ASR/training uncertainty.",
        bootstrap_seed=seed, bootstrap_repeats=repeats,
        clusters_per_length={group: len(clusters) for group, clusters in groups.items()},
        excluded_incomplete_text_clusters=len(missing_clusters), observed_seed_sets=[list(x) for x in sorted(seed_sets)],
        delta_long_minus_short_wer=None, delta_wer_ci95=None)
    if len(seed_sets) != 1 or not seed_sets:
        result["unavailable_reason"] = "Observed seed sets differ by text cluster, or no complete text cluster"
        return result
    if any(len(clusters) < 2 for clusters in groups.values()):
        result["unavailable_reason"] = "At least two complete text clusters per length group are required"
        return result
    rng, draws, points = np.random.default_rng(seed), {}, {}
    for group in ("short", "long"):
        values = np.array([[v["errors"], v["words"]] for v in groups[group].values()], dtype=np.int64)
        if (values[:, 1] <= 0).any():
            raise ValueError("Empty reference denominator in bootstrap cluster")
        total = values.sum(axis=0)
        points[group] = float(total[0]/total[1])
        indices = rng.integers(0, len(values), (repeats, len(values)))
        sampled = values[indices].sum(axis=1)
        draws[group] = sampled[:, 0]/sampled[:, 1]
    delta = draws["long"] - draws["short"]
    result.update(short_wer=points["short"], long_wer=points["long"],
        delta_long_minus_short_wer=points["long"]-points["short"],
        delta_wer_ci95=np.quantile(delta, [0.025, 0.975]).tolist(), valid_replicates=len(delta))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw.jsonl or *.rescored.jsonl; latest event per key wins")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-minimum", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--expected-items", type=int, help="Report completeness against this expected item count")
    args = parser.parse_args()
    if args.tail_minimum < 1 or args.bootstrap < 1:
        parser.error("Positive tail minimum and bootstrap repetitions required")
    source, provenance = read_latest(args.input)
    profiles = [profile(row, args.tail_minimum) for row in source]
    source_manifest_path = args.input.parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else None
    if args.input.name == "raw.jsonl" and source_manifest and source_manifest.get("run_id"):
        if provenance["source_run_ids"] != [source_manifest["run_id"]]:
            raise ValueError("Raw results do not match adjacent run manifest")
    manifest = dict(input=str(args.input.resolve()), **provenance, source_manifest=source_manifest,
        source_manifest_sha256=hashlib.sha256(source_manifest_path.read_bytes()).hexdigest() if source_manifest_path.exists() else None,
        normalizer=NORMALIZER_VERSION, implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normalization_implementation_sha256=hashlib.sha256((ROOT / "src/tts_reliability/evaluation.py").read_bytes()).hexdigest(),
        settings=dict(tail_minimum=args.tail_minimum, bootstrap=args.bootstrap, seed=args.seed, expected_items=args.expected_items),
        exact_command=shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]), cwd=str(Path.cwd()),
        interpretation="ASR alignment candidates only, not human-verified omissions. Empty/error transcripts are missing profiles, never zero-tail successes. No final text source or model is read.")
    manifest["analysis_id"] = content_hash(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["analysis_id"] != manifest["analysis_id"]:
        raise ValueError("Analysis inputs/code/settings differ; choose a new output directory")
    write_json(manifest_path, manifest)
    by_length = defaultdict(list)
    for row in profiles:
        by_length[row["length_group"]].append(row)
    summary = dict(overall=aggregate(profiles), by_length={g: aggregate(rs) for g,rs in by_length.items()},
        expected_items=args.expected_items, complete_expected_count=(len(profiles)==args.expected_items and all(row["profile_available"] for row in profiles)) if args.expected_items is not None else None,
        missing_profile_policy="Exclude missing/error ASR from tail/score denominators; report it explicitly. Bootstrap excludes its entire observed text cluster.")
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "bootstrap.json", independent_length_bootstrap(profiles, args.bootstrap, args.seed))
    for name, rows in (("profiles.jsonl", profiles), ("tail_candidates.jsonl", [r for r in profiles if r["tail_deletion_candidate"] is True])):
        (args.output / name).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
    (args.output / "reproduce.sh").write_text("#!/bin/sh\nset -eu\ncd " + shlex.quote(str(Path.cwd())) + "\n" + manifest["exact_command"] + "\n")
    print(json.dumps(dict(items=len(profiles), profile_available=summary["overall"]["profile_available_items"],
        tail_candidates=summary["overall"]["tail_candidate_items"], output=str(args.output.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
