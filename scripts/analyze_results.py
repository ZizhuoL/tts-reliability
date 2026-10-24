#!/usr/bin/env python3
"""Offline, current-normalizer rescoring and paired TTS result analysis.

Example:
  python scripts/analyze_results.py --system starter=results/starter-dev \
      --system method=results/method-dev --baseline starter \
      --output results/analysis/dev --bootstrap 2000

Only the named raw.jsonl files and their run manifests are read; this script
never follows manifest text paths or runs synthesis/ASR. WER is ASR disagreement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tts_reliability.evaluation import (
    NORMALIZER_VERSION, content_hash, normalize_text, word_errors,
)

COUNT_FIELDS = ("errors", "substitutions", "deletions", "insertions", "reference_words")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def load_run(name, folder):
    """Keep last event per key without modifying an interrupted input file."""
    path = folder / "raw.jsonl"
    data = path.read_bytes()
    events, ignored_tail = {}, False
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines)-1 and not line.endswith(b"\n"):
                ignored_tail = True
                break
            raise ValueError(f"Corrupt complete JSON line {index+1}: {path}")
        if not isinstance(row, dict) or not row.get("key"):
            raise ValueError(f"Missing event key at {path}:{index+1}")
        events[row["key"]] = row
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    run_ids = {r.get("run_id") for r in events.values()}
    if len(run_ids) > 1 or (manifest.get("run_id") and run_ids and run_ids != {manifest["run_id"]}):
        raise ValueError(f"Mixed run identities: {folder}")
    rows, identities = [], set()
    for original in events.values():
        if not isinstance(original.get("text"), str) or not original["text"].strip():
            raise ValueError(f"Nonempty reference text required: {folder}/{original['key']}")
        if "id" not in original or "seed" not in original:
            raise ValueError(f"id and seed required: {folder}/{original['key']}")
        identity = (str(original["id"]), original["seed"])
        if identity in identities:
            raise ValueError(f"Different event keys share an id/seed: {folder}/{identity}")
        identities.add(identity)
        row = dict(original)
        row["id"] = str(row["id"])
        row["system"] = name
        row["source_directory"] = str(folder.resolve())
        row["wav_path"] = str((folder / row["wav"]).resolve()) if row.get("wav") else None
        row["original_metrics"] = original.get("metrics")
        row["original_normalizer"] = manifest.get("normalizer", "unknown")
        row["analysis_normalizer"] = NORMALIZER_VERSION
        row["metrics"] = None
        row["length_group"] = row.get("length_group", "unspecified")
        row["text_cluster"] = content_hash(normalize_text(row["text"]))
        if row.get("status") == "complete" and isinstance(row.get("asr_transcript"), str):
            row["metrics"] = word_errors(row["text"], row["asr_transcript"])
            if row["metrics"]["reference_words"] == 0:
                row["analysis_score_error"] = "Reference becomes empty after normalization."
                row["metrics"] = None
        elif row.get("status") == "complete":
            row["analysis_score_error"] = "Complete event has no ASR transcript string."
        row["deletion_sensitivity_metrics"] = row["metrics"] or word_errors(row["text"], "")
        rows.append(row)
    provenance = {
        "directory": str(folder.resolve()), "raw_sha256": hashlib.sha256(data).hexdigest(),
        "raw_bytes_at_read": len(data), "deduplicated_items": len(rows),
        "event_lines": len(lines), "ignored_torn_final_line": ignored_tail,
        "manifest": manifest,
    }
    return rows, provenance


def score_totals(scores):
    total = {k: sum(s[k] for s in scores) for k in COUNT_FIELDS}
    words = total["reference_words"]
    total.update(wer=total["errors"]/words if words else None)
    for count, rate in (("substitutions", "substitution_rate"),
                        ("deletions", "deletion_rate"), ("insertions", "insertion_rate")):
        total[rate] = total[count]/words if words else None
    return total


def aggregate(rows):
    scored = [r for r in rows if r["metrics"] is not None]
    failed = [r for r in rows if r.get("status") == "error"]
    pending = [r for r in rows if r.get("status") not in ("complete", "error")]
    n = len(rows)
    eos_audio = sum(r.get("stop_reason") == "eos" and
                    (r.get("wav") is not None or r.get("status") == "complete") for r in rows)
    scores = score_totals([r["metrics"] for r in scored])
    sensitivity = score_totals([r["deletion_sensitivity_metrics"] for r in rows])
    seconds = {key: sum(r.get(key, 0) or 0 for r in rows)
               for key in ("generation_seconds", "asr_seconds", "audio_seconds")}
    stops = Counter(r.get("stop_reason", "unknown") for r in rows)
    return dict(
        items=n, text_clusters=len({r["text_cluster"] for r in rows}),
        scored_items=len(scored), failed_items=len(failed), pending_items=len(pending),
        score_errors=sum("analysis_score_error" in r for r in rows),
        scored_rate=len(scored)/n if n else None,
        exception_rate=len(failed)/n if n else None,
        exception_stages=dict(Counter(r.get("error_stage", "unknown") for r in failed)),
        stop_reasons=dict(stops),
        truncation_rate=sum(stops[k] for k in ("output_limit", "context_limit"))/n if n else None,
        eos_audio_success_rate=eos_audio/n if n else None,
        normalized_exact_match_rate=sum(r["metrics"]["errors"] == 0 for r in scored)/len(scored) if scored else None,
        scored=scores, unscored_as_full_deletions=sensitivity,
        imputed_items=n-len(scored), **seconds,
        aggregate_generation_rtf=seconds["generation_seconds"]/seconds["audio_seconds"] if seconds["audio_seconds"] else None,
    )


def group_run(rows):
    lengths, seeds, intersections = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        lengths[row["length_group"]].append(row)
        seeds[str(row["seed"])].append(row)
        intersections[(row["length_group"], str(row["seed"]))].append(row)
    return dict(
        overall=aggregate(rows),
        by_length={k: aggregate(v) for k, v in sorted(lengths.items())},
        by_seed={k: aggregate(v) for k, v in sorted(seeds.items())},
        by_length_and_seed=[dict(length_group=k[0], seed=k[1], **aggregate(v))
                            for k, v in sorted(intersections.items())],
    )


def percentile(values, q):
    values = sorted(values)
    index = (len(values)-1)*q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def paired_bootstrap(pairs, metric_key, repeats, seed):
    """Resample whole normalized-text clusters within fixed length strata."""
    strata, cluster_lengths = defaultdict(dict), {}
    for left, right in pairs:
        cluster = left["text_cluster"]
        length = left["length_group"]
        if cluster in cluster_lengths and cluster_lengths[cluster] != length:
            raise ValueError("A text cluster crosses length strata.")
        cluster_lengths[cluster] = length
        if left[metric_key]["reference_words"] != right[metric_key]["reference_words"]:
            raise ValueError("Paired metrics have different reference lengths.")
        value = strata[length].setdefault(cluster, [0, 0, 0, 0])
        value[0] += left[metric_key]["errors"]
        value[1] += right[metric_key]["errors"]
        value[2] += left[metric_key]["reference_words"]
        value[3] += 1
    groups = [list(clusters.values()) for _, clusters in sorted(strata.items())]
    left_errors = sum(v[0] for group in groups for v in group)
    right_errors = sum(v[1] for group in groups for v in group)
    words = sum(v[2] for group in groups for v in group)
    result = dict(
        paired_items=len(pairs), text_clusters=len(cluster_lengths),
        clusters_per_length={k: len(v) for k, v in strata.items()},
        reference_words=words,
        baseline_wer=left_errors/words if words else None,
        method_wer=right_errors/words if words else None,
        delta_wer=(right_errors-left_errors)/words if words else None,
        delta_wer_ci95=None, bootstrap_repeats=repeats, bootstrap_seed=seed,
        method="Percentile bootstrap: sample text clusters with replacement within length stratum; retain every paired seed in the sampled cluster; recompute micro WER.",
    )
    if not words:
        result["ci_note"] = "No scored reference words."
        return result
    if repeats < 1:
        result["ci_note"] = "Bootstrap disabled."
        return result
    if not any(len(group) > 1 for group in groups):
        result["ci_note"] = "No resampling variability: each populated length stratum has only one text cluster."
        return result
    rng, deltas = random.Random(seed), []
    for _ in range(repeats):
        left_total = right_total = denominator = 0
        for group in groups:
            for value in rng.choices(group, k=len(group)):
                left_total += value[0]
                right_total += value[1]
                denominator += value[2]
        if denominator:
            deltas.append((right_total-left_total)/denominator)
    result["bootstrap_valid_replicates"] = len(deltas)
    if not deltas:
        result["ci_note"] = "No resampled replicate contains scored reference words."
        return result
    result["delta_wer_ci95"] = [percentile(deltas, .025), percentile(deltas, .975)]
    result["ci_note"] = "Conditional on these text sources, observed seed pairs, and available scores; excludes uncertainty from training randomness and the ASR model."
    return result


def compare_runs(baseline, method, repeats, seed):
    left = {(r["id"], r["seed"]): r for r in baseline}
    right = {(r["id"], r["seed"]): r for r in method}
    shared = sorted(left.keys() & right.keys(), key=lambda key: (key[0], str(key[1])))
    pairs, examples = [], []
    for key in shared:
        a, b = left[key], right[key]
        if a["text"] != b["text"] or a["length_group"] != b["length_group"]:
            raise ValueError(f"Paired id/seed has different reference or length group: {key}")
        pairs.append((a, b))
        examples.append(dict(
            id=key[0], seed=key[1], text=a["text"], length_group=a["length_group"],
            baseline=dict(status=a.get("status"), asr_transcript=a.get("asr_transcript"),
                          metrics=a["metrics"], original_metrics=a.get("original_metrics"),
                          stop_reason=a.get("stop_reason"), wav=a.get("wav"),
                          wav_path=a.get("wav_path"), source_directory=a["source_directory"],
                          generation_seconds=a.get("generation_seconds"), audio_seconds=a.get("audio_seconds")),
            method=dict(status=b.get("status"), asr_transcript=b.get("asr_transcript"),
                        metrics=b["metrics"], original_metrics=b.get("original_metrics"),
                        stop_reason=b.get("stop_reason"), wav=b.get("wav"),
                        wav_path=b.get("wav_path"), source_directory=b["source_directory"],
                        generation_seconds=b.get("generation_seconds"), audio_seconds=b.get("audio_seconds")),
            common_success=a["metrics"] is not None and b["metrics"] is not None,
            delta_wer=(b["metrics"]["wer"]-a["metrics"]["wer"])
            if a["metrics"] and b["metrics"] else None,
        ))
    common = [(a, b) for a, b in pairs if a["metrics"] is not None and b["metrics"] is not None]
    def scope(rows):
        successful = [(a, b) for a, b in rows if a["metrics"] is not None and b["metrics"] is not None]
        return dict(
            shared_attempted_pairs=len(rows), common_success_pairs=len(successful),
            common_success_baseline=score_totals([a["metrics"] for a, _ in successful]),
            common_success_method=score_totals([b["metrics"] for _, b in successful]),
            common_success=paired_bootstrap(successful, "metrics", repeats, seed),
            unscored_as_full_deletions=paired_bootstrap(rows, "deletion_sensitivity_metrics", repeats, seed),
        )
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair[0]["length_group"]].append(pair)
    report = dict(
        baseline_only_items=len(left.keys()-right.keys()),
        method_only_items=len(right.keys()-left.keys()),
        common_success_unique_texts=len({a["text_cluster"] for a, _ in common}),
        shared_attempted_unique_texts=len({a["text_cluster"] for a, _ in pairs}),
        overall=scope(pairs), by_length={k: scope(v) for k, v in sorted(groups.items())},
        comparability_note="Pairs match exact id, seed, raw reference text, and length group. Unmatched attempts are excluded from paired analyses and counted separately; configurations may still differ beyond the intended intervention.",
    )
    return report, examples


def flat_summary(system, scope_name, group, result):
    flat = dict(system=system, scope=scope_name, group=group)
    for key, value in result.items():
        if isinstance(value, dict) and key in ("scored", "unscored_as_full_deletions"):
            flat.update({f"{key}_{k}": v for k, v in value.items()})
        else:
            flat[key] = value
    return flat


def plot_results(report, output):
    import math
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    systems = list(report["systems"])
    lengths = sorted(set().union(*(r["by_length"].keys() for r in report["systems"].values())),
                     key=lambda x: ({"short": 0, "medium": 1, "long": 2}.get(x, 3), x))
    if not lengths:
        return
    legend_rows = math.ceil(len(systems) / 6)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5 + .25 * max(0, legend_rows - 1)))
    width = .8/max(1, len(systems))
    finite_values = []
    for index, name in enumerate(systems):
        entries = report["systems"][name]["by_length"]
        positions = [i-.4+width*(index+.5) for i in range(len(lengths))]
        for axis, metric, label in ((axes[0], "scored", "WER on scored items (%)"),
                                   (axes[1], "unscored_as_full_deletions", "Unscored = deletion sensitivity WER (%)")):
            values = [entries.get(length, {}).get(metric, {}).get("wer") for length in lengths]
            percentages = [100*x if x is not None and math.isfinite(x) else float("nan") for x in values]
            finite_values.extend(x for x in percentages if math.isfinite(x))
            axis.bar(positions, percentages, width=width, label=name)
            axis.set_xticks(range(len(lengths)), lengths)
            axis.set_ylabel(label)
            axis.grid(axis="y", alpha=.2)
    upper = max(1.0, max(finite_values, default=0.0) * 1.12)
    for axis in axes:
        axis.set_ylim(0, upper)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .93),
                  ncol=min(6, max(1, len(systems))), frameon=False)
    figure.suptitle("ASR disagreement by input length; not a perceptual quality measure", y=.99)
    figure.tight_layout(rect=(0, 0, 1, .84 - .04 * max(0, legend_rows - 1)))
    figure.savefig(output / "wer_by_length.png", dpi=180)
    figure.savefig(output / "wer_by_length.pdf")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--baseline", help="Name of baseline for pairing each remaining system")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1832026)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    systems, provenance = {}, {}
    for item in args.system:
        if "=" not in item:
            parser.error("--system requires NAME=DIRECTORY")
        name, path = item.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in systems:
            parser.error("System names must be unique and contain only letters, digits, '_' or '-'.")
        systems[name], provenance[name] = load_run(name, Path(path))
    if args.baseline and args.baseline not in systems:
        parser.error("--baseline must name one of the supplied systems")
    output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be empty or new; refusing to overwrite analysis evidence")
    # Validate all paired references before writing any output.
    comparisons, paired_examples = {}, {}
    if args.baseline:
        for name, rows in systems.items():
            if name != args.baseline:
                comparisons[name], paired_examples[name] = compare_runs(
                    systems[args.baseline], rows, args.bootstrap, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {name: group_run(rows) for name, rows in systems.items()}
    report = dict(
        normalizer=NORMALIZER_VERSION, baseline=args.baseline,
        systems=summaries, paired=comparisons,
        interpretation={
            "denominators": "Per-system rates use observed deduplicated attempts. WER uses reference words of scored items. No inference is made about jobs absent from raw.jsonl.",
            "completeness": "eos_audio_success_rate is audio available plus EOS / attempted items. It is operational completion, not proof all words were spoken. Normalized exact match is ASR transcript agreement.",
            "sensitivity": "Every unscored attempted item (failure, pending, invalid score) is assigned an empty hypothesis: all reference words deleted. This penalizes missing output but is NOT a mathematical upper bound; WER can exceed 100% through insertions.",
            "pairing": "Common-success analysis and deletion sensitivity use only shared attempted id/seed pairs. Missing jobs are counted, never invented as completed or failed.",
            "uncertainty": "Length-stratified text-cluster bootstrap retains all observed paired seeds per text, recomputes micro WER, and reports method minus baseline; negative delta favors method.",
            "time": "Generation/ASR durations are summed from last records. Audio duration includes inserted segmentation pauses; failed attempts may lack duration. These totals are not full wall-clock run duration.",
        },
    )
    write_json(output / "summary.json", report)
    write_json(output / "analysis_manifest.json", {
        "normalizer": NORMALIZER_VERSION, "arguments": {**vars(args), "output": str(output.resolve())},
        "inputs": provenance,
        "implementation": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), ROOT / "src/tts_reliability/evaluation.py")},
    })
    csv_rows = []
    for name, rows in systems.items():
        write_jsonl(output / f"{name}.rescored.jsonl", rows)
        csv_rows.append(flat_summary(name, "overall", "all", summaries[name]["overall"]))
        for scope_name in ("by_length", "by_seed"):
            csv_rows.extend(flat_summary(name, scope_name, group, result)
                            for group, result in summaries[name][scope_name].items())
        csv_rows.extend(flat_summary(name, "by_length_and_seed", f"{r['length_group']}:{r['seed']}", r)
                        for r in summaries[name]["by_length_and_seed"])
    write_csv(output / "metrics.csv", csv_rows)
    paired_csv = []
    for name, examples in paired_examples.items():
        write_jsonl(output / f"{args.baseline}_vs_{name}.examples.jsonl", examples)
        for scope_name, result in [("overall", comparisons[name]["overall"])] + list(comparisons[name]["by_length"].items()):
            for kind in ("common_success", "unscored_as_full_deletions"):
                statistics = dict(result[kind])
                statistics["bootstrap_method"] = statistics.pop("method")
                paired_csv.append(dict(baseline=args.baseline, method=name,
                                       method_system=name, scope=scope_name,
                                       analysis=kind, **statistics))
    write_csv(output / "paired.csv", paired_csv)
    if args.plots:
        plot_results(report, output)
    # Written only after every requested export succeeds. Earlier run manifests
    # describe inputs, not completion; partial outputs must not be mistaken for
    # a finished analysis merely because summary.json exists.
    write_json(output / "completion.json", dict(
        status="complete",
        artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(output.iterdir()) if p.is_file()},
    ))
    print(json.dumps({"output": str(output.resolve()), "systems": {k: len(v) for k, v in systems.items()},
                      "paired_methods": list(comparisons), "normalizer": NORMALIZER_VERSION}))
    return report


if __name__ == "__main__":
    main()
