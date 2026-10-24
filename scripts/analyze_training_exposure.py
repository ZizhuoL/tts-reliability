#!/usr/bin/env python3
"""Read-only reconstruction of training exposure, or explicitly labeled plans.

Uses the small [text_length, speech_length] NPY and the actual BucketSampler.
Does not load train.pt, use a GPU, read held-out texts, or modify a training run.
Optional checkpoint verification uses mmap; checkpoint tensor storage is not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import numpy as np

from tts_reliability.training import BucketSampler


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Simulate exposure only; not training evidence")
    parser.add_argument("--config", type=Path, help="Saved training config.json; required outside --plan")
    parser.add_argument("--log", type=Path, nargs="+", help="Chronological train.jsonl segments")
    parser.add_argument("--checkpoint", type=Path, help="Optional mmap-only counters/sampler verification")
    parser.add_argument("--lengths", type=Path, default=ROOT / "data/processed/train_lengths.npy")
    parser.add_argument("--output", type=Path, required=True, help="New analysis JSON, outside the run directory")
    parser.add_argument("--token-budget", type=int, help="Plan override; actual mode verifies saved config")
    parser.add_argument("--steps", type=int, help="Plan step cap, or actual prefix ending at this logged step")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--long-weight", type=float)
    parser.add_argument("--pad-multiple", type=int)
    parser.add_argument("--context-capacity", type=int, default=2048)
    parser.add_argument("--text-short-max", type=int, default=17)
    parser.add_argument("--text-medium-max", type=int, default=43)
    parser.add_argument("--label")
    args = parser.parse_args()
    if not args.plan and args.config is None:
        parser.error("actual reconstruction requires --config")
    if args.plan and (args.log or args.checkpoint):
        parser.error("--plan does not read run logs/checkpoints")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.text_short_max >= args.text_medium_max:
        parser.error("text-short-max must be smaller than text-medium-max")
    return args


def read_logs(paths):
    logs, sources, partial_tails = {}, [], []
    rollback_events, discarded, all_rows = [], [], []
    frontier = 0
    for path in paths:
        content = path.read_bytes()
        lines = content.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            partial_tails.append(str(path))
            lines.pop()
        sources.append(dict(path=str(path.resolve()), sha256_snapshot=sha256_bytes(content),
                            snapshot_bytes=len(content)))
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            step = row["step"]
            if not isinstance(step, int) or step < 1:
                raise ValueError(f"invalid step at {path}:{line_number}")
            all_rows.append(row)
            if logs and step <= frontier:
                # A resumed branch invalidates the entire old tail, including
                # higher old step numbers that have not yet been rerun.
                dropped_steps = sorted(key for key in logs if key >= step)
                rollback_events.append(dict(
                    source=str(path.resolve()), line=line_number, resumed_at_step=step,
                    previous_frontier=frontier, discarded_steps=dropped_steps,
                    inference="Chronological step rollback; old tail excluded from effective model budget."))
                discarded.extend(logs.pop(key) for key in dropped_steps)
            logs[step] = row
            frontier = step
    replay = dict(
        policy="Chronological logs: on a non-increasing step, invalidate every old step >= the resumed step; retain the latest branch only.",
        rollback_events=rollback_events, discarded_rows=discarded, all_rows=all_rows)
    return logs, sources, partial_tails, replay


def group_stats(mask, counts, target_lengths):
    baseline_examples = int(mask.sum())
    exposed_examples = int(counts[mask].sum())
    exposed_targets = int(np.dot(counts[mask], target_lengths[mask]))
    total_examples = int(counts.sum())
    total_targets = int(np.dot(counts, target_lengths))
    return dict(
        dataset_examples=baseline_examples,
        dataset_sample_share=baseline_examples / len(counts),
        dataset_target_tokens=int(target_lengths[mask].sum()),
        dataset_target_share=float(target_lengths[mask].sum() / target_lengths.sum()),
        exposure_examples=exposed_examples,
        exposure_sample_share=exposed_examples / total_examples if total_examples else None,
        exposure_target_tokens=exposed_targets,
        exposure_target_share=exposed_targets / total_targets if total_targets else None,
        unique_examples=int(np.count_nonzero(counts[mask])),
        unique_coverage=float(np.count_nonzero(counts[mask]) / baseline_examples)
        if baseline_examples else None)


def observed_runtime(logs):
    rows = list(logs)
    if not rows:
        return None
    memory_keys = (
        "rss_bytes", "mps_allocated_bytes", "mps_driver_bytes",
        "system_available_bytes", "system_swap_used_bytes")
    first = rows[0].get("memory_before_empty_cache", rows[0])
    observations = []
    for row in rows:
        observations.extend([row.get("memory_before_empty_cache", {}),
                             row, row.get("memory_after_empty_cache", {})])
    memory = {}
    for key in memory_keys:
        values = [point[key] for point in observations if key in point]
        if values:
            initial = first.get(key, values[0])
            final = rows[-1].get(key, values[-1])
            memory[key] = dict(first_observed=initial, final_observed=final,
                               min_observed=min(values), max_observed=max(values),
                               final_change_from_first=final - initial,
                               max_increase_from_first=max(values) - initial)
    return dict(
        observed_update_rows=len(rows),
        compute_seconds=sum(row.get("compute_seconds", 0) for row in rows),
        step_wall_seconds=sum(row.get("total_seconds", 0) for row in rows),
        empty_cache_seconds=sum(row.get("empty_cache_seconds", 0) for row in rows),
        wall_time_scope="Sum of logged update durations, excluding inter-step validation, checkpoints, startup and unlogged work.",
        memory_baseline="First logged update, before cache clearing when available; not process-start memory.",
        memory=memory)


def analyze(args):
    config_bytes = args.config.read_bytes() if args.config else None
    config = json.loads(config_bytes) if config_bytes else {}
    settings = {}
    defaults = dict(batch_size=8, seed=183, long_weight=1.0, pad_multiple=64)
    for key, fallback in defaults.items():
        override = getattr(args, key)
        saved = config.get(key, fallback)
        if not args.plan and override is not None and override != saved:
            raise ValueError(f"actual reconstruction cannot override saved {key}")
        settings[key] = saved if override is None else override
    if settings["batch_size"] <= 0 or settings["pad_multiple"] <= 0 or settings["long_weight"] < 1:
        raise ValueError("invalid sampling/padding settings")
    lengths = np.load(args.lengths, mmap_mode="r", allow_pickle=False)
    if lengths.ndim != 2 or lengths.shape[1] != 2 or np.any(lengths <= 0):
        raise ValueError("lengths must be a positive [N,2] integer array")
    if not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("lengths must have integer dtype")
    source = [(index, int(text), int(speech)) for index, (text, speech) in enumerate(lengths)]
    targets = np.asarray(lengths[:, 1], dtype=np.int64) + 1
    full_epoch_targets = int(targets.sum())
    if not args.plan:
        if config.get("valid_examples") != len(lengths):
            raise ValueError("length array count differs from saved valid_examples")
        if config.get("full_epoch_target_tokens") != full_epoch_targets:
            raise ValueError("length array target sum differs from saved full_epoch_target_tokens")
    budget = args.token_budget if args.token_budget is not None else config.get(
        "effective_token_budget", full_epoch_targets)
    if budget <= 0:
        raise ValueError("positive token budget required")
    if not args.plan and args.token_budget is not None and budget != config["effective_token_budget"]:
        raise ValueError("actual reconstruction cannot override the saved token budget")
    sampler = BucketSampler(source, settings["batch_size"], settings["seed"],
                            settings["long_weight"])
    settings.update(token_budget=budget, context_capacity=args.context_capacity,
                    bucket_batches=sampler.bucket_batches)
    logs, log_sources, partial_tails = {}, [], []
    replay = dict(rollback_events=[], discarded_rows=[], all_rows=[])
    if not args.plan:
        paths = args.log or [args.config.parent / "train.jsonl"]
        logs, log_sources, partial_tails, replay = read_logs(paths)
        if not logs:
            raise ValueError("no complete update rows; no observed exposure to reconstruct")
        step_limit = args.steps or max(logs)
        missing = sorted(set(range(1, step_limit + 1)) - logs.keys())
        if missing:
            raise ValueError(f"missing logged steps through requested prefix: {missing[:10]}")
        logs = {step: logs[step] for step in range(1, step_limit + 1)}
    else:
        step_limit = args.steps
    checkpoint = None
    if args.checkpoint:
        import torch
        # mmap reads metadata without materializing the large weight/optimizer
        # storages. Never fall back to a potentially multi-GB eager load.
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if state.get("kind") != "tts_reliability_training":
            raise ValueError("expected project training checkpoint")
        checkpoint = dict(counters=state["counters"], sampler=state["sampler"])
        del state
    counts = np.zeros(len(lengths), dtype=np.int64)
    totals = dict(steps=0, target_tokens=0, examples=0, unique_examples=0,
                  effective_input_tokens=0, padded_input_tokens=0)
    mismatches = []
    checkpoint_verification = None

    def verify(key, expected, observed, step):
        if isinstance(expected, float):
            matches = math.isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-10)
        else:
            matches = expected == observed
        if not matches:
            mismatches.append(dict(step=step, field=key, reconstructed=expected, logged=observed))

    while (step_limit is None or totals["steps"] < step_limit) and (
            not args.plan or totals["target_tokens"] < budget):
        ids = sampler.peek()
        sampled_lengths = lengths[ids]
        count = len(ids)
        target_count = int(targets[ids].sum())
        effective = int(sampled_lengths.sum()) + 2 * count
        raw_length = int(sampled_lengths.sum(axis=1).max()) + 2
        if raw_length > args.context_capacity:
            raise ValueError(f"batch contains over-capacity sequence {raw_length}")
        padded_length = min(args.context_capacity,
                            math.ceil(raw_length / settings["pad_multiple"]) * settings["pad_multiple"])
        padded = padded_length * count
        totals["unique_examples"] += sum(counts[index] == 0 for index in set(ids))
        np.add.at(counts, ids, 1)
        totals["steps"] += 1
        totals["target_tokens"] += target_count
        totals["examples"] += count
        totals["effective_input_tokens"] += effective
        totals["padded_input_tokens"] += padded
        sampler.advance()
        if not args.plan:
            row = logs[totals["steps"]]
            expected = dict(
                target_tokens=target_count, examples=count, text_tokens=int(sampled_lengths[:, 0].sum()),
                speech_tokens=target_count - count, effective_input_tokens=effective,
                padded_input_tokens=padded, padding_fraction=1 - effective / padded,
                max_sequence=padded_length, total_target_tokens=totals["target_tokens"],
                unique_examples=int(totals["unique_examples"]), epoch=sampler.epoch,
                epoch_batch_cursor=sampler.cursor)
            for key, value in expected.items():
                if key not in row:
                    mismatches.append(dict(step=totals["steps"], field=key, error="missing from log"))
                else:
                    verify(key, value, row[key], totals["steps"])
        if checkpoint is not None and totals["steps"] == checkpoint["counters"]["steps"]:
            checkpoint_errors = []
            for key in ("target_tokens", "examples", "effective_input_tokens", "padded_input_tokens"):
                if totals[key] != checkpoint["counters"][key]:
                    checkpoint_errors.append(key)
            if sampler.state_dict() != checkpoint["sampler"]:
                checkpoint_errors.append("sampler")
            if np.flatnonzero(counts).tolist() != checkpoint["counters"]["seen_indices"]:
                checkpoint_errors.append("seen_indices")
            checkpoint_verification = dict(step=totals["steps"], matched=not checkpoint_errors,
                                           mismatched_fields=checkpoint_errors)
    totals["unique_examples"] = int(totals["unique_examples"])
    totals.update(
        completed_sampler_epochs=sampler.epoch, next_epoch_batch_cursor=sampler.cursor,
        unique_coverage=totals["unique_examples"] / len(lengths),
        target_budget_overshoot=max(0, totals["target_tokens"] - budget),
        padding_fraction=1 - totals["effective_input_tokens"] / totals["padded_input_tokens"],
        mean_exposures_per_dataset_example=totals["examples"] / len(lengths),
        max_exposures_for_one_example=int(counts.max()),
        repeated_exposure_examples=totals["examples"] - totals["unique_examples"])
    threshold = sorted(int(value) for value in lengths[:, 1])[int(0.75 * (len(lengths) - 1))]
    text = lengths[:, 0]
    text_groups = {
        "short": text <= args.text_short_max,
        "medium": (text > args.text_short_max) & (text <= args.text_medium_max),
        "long": text > args.text_medium_max}
    multiplicities, multiplicity_counts = np.unique(counts, return_counts=True)
    report = dict(
        kind="planned_training_exposure" if args.plan else "observed_training_exposure_reconstruction",
        evidence_status="PLAN_ONLY_NOT_EXECUTED" if args.plan else "RECONSTRUCTED_FROM_LOGGED_UPDATES",
        label=args.label, settings=settings, totals=totals,
        interpretation=("Deterministic sampling simulation only. These are planned exposures, not completed training or measured timings."
                        if args.plan else
                        "Reconstructed logical update exposure. Compared against each complete log row; logs prove recorded updates, not independently audited hardware execution."),
        dataset=dict(valid_examples=len(lengths), full_epoch_target_tokens=full_epoch_targets,
                     lengths_path=str(args.lengths.resolve()), lengths_sha256=fingerprint(args.lengths)),
        source=dict(config_sha256=sha256_bytes(config_bytes) if config_bytes else None,
                    log_snapshots=log_sources,
                    sampler_source_sha256=fingerprint(ROOT / "src/tts_reliability/training.py"),
                    analysis_source_sha256=fingerprint(Path(__file__))),
        text_group_definition=dict(
            short=f"1..{args.text_short_max}", medium=f"{args.text_short_max + 1}..{args.text_medium_max}",
            long=f">={args.text_medium_max + 1}",
            note="Exhaustive training groups; cannot be equated directly with evaluation strata. Training short/long include tails outside evaluation range 6..72."),
        training_length_groups={name: group_stats(mask, counts, targets) for name, mask in text_groups.items()},
        evaluation_range_length_groups={
            name: group_stats(mask, counts, targets) for name, mask in {
                "below_eval_1_to_5": text < 6, "eval_short_6_to_17": (text >= 6) & (text <= 17),
                "eval_medium_18_to_43": (text >= 18) & (text <= 43),
                "eval_long_44_to_72": (text >= 44) & (text <= 72),
                "above_eval_73_plus": text >= 73}.items()},
        longest_speech_quartile=dict(
            threshold_inclusive=threshold,
            threshold_rule="sorted speech_lengths[int(0.75*(N-1))], exactly matching BucketSampler; ties can exceed 25% of samples",
            **group_stats(lengths[:, 1] >= threshold, counts, targets)),
        exposure_multiplicity_histogram={str(int(value)): int(number)
                                        for value, number in zip(multiplicities, multiplicity_counts)},
        exposure_counts_sha256=sha256_bytes(counts.tobytes()),
        verification=dict(
            checked_logged_steps=len(logs), all_logged_fields_match=not mismatches if not args.plan else None,
            mismatch_count=len(mismatches), mismatches=mismatches[:100],
            partial_final_log_lines_ignored=partial_tails,
            checkpoint=checkpoint_verification,
            checkpoint_outside_logged_prefix=(checkpoint is not None and checkpoint_verification is None)),
        replayed_or_rolled_back_log_work=dict(
            policy=replay.get("policy"), events=replay["rollback_events"],
            discarded_logged_updates=len(replay["discarded_rows"]),
            discarded_logged_target_tokens=sum(row.get("target_tokens", 0)
                                                for row in replay["discarded_rows"]),
            discarded_work_is_counted_in_effective_budget=False,
            discarded_runtime=observed_runtime(replay["discarded_rows"])),
        runtime=observed_runtime(logs.values()) if not args.plan else None,
        runtime_all_log_observations=observed_runtime(replay["all_rows"]) if not args.plan else None)
    if args.output.exists():
        raise FileExistsError("analysis output exists; choose a fresh --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(
        evidence_status=report["evidence_status"], label=args.label, totals=totals,
        training_text_sample_shares={name: stats["exposure_sample_share"] for name, stats in report["training_length_groups"].items()},
        speech_longest_quartile_sample_share=report["longest_speech_quartile"]["exposure_sample_share"],
        speech_longest_quartile_target_share=report["longest_speech_quartile"]["exposure_target_share"],
        mismatch_count=len(mismatches), output=str(args.output)), indent=2))
    return 2 if mismatches or (checkpoint_verification and not checkpoint_verification["matched"]) else 0


if __name__ == "__main__":
    raise SystemExit(analyze(arguments()))
