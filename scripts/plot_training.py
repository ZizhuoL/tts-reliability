#!/usr/bin/env python3
"""Snapshot local training logs and render static PNG/PDF research figures.

No GPU or held-out text access. A trailing window covers supervised target
tokens, including EOS. Partial boundary batches contribute proportionally.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from analyze_training_exposure import read_logs

COLORS = ["#006BA4", "#E66100", "#4C9F38", "#9B59B6", "#A34B53", "#008C95"]
GIB = 1024 ** 3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trailing_token_mean(values, weights, window):
    """Exact piecewise-constant integral over the last `window` target tokens."""
    values, weights = np.asarray(values, float), np.asarray(weights, np.int64)
    if np.any(weights <= 0) or window <= 0:
        raise ValueError("positive target counts and window required")
    ends = np.cumsum(weights)
    lower = np.maximum(0, ends - window)
    weighted_prefix = np.r_[0.0, np.cumsum(values * weights)]
    token_prefix = np.r_[0, ends]
    first = np.searchsorted(ends, lower, side="right")
    excluded = weighted_prefix[first] + (lower - token_prefix[first]) * values[first]
    span = ends - lower
    return (weighted_prefix[1:] - excluded) / span, span


def snapshot(path, destination, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return dict(
        original_path=str(path.resolve()), snapshot_path=str(destination.resolve()),
        sha256=hashlib.sha256(content).hexdigest(), snapshot_bytes=len(content),
        source_size_before=before.st_size, source_size_after=after.st_size,
        changed_during_snapshot=before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns)


def retained_recovery_boundaries(events, accepted_rows):
    """Map surviving resume boundaries to retained tokens, never wall-clock time.

    A later rollback at an earlier step erases later boundaries from the
    accepted trajectory. Repeated retries at one step share one marker.
    """
    surviving, by_step = set(), {}
    for event in events:
        step = event["resumed_at_step"]
        by_step.setdefault(step, []).append(event)
        surviving = {previous for previous in surviving if previous < step}
        surviving.add(step)
    boundaries = []
    for step in sorted(surviving):
        if step not in accepted_rows:
            continue
        tokens_before = accepted_rows[step - 1]["total_target_tokens"] if step > 1 else 0
        matching = by_step[step]
        boundaries.append(dict(
            resumed_at_step=step, retained_tokens_before_resume=tokens_before,
            x_million_targets=tokens_before / 1e6, rollback_count=len(matching),
            discarded_update_rows=sum(len(event["discarded_steps"]) for event in matching),
            rollback_source_lines=[event["line"] for event in matching],
            coordinate_interpretation="Boundary before first accepted resumed update; not a measured restart timestamp."))
    return boundaries


def load_run(name, directory, output, window, force_partial, index):
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    folder = output / "inputs" / f"{index:02d}-{slug}"
    files = {}
    for filename in ("config.json", "train.jsonl", "validation.jsonl", "summary.json"):
        files[filename] = snapshot(directory / filename, folder / filename,
                                   required=filename in ("config.json", "train.jsonl"))
    config = json.loads((folder / "config.json").read_text())
    logs, _, partial, replay = read_logs([folder / "train.jsonl"])
    if not logs:
        raise ValueError(f"{name}: no complete training update rows")
    frontier = max(logs)
    if sorted(logs) != list(range(1, frontier + 1)):
        raise ValueError(f"{name}: effective log branch has missing steps")
    rows = [logs[step] for step in range(1, frontier + 1)]
    weights = np.array([row["target_tokens"] for row in rows], dtype=np.int64)
    cumulative = np.cumsum(weights)
    recorded = np.array([row["total_target_tokens"] for row in rows], dtype=np.int64)
    if not np.array_equal(cumulative, recorded):
        raise ValueError(f"{name}: cumulative target counters differ from effective logged batches")
    for metric in ("loss", "accuracy", "learning_rate"):
        if not np.isfinite([row[metric] for row in rows]).all():
            raise ValueError(f"{name}: nonfinite {metric} in snapshot")
    loss, span = trailing_token_mean([row["loss"] for row in rows], weights, window)
    accuracy, _ = trailing_token_mean([row["accuracy"] for row in rows], weights, window)
    budget = config["effective_token_budget"]
    summary = json.loads((folder / "summary.json").read_text()) if files["summary.json"] else None
    budget_complete = bool(
        summary and summary.get("stop_reason") == "token_budget"
        and summary.get("steps") == frontier and cumulative[-1] >= budget)
    partial_snapshot = force_partial or not budget_complete
    validation, validation_replay, validation_partial, ignored_validation = [], None, [], []
    if files["validation.jsonl"]:
        val_logs, _, validation_partial, validation_replay = read_logs([folder / "validation.jsonl"])
        for step, row in sorted(val_logs.items()):
            if step not in logs:
                ignored_validation.append(step)
                continue
            validation.append(dict(row, total_target_tokens=logs[step]["total_target_tokens"]))
    recovery_boundaries = retained_recovery_boundaries(replay["rollback_events"], logs)
    first_memory = rows[0].get("memory_before_empty_cache", rows[0])
    first_swap = first_memory.get("system_swap_used_bytes")
    table = []
    for position, row in enumerate(rows):
        pre = row.get("memory_before_empty_cache", {})
        post = row.get("memory_after_empty_cache", row)
        swap = row.get("system_swap_used_bytes")
        table.append(dict(
            step=row["step"], total_target_tokens=int(cumulative[position]),
            target_tokens=row["target_tokens"], loss=row["loss"], accuracy=row["accuracy"],
            rolling_loss=float(loss[position]), rolling_accuracy=float(accuracy[position]),
            rolling_window_effective_tokens=int(span[position]),
            learning_rate=row["learning_rate"], padding_fraction=row.get("padding_fraction"),
            mps_active_gib=row.get("mps_allocated_bytes", np.nan) / GIB,
            mps_driver_before_clear_gib=pre.get("mps_driver_bytes", np.nan) / GIB,
            mps_driver_after_clear_gib=post.get("mps_driver_bytes", np.nan) / GIB,
            host_swap_change_gib=(swap - first_swap) / GIB if swap is not None and first_swap is not None else np.nan))
    csv_path = output / f"{index:02d}-{slug}-metrics.csv"
    with csv_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=table[0])
        writer.writeheader()
        writer.writerows(table)
    manifest = dict(
        name=name, directory=str(directory.resolve()), input_snapshots=files,
        status="PARTIAL_SNAPSHOT" if partial_snapshot else "BUDGET_COMPLETE_SNAPSHOT",
        observed_steps=frontier, observed_target_tokens=int(cumulative[-1]),
        configured_target_budget=budget, budget_fraction=float(cumulative[-1] / budget),
        last_logged_sampler_epoch=rows[-1].get("epoch"),
        last_logged_epoch_batch_cursor=rows[-1].get("epoch_batch_cursor"),
        validation_points=len(validation),
        plotted_recovery_boundaries=recovery_boundaries,
        validation_scope=config.get("validation", {}).get("purpose", "course dev; not independent final test"),
        configured_validation_max_batches=config.get("val_max_batches"),
        integrity=dict(
            contiguous_effective_steps=True, cumulative_target_counts_match=True,
            ignored_partial_train_tail=partial, training_rollback_events=replay["rollback_events"],
            discarded_training_update_rows=len(replay["discarded_rows"]),
            ignored_partial_validation_tail=validation_partial,
            validation_rollback_events=validation_replay["rollback_events"] if validation_replay else [],
            ignored_validation_steps_outside_effective_train_branch=ignored_validation,
            full_sampler_exposure_reconstruction_performed=False,
            notes="Counter checks only. Use analyze_training_exposure.py for independent length/sampler reconstruction."))
    return dict(name=name, rows=rows, table=table, validation=validation, manifest=manifest,
                x=cumulative / 1e6, loss=loss, accuracy=accuracy,
                recovery_boundaries=recovery_boundaries)


def format_axes(axes):
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDE3E9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.set_xlabel("Cumulative supervised target tokens (millions, including EOS)")


def annotate_recoveries(axes, runs):
    """One dashed line per shared retained-token coordinate on every axis."""
    markers = {}
    for run in runs:
        for boundary in run["recovery_boundaries"]:
            markers.setdefault(boundary["retained_tokens_before_resume"], []).append((run["name"], boundary))
    for tokens, entries in sorted(markers.items()):
        labels = []
        for name, boundary in entries:
            prefix = f"{name}: " if len(runs) > 1 else ""
            count = boundary["rollback_count"]
            labels.append(f"{prefix}resume step {boundary['resumed_at_step']} | {count} rollback{'s' if count != 1 else ''}")
        for axis in axes:
            x = tokens / 1e6
            axis.axvline(x, color="#985D45", linestyle=(0, (4, 3)), linewidth=1.05, alpha=.9, zorder=4)
            left, right = axis.get_xlim()
            alignment = "right" if x > (left + right) / 2 else "left"
            # Place labels outside the data region so no curve or point is hidden.
            axis.annotate("\n".join(labels), xy=(x, 1.015), xycoords=axis.get_xaxis_transform(),
                          xytext=(0, 2), textcoords="offset points", ha=alignment, va="bottom",
                          fontsize=7.5, color="#79452F", annotation_clip=False)


def finish_figure(figure, title, subtitle, footnote, output, stem, recovery=False):
    figure.suptitle(title, x=0.11, ha="left", y=0.985, fontsize=15, fontweight="bold")
    figure.text(0.11, 0.945, subtitle, ha="left", va="top", fontsize=9, color="#425466")
    if recovery:
        footnote += ("\nDashed: recovery boundary on the retained-token axis, not a restart time. Failed old tails are excluded from this axis;"
                     "\nactual costs are counted separately (B-recovery-costs.json). Memory changes alone do not establish restart causality.")
    figure.text(0.11, 0.018, footnote, ha="left", va="bottom", fontsize=8, color="#425466")
    figure.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.19 if recovery else 0.13, hspace=0.46)
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        figure.savefig(path, dpi=180, facecolor="white")
        paths.append(path)
    plt.close(figure)
    return paths


def render(runs, args):
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "axes.titlelocation": "left"})
    statuses = [run["manifest"]["status"] for run in runs]
    status = "PARTIAL SNAPSHOT" if "PARTIAL_SNAPSHOT" in statuses else "BUDGET-COMPLETE SNAPSHOT"
    subtitle = " | ".join(
        f"{run['name']}: {run['manifest']['observed_target_tokens'] / 1e6:.3f}M targets "
        f"({run['manifest']['budget_fraction']:.1%} of budget)" for run in runs)
    if len(subtitle) > 130:
        subtitle = "\n".join(subtitle.split(" | "))
    window = f"{args.window_tokens:,}"
    paths = []
    has_recovery = any(run["recovery_boundaries"] for run in runs)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.2))
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        choose = np.linspace(0, len(run["rows"]) - 1, min(args.max_raw_points, len(run["rows"])), dtype=int)
        for axis, metric, smooth in zip(axes, ("loss", "accuracy"), (run["loss"], run["accuracy"])):
            raw = np.array([row[metric] for row in run["rows"]])
            axis.scatter(run["x"][choose], raw[choose], color=color, s=5, alpha=0.10,
                         linewidths=0, rasterized=True)
            axis.plot(run["x"], smooth, color=color, linewidth=1.6, label=f"{run['name']} train, token-weighted")
            if run["validation"]:
                axis.scatter([row["total_target_tokens"] / 1e6 for row in run["validation"]],
                             [row[metric] for row in run["validation"]], marker="D",
                             s=38, edgecolors=color, facecolors="white", linewidths=1.3,
                             label=f"{run['name']} course-dev", zorder=5)
    axes[0].set_ylabel("Cross entropy (nats / target)")
    axes[0].set_title("Teacher-forced token prediction")
    axes[1].set_ylabel("Next-token accuracy")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, loc="best", fontsize=8)
    if not any(run["validation"] for run in runs):
        axes[0].text(0.02, 0.05, "No course-dev validation points recorded in this snapshot.",
                     transform=axes[0].transAxes, fontsize=8, color="#425466")
    format_axes(axes)
    if has_recovery:
        annotate_recoveries(axes, runs)
    paths += finish_figure(
        fig, f"Training trajectory — {status}", subtitle,
        f"Solid: trailing {window} target-token window (shorter at startup), weighted by target count; partial batches prorated.\n"
        "Faint points: raw batch metrics; length composition changes their variability. Course-dev points are a separate dataset.",
        args.output, "training-metrics", recovery=has_recovery)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9.6))
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        axes[0].plot(run["x"], [row["learning_rate"] for row in run["rows"]],
                     color=color, linewidth=1.5, label=run["name"])
        choose = np.linspace(0, len(run["rows"]) - 1, min(args.max_raw_points, len(run["rows"])), dtype=int)
        for axis, metric in zip(axes[1:], ("target_tokens", "padding_fraction")):
            values = np.array([row[metric] for row in run["rows"]])
            axis.scatter(run["x"][choose], values[choose], s=6, alpha=0.32,
                         color=color, linewidths=0, rasterized=True)
    axes[0].set_ylabel("Learning rate")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("Targets in each batch")
    axes[2].set_ylabel("Padding fraction")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    format_axes(axes)
    if has_recovery:
        annotate_recoveries(axes, runs)
    paths += finish_figure(
        fig, f"Optimization & batches — {status}", subtitle,
        "Batch targets include EOS; long batches contribute more weight to the training-metric window.\n"
        f"Raw scatter is uniformly thinned to at most {args.max_raw_points:,} observations per run; complete data are in the CSV files.",
        args.output, "optimization-and-batches", recovery=has_recovery)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8.2))
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        for key, style, alpha, suffix in (
                ("mps_active_gib", ":", 0.95, "active"),
                ("mps_driver_after_clear_gib", "-", 0.85, "driver / after clear"),
                ("mps_driver_before_clear_gib", "--", 0.35, "driver before clear")):
            if args.hide_cache_series and key == "mps_driver_before_clear_gib":
                continue
            values = np.array([row[key] for row in run["table"]])
            if np.isfinite(values).any():
                axes[0].plot(run["x"], values, style, color=color, alpha=alpha,
                             linewidth=0.85, label=f"{run['name']} {suffix}")
                if key == "mps_driver_before_clear_gib":
                    maximum = int(np.nanargmax(values))
                    axes[0].scatter([run["x"][maximum]], [values[maximum]], color=color, marker="x", s=30, zorder=5)
        swap = np.array([row["host_swap_change_gib"] for row in run["table"]])
        if np.isfinite(swap).any():
            axes[1].plot(run["x"], swap, color=color, linewidth=1.2, label=run["name"])
    axes[0].set_ylabel("MPS memory (GiB)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    axes[1].set_ylabel("Host swap change (GiB)")
    axes[1].axhline(0, color="#788796", linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8)
    format_axes(axes)
    if has_recovery:
        annotate_recoveries(axes, runs)
    paths += finish_figure(
        fig, f"Observed memory — {status}", subtitle,
        "Memory is sampled at logged updates/cache clears; these observations are not continuous system peaks.\n"
        "Swap is host-wide, relative to the first logged observation, and may include other processes. × marks highest logged pre-clear driver.",
        args.output, "observed-memory", recovery=has_recovery)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, metavar="NAME=RUN_DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-tokens", type=int, default=500_000)
    parser.add_argument("--max-raw-points", type=int, default=5000)
    parser.add_argument("--partial", action="store_true", help="Explicitly label every input as a partial snapshot")
    parser.add_argument("--hide-cache-series", action="store_true")
    args = parser.parse_args()
    if args.window_tokens <= 0 or args.max_raw_points <= 0:
        parser.error("window-tokens/max-raw-points must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("choose a fresh plot output directory; existing artifacts are preserved")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = []
    for spec in args.runs:
        if "=" not in spec:
            parser.error("each run must be NAME=RUN_DIRECTORY")
        name, directory = spec.split("=", 1)
        inputs.append((name, Path(directory)))
    if len({name for name, _ in inputs}) != len(inputs):
        parser.error("run names must be unique")
    runs = [load_run(name, directory, args.output, args.window_tokens, args.partial, index)
            for index, (name, directory) in enumerate(inputs)]
    manifest = dict(
        artifact_kind="static_training_figures", created_utc=datetime.now(timezone.utc).isoformat(),
        plotting_source_sha256=sha256(Path(__file__)),
        rollback_reader_source_sha256=sha256(ROOT / "scripts/analyze_training_exposure.py"),
        window=dict(target_tokens=args.window_tokens, weighting="supervised target count including EOS",
                    boundary="exact proportional contribution from partially covered starting batch",
                    startup="use all available targets until the full window exists"),
        partial_snapshot=any(run["manifest"]["status"] == "PARTIAL_SNAPSHOT" for run in runs),
        runs=[run["manifest"] for run in runs], figures={})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    figures = render(runs, args)
    manifest["figures"] = {path.name: dict(sha256=sha256(path), bytes=path.stat().st_size)
                           for path in figures}
    manifest["artifact_generation_complete"] = True
    manifest["training_completion_claimed"] = not manifest["partial_snapshot"]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "completion.json").write_text(json.dumps(dict(
        status="figure_generation_complete",
        training_status="partial_snapshot" if manifest["partial_snapshot"] else "budget_complete_snapshot",
        manifest_sha256=sha256(args.output / "manifest.json"),
        generated_files=[path.name for path in figures]), indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), partial_snapshot=manifest["partial_snapshot"],
                          figures=[path.name for path in figures],
                          snapshots=[dict(name=run["name"], steps=run["manifest"]["observed_steps"],
                                          targets=run["manifest"]["observed_target_tokens"])
                                     for run in runs]), indent=2))


if __name__ == "__main__":
    main()
