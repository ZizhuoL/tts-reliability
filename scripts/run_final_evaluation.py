#!/usr/bin/env python3
"""Freeze a development-selected protocol, then explicitly run serial final evaluation.

All pre-freeze commands are stdlib-only and deny reads of final.sealed.jsonl.
The freeze operation writes a new protocol plus SHA receipt; only `run` may
open the sealed input, after verifying that receipt and all frozen bindings.
This script's existence is NOT evidence that a protocol is frozen or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/eval/final.sealed.jsonl"
TEMPLATE = ROOT / "configs/final_evaluation.template.json"
DEV_SYSTEMS = {
    "A": ("results/starter-standard-dev", "A", "standard"),
    "ANoLM": ("results/starter-no-lm-dev", "A", "no-lm"),
    "ASplit": ("results/starter-split-dev", "A", "split"),
    "B": ("results/B-standard-dev", "B", "standard"),
    "BNoLM": ("results/B-no-lm-dev", "B", "no-lm"),
    "BSplit": ("results/B-split-dev", "B", "split"),
}
REPLICATION_SYSTEMS = {
    "B184": ("results/B184-standard-dev", "B184", "standard"),
    "B184NoLM": ("results/B184-no-lm-dev", "B184", "no-lm"),
    "B184Split": ("results/B184-split-dev", "B184", "split"),
}
CHECKPOINTS = {
    "A": "models/hw4-tts-starter.pt",
    "B": "checkpoints/B-ordinary-seed183/latest.pt",
    "B184": "checkpoints/B-ordinary-seed184/latest.pt",
}
CODE = (
    "scripts/run_final_evaluation.py", "scripts/evaluate.py",
    "scripts/analyze_results.py", "scripts/analyze_strategy_costs.py",
    "scripts/analyze_failure_profiles.py", "scripts/prepare_listening_bundle.py",
    "scripts/analyze_runtime_resources.py",
    "src/tts_reliability/evaluation.py", "src/tts_reliability/model.py",
    "upstream/eecs183-hw4/hw4_util.py", "upstream/eecs183-hw4/hw4_runtime.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def local(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), f"Path escapes project: {value}")
    return path


def unsealed(path):
    path = Path(path).resolve()
    require(path != FINAL.resolve() and path.name != "final.sealed.jsonl",
            "Final text reads/hashes are forbidden before the run gate")
    return path


def sha(path, *, final_gate=False):
    path = Path(path) if final_gate else unsealed(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(unsealed(path).read_text())


def write_new(path, value):
    """Exclusive durable write; never replace existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def asset_names(folder):
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*")
                  if path.is_file() and not any(part.startswith(".") for part in path.relative_to(folder).parts))


def validate_candidate(candidate, *, allow_unselected=False):
    template = read(TEMPLATE)
    require(candidate.get("schema_version") == 1, "Unsupported candidate version")
    require(candidate.get("status") == "TEMPLATE_NOT_FROZEN",
            "Candidate must remain TEMPLATE_NOT_FROZEN; freeze creates the frozen record")
    require(set(candidate) == set(template), "Candidate keys differ from the template")
    # This runner intentionally exposes selection and budget scope, not new tuning.
    require(candidate["generation"] == template["generation"],
            "Generation settings are fixed to the completed development protocol")
    require(candidate["analysis"] == template["analysis"], "Analysis settings differ from template")
    require(candidate["policy"] == template["policy"], "Primary/failure policy changed")
    require(candidate["policy"]["primary_training_seed"] == 183 and
            candidate["policy"]["training_endpoint"] == "full_epoch_latest" and
            candidate["generation"]["seeds"] == [17, 42],
            "Primary seed, full endpoint and evaluation seeds must not change")
    output = local(candidate["output"])
    require(output.is_relative_to(ROOT / "results"), "Output must be inside results/")
    require(not output.exists(), "Final output already exists; use a fresh output before freezing")
    require(type(candidate["include_seed184_sensitivity"]) is bool, "Sensitivity flag must be boolean")
    if not candidate["include_seed184_sensitivity"]:
        require(isinstance(candidate["omit_seed184_final_reason"], str) and
                len(candidate["omit_seed184_final_reason"].strip()) >= 20,
                "Omitting seed184 final requires a concrete pre-final budget reason")
    decision = candidate["selection"]
    require(set(decision) == set(template["selection"]), "Unexpected selection fields")
    if not allow_unselected:
        require(decision["selected_system"] in DEV_SYSTEMS, "Select one of the six development systems")
        for key in ("rationale", "missingness_review"):
            require(isinstance(decision[key], str) and len(decision[key].strip()) >= 20 and
                    not re.search(r"\b(TODO|TBD)\b", decision[key], re.I),
                    f"Complete a substantive {key} from development evidence")
    return candidate


def terminal_records(folder, expected, *, verify_wav=False):
    manifest = read(folder / "manifest.json")
    rows = {}
    with unsealed(folder / "raw.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)  # Torn/invalid lines must be repaired explicitly.
            require(row.get("run_id") == manifest["run_id"], f"Mixed run identity: {folder}")
            rows[row["key"]] = row
    identities = {(str(r["id"]), r["seed"]) for r in rows.values()}
    require(len(rows) == len(identities) and identities == expected,
            f"Attempt identity mismatch: {folder}; {len(rows)} observed / {len(expected)} expected")
    require(all(r["status"] in ("complete", "error") for r in rows.values()),
            f"Pending synthesis/ASR remains in {folder}")
    if verify_wav:
        for row in rows.values():
            if row.get("wav"):
                wav = (folder / row["wav"]).resolve()
                require(wav.is_relative_to(folder.resolve()) and wav.is_file(), "Missing/escaping WAV")
                require(sha(wav) == row.get("wav_sha256"), f"Changed WAV: {wav}")
            if row["status"] == "complete":
                require(row.get("wav") and row.get("wav_sha256"), "Complete row lacks waveform evidence")
    return manifest, rows


def verify_analysis(folder):
    completion = read(folder / "completion.json")
    require(completion.get("status") == "complete" and completion.get("artifacts"),
            f"Analysis has not completed: {folder}")
    for name, digest in completion["artifacts"].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()) and sha(path) == digest,
                f"Analysis artifact changed: {path}")


def selection_check(candidate, report):
    """Check the published point-estimate rule; human failure review remains explicit."""
    systems = report["systems"]
    require(set(systems) == set(DEV_SYSTEMS), "Development overview must contain exactly six cells")
    base = systems["A"]
    eligible = []
    unequal_availability = []
    for name in list(DEV_SYSTEMS)[1:]:
        item = systems[name]
        if any(item[scope].get("scored_items") != base[scope].get("scored_items")
               for scope in ("overall",)):
            unequal_availability.append(name)
        meets = (
            item["overall"]["scored"]["wer"] is not None and
            base["overall"]["scored"]["wer"] is not None and
            item["overall"]["scored"]["wer"] < base["overall"]["scored"]["wer"] and
            all(item["by_length"][g]["scored"]["wer"] is not None and
                base["by_length"][g]["scored"]["wer"] is not None and
                item["by_length"][g]["scored"]["wer"] <= base["by_length"][g]["scored"]["wer"]
                for g in ("short", "long")) and
            item["overall"]["truncation_rate"] <= base["overall"]["truncation_rate"] and
            item["overall"]["exception_rate"] <= base["overall"]["exception_rate"]
        )
        if meets:
            eligible.append(name)
    # Equal counts can still hide different missing pairs. Force review whenever
    # ANY system is missing scored items, not only when counts differ.
    missing = any(systems[n]["overall"]["scored_items"] != 90 for n in systems)
    reviewed = candidate["selection"]["failure_robust_eligible_systems"]
    if missing:
        require(isinstance(reviewed, list) and len(reviewed) == len(set(reviewed)) and
                set(reviewed).issubset(eligible),
                "Missing ASR requires explicit failure_robust_eligible_systems from paired/sensitivity review")
        eligible = [name for name in eligible if name in reviewed]
    else:
        require(reviewed is None, "No missing scores: leave failure_robust_eligible_systems null")
    for name in eligible:
        require(isinstance(systems[name]["overall"]["generation_seconds"], (float, int)),
                "Generation seconds missing for eligible candidate tie-breaking")
    rank = {"standard": 0, "no-lm": 1, "split": 2}
    ordered = sorted(eligible, key=lambda n: (
        systems[n]["overall"]["scored"]["wer"],
        systems[n]["overall"]["generation_seconds"], rank[DEV_SYSTEMS[n][2]],
        list(DEV_SYSTEMS).index(n)))
    recommendation = ordered[0] if ordered else "A"
    require(candidate["selection"]["selected_system"] == recommendation,
            f"Selected system disagrees with development rule: expected {recommendation}, got "
            f"{candidate['selection']['selected_system']}")
    return dict(eligible_systems_in_order=ordered, selected_system=recommendation,
                missing_scores_require_human_review=missing,
                unequal_scoring_counts=unequal_availability,
                review_limitation="Mechanical point checks cannot prove the supplied missingness review. "
                                  "Inspect common-success pairs and missing-as-deletions before freezing.")


def freeze(candidate_path, frozen_dir):
    candidate = validate_candidate(read(candidate_path))
    require(not frozen_dir.exists(), "Frozen protocol directory must be new")
    bindings = {}

    def remember(path):
        path = unsealed(path)
        require(path.is_relative_to(ROOT), f"Evidence outside project: {path}")
        bindings[str(path.relative_to(ROOT))] = sha(path)

    remember(candidate_path)
    remember(TEMPLATE)
    for path in CODE:
        remember(local(path))
    for path in ("configs/development_selection_rule.json", "configs/protocol.json",
                 "data/eval/manifest.json", "data/eval/dev.jsonl",
                 "configs/reference_prompt.json", "data/reference/prompt.wav"):
        remember(local(path))
    metadata = read(local("data/eval/manifest.json"))
    require(metadata["counts"]["final"] == {"short": 20, "medium": 20, "long": 20},
            "Final metadata count changed")
    dev = [json.loads(line) for line in local("data/eval/dev.jsonl").read_text().splitlines() if line.strip()]
    require(len(dev) == len({r["id"] for r in dev}) == 45, "Development identity count changed")
    require(sha(local("data/eval/dev.jsonl")) == metadata["dev_sha256"], "Development source changed")
    expected = {(str(r["id"]), seed) for r in dev for seed in (17, 42)}
    checkpoints = {}
    shared_validation = None
    for name, validation in (("A", "starter-full-validation"), ("B", "B-full-validation"),
                             ("B184", "B184-full-validation")):
        folder = local("results/" + validation)
        result, config = read(folder / "validation.json"), read(folder / "config.json")
        require(result["examples"] == 9756 and result["target_tokens"] == 1362834 and
                config["val_max_batches"] == 0, f"Full validation incomplete: {name}")
        # Full validation selects every row; subset_seed differs by training
        # seed but must not be mistaken for different full-set actual batches.
        validation_identity = {k: config["validation"][k] for k in ("path", "bytes", "mtime_ns", "batches")}
        if shared_validation is None:
            shared_validation = validation_identity
        else:
            require(validation_identity == shared_validation,
                    f"Full validation input/batches differ: {name}")
        remember(folder / "validation.json")
        remember(folder / "config.json")
        if name != "A":
            training = local(str(Path(CHECKPOINTS[name]).parent))
            summary, settings = read(training / "summary.json"), read(training / "config.json")
            require(summary["stop_reason"] == "token_budget" and summary["completed_epochs"] == 1 and
                    summary["unique_examples"] == summary["valid_examples"] == 354780 and
                    summary["target_tokens"] == 50450231 and summary["steps"] == 44348,
                    f"Full-budget latest endpoint incomplete: {name}")
            require(settings["seed"] == (183 if name == "B" else 184) and settings["long_weight"] == 1.0,
                    f"Unexpected training seed/distribution: {name}")
            require(settings["initial_checkpoint_sha256"] == checkpoints["A"]["sha256"],
                    f"Continuation did not start from the same starter: {name}")
            remember(training / "summary.json")
            remember(training / "config.json")
        checkpoint = local(CHECKPOINTS[name])
        remember(checkpoint)
        require(bindings[CHECKPOINTS[name]] == config["checkpoint_sha256"],
                f"Latest checkpoint differs from full validation: {name}")
        checkpoints[name] = dict(path=CHECKPOINTS[name], sha256=bindings[CHECKPOINTS[name]],
                                 endpoint="starter" if name == "A" else "full_epoch_latest",
                                 training_seed=None if name == "A" else (183 if name == "B" else 184))
    all_dev = dict(DEV_SYSTEMS, **REPLICATION_SYSTEMS)
    packages = None
    asset_inventory = {}
    for name, (folder_name, checkpoint_name, strategy) in all_dev.items():
        folder = local(folder_name)
        manifest, rows = terminal_records(folder, expected, verify_wav=True)
        require(manifest["checkpoint_sha256"] == checkpoints[checkpoint_name]["sha256"] and
                manifest["texts_sha256"] == metadata["dev_sha256"],
                f"Development input identity mismatch: {name}")
        settings = manifest["settings"]
        generation = candidate["generation"]
        for key, value in generation.items():
            actual = not settings["no_cache"] if key == "use_cache" else settings[key]
            require(actual == value, f"Development generation mismatch: {name}/{key}")
        require(settings["no_lm_prompt"] == (strategy == "no-lm") and
                settings["split_sentences"] == (strategy == "split"), f"Strategy mismatch: {name}")
        require(manifest["prompt_sha256"] == bindings["data/reference/prompt.wav"] and
                manifest["prompt_metadata_sha256"] == bindings["configs/reference_prompt.json"],
                f"Reference changed: {name}")
        if packages is None:
            packages = manifest["packages"]
        require(manifest["packages"] == packages and
                all(importlib.metadata.version(key) == value for key, value in packages.items()),
                f"Installed package versions differ from development: {name}")
        for path, digest in manifest["implementation"].items():
            remember(local(path))
            require(bindings[path] == digest, f"Implementation differs from dev run: {name}/{path}")
        for asset_group, directory in (("cosy_assets", "models/cosy"), ("whisper_assets", "models/whisper")):
            if directory not in asset_inventory:
                asset_inventory[directory] = asset_names(local(directory))
            require(asset_inventory[directory] == sorted(manifest[asset_group]),
                    f"Asset inventory differs from dev: {directory}")
            for asset, digest in manifest[asset_group].items():
                path = local(directory + "/" + asset)
                if str(path.relative_to(ROOT)) not in bindings:
                    remember(path)
                require(bindings[str(path.relative_to(ROOT))] == digest, f"Asset differs from dev: {path}")
        for filename in ("raw.jsonl", "manifest.json", "summary.json"):
            remember(folder / filename)
    analyses = ["development-comparisons", "development-within-B",
                "development-training-no-lm", "development-training-split",
                "replication-standard", "replication-no-lm", "replication-split"]
    for name in analyses:
        folder = local("results/analysis/" + name)
        verify_analysis(folder)
        for path in folder.iterdir():
            if path.is_file():
                remember(path)
    overview = local("results/analysis/development-comparisons")
    overview_manifest = read(overview / "analysis_manifest.json")
    for name, (folder_name, _, _) in DEV_SYSTEMS.items():
        source = overview_manifest["inputs"][name]
        require(source["directory"] == str(local(folder_name)) and
                source["raw_sha256"] == bindings[folder_name + "/raw.jsonl"] and
                source["deduplicated_items"] == 90 and not source["ignored_torn_final_line"],
                f"Selection analysis is stale or incomplete: {name}")
    execution = local("results/B-seed184-execution/completion.json")
    require(read(execution)["status"] == "complete", "Seed184 pipeline not complete")
    remember(execution)
    selection = selection_check(candidate, read(local("results/analysis/development-comparisons/summary.json")))
    cells = [dict(name=name, checkpoint=model, strategy=strategy,
                  role="prespecified_six_cell" if name in DEV_SYSTEMS else "training_seed_sensitivity")
             for name, (_, model, strategy) in all_dev.items()
             if name in DEV_SYSTEMS or candidate["include_seed184_sensitivity"]]
    protocol = dict(schema_version=1, status="FROZEN_BEFORE_FINAL_READ", frozen_utc=utc(),
                    candidate=candidate, selection_check=selection, checkpoints=checkpoints, cells=cells,
                    expected=dict(texts=60, seeds=[17, 42], attempts_per_cell=120,
                                  length_counts={"short": 20, "medium": 20, "long": 20}),
                    final_input=dict(path="data/eval/final.sealed.jsonl",
                                     sha256_from_preexisting_metadata=metadata["final_sha256"],
                                     contents_read_during_freeze=False),
                    bindings=bindings, packages=packages, asset_inventory=asset_inventory,
                    primary_comparison=None if selection["selected_system"] == "A" else
                    ["A", selection["selected_system"]],
                    always_report_training_comparison=["A", "B"],
                    training_comparison_role="primary_selected_comparison" if selection["selected_system"] == "B"
                    else "prespecified_research_comparison_exploratory",
                    all_other_contrasts="exploratory; training-seed sensitivity is secondary",
                    fallback="If selected A, report fallback/no selected improvement; do not invent an A-vs-A test.")
    frozen_dir.mkdir(parents=True, exist_ok=False)
    path = frozen_dir / "protocol.json"
    write_new(path, protocol)
    write_new(frozen_dir / "protocol.sha256.json", dict(
        protocol_sha256=sha(path), written_utc=utc(), final_input_read=False,
        status="FREEZE_RECEIPT", note="Integrity receipt, not a cryptographic timestamp or access control."))
    print(json.dumps(dict(status="frozen", protocol=str(path), sha256=sha(path),
                          final_read=False, cells=len(cells)), ensure_ascii=False))


def load_frozen(directory, *, bindings=False):
    path = directory / "protocol.json"
    receipt = read(directory / "protocol.sha256.json")
    require(receipt["status"] == "FREEZE_RECEIPT" and not receipt["final_input_read"],
            "Missing freeze receipt")
    require(sha(path) == receipt["protocol_sha256"], "Frozen protocol hash mismatch")
    protocol = read(path)
    require(protocol["status"] == "FROZEN_BEFORE_FINAL_READ" and
            protocol["final_input"]["contents_read_during_freeze"] is False, "Protocol is not frozen")
    if bindings:
        for relative, digest in protocol["bindings"].items():
            require(sha(local(relative)) == digest, f"Frozen input changed: {relative}")
        require(all(importlib.metadata.version(key) == value for key, value in protocol["packages"].items()),
                "Installed package versions changed after freezing")
        for directory, names in protocol["asset_inventory"].items():
            require(asset_names(local(directory)) == names, f"Asset inventory changed after freezing: {directory}")
    return protocol, receipt["protocol_sha256"]


def commands(protocol):
    """Pure plan construction: never read evaluation texts."""
    py = str(ROOT / ".venv/bin/python")
    out = local(protocol["candidate"]["output"])
    generation = protocol["candidate"]["generation"]
    analysis = protocol["candidate"]["analysis"]
    stages = []
    for cell in protocol["cells"]:
        name = cell["name"]
        output = out / "runs" / name
        args = [py, "-u", str(ROOT / "scripts/evaluate.py"),
                "--checkpoint", str(local(protocol["checkpoints"][cell["checkpoint"]]["path"])),
                "--texts", str(FINAL), "--output", str(output),
                "--prompt", str(local("data/reference/prompt.wav")),
                "--prompt-metadata", str(local("configs/reference_prompt.json")),
                "--cosy-dir", str(local("models/cosy")), "--whisper-dir", str(local("models/whisper"))]
        for key, value in generation.items():
            if key == "use_cache":
                require(value is True, "Validated cache must stay enabled")
            else:
                args += ["--" + key.replace("_", "-")]
                args += list(map(str, value)) if isinstance(value, list) else [str(value)]
        if cell["strategy"] != "standard":
            args.append("--no-lm-prompt" if cell["strategy"] == "no-lm" else "--split-sentences")
        stages.append(dict(name="eval-" + name, kind="evaluation", system=name, output=str(output), argv=args))
    names = [cell["name"] for cell in protocol["cells"]]

    def systems_args(selected, directory=None):
        return [item for n in selected for item in
                ("--system", f"{n}={directory or out / 'runs' / n}")]

    def analyze(label, selected, baseline):
        folder = out / "analysis" / label
        argv = [py, str(ROOT / "scripts/analyze_results.py"), *systems_args(selected),
                "--baseline", baseline, "--output", str(folder), "--bootstrap", str(analysis["bootstrap"]),
                "--seed", str(analysis["seed"]), "--plots"]
        stages.append(dict(name="analysis-" + label, kind="offline", output=str(folder), argv=argv))
        return folder

    overview = analyze("overview", names, "A")
    analyze("within-B183", ["B", "BNoLM", "BSplit"], "B")
    analyze("training-no-lm", ["ANoLM", "BNoLM"], "ANoLM")
    analyze("training-split", ["ASplit", "BSplit"], "ASplit")
    if "B184" in names:
        for label, pair in (("standard", ["B", "B184"]), ("no-lm", ["BNoLM", "B184NoLM"]),
                            ("split", ["BSplit", "B184Split"])):
            analyze("seed-sensitivity-" + label, pair, pair[0])
        analyze("within-B184", ["B184", "B184NoLM", "B184Split"], "B184")
    for label, selected, baseline in (("all", names, "A"), ("within-B183", ["B", "BNoLM", "BSplit"], "B")):
        folder = out / "costs" / label
        stages.append(dict(name="costs-" + label, kind="offline", output=str(folder), argv=[
            py, str(ROOT / "scripts/analyze_strategy_costs.py"), *systems_args(selected),
            "--baseline", baseline, "--expected-items", "120", "--output", str(folder)]))
    resources = out / "resources"
    stages.append(dict(name="runtime-resources", kind="offline", output=str(resources), argv=[
        py, str(ROOT / "scripts/analyze_runtime_resources.py"), *systems_args(names),
        "--expected-items", "120", "--output", str(resources)]))
    for name in names:
        folder = out / "failures" / name
        stages.append(dict(name="failures-" + name, kind="offline", output=str(folder), argv=[
            py, str(ROOT / "scripts/analyze_failure_profiles.py"), "--input",
            str(overview / (name + ".rescored.jsonl")), "--output", str(folder), "--expected-items", "120",
            "--bootstrap", str(analysis["failure_bootstrap"]), "--seed", str(analysis["failure_seed"]),
            "--tail-minimum", str(analysis["tail_minimum"])]))
    listen_args = [py, str(ROOT / "scripts/prepare_listening_bundle.py"), *systems_args(names, overview),
                   "--output", str(out / "listening"), "--per-category", str(analysis["listening_per_category"]),
                   "--audio-mode", "copy"]
    pair_folders = [("B", "BNoLM", "within-B183"), ("B", "BSplit", "within-B183"),
                    ("ANoLM", "BNoLM", "training-no-lm"), ("ASplit", "BSplit", "training-split")]
    if "B184" in names:
        pair_folders += [("B", "B184", "seed-sensitivity-standard"),
                         ("BNoLM", "B184NoLM", "seed-sensitivity-no-lm"),
                         ("BSplit", "B184Split", "seed-sensitivity-split"),
                         ("B184", "B184NoLM", "within-B184"), ("B184", "B184Split", "within-B184")]
    for base, method, folder in pair_folders:
        listen_args += ["--pair", f"{base}:{method}={out / 'analysis' / folder / f'{base}_vs_{method}.examples.jsonl'}"]
    stages.append(dict(name="listening", kind="offline", output=str(out / "listening"), argv=listen_args))
    return stages


def active_gpu_candidates():
    result = subprocess.run(["ps", "-axo", "pid=,command="], text=True, capture_output=True, check=True)
    patterns = ("scripts/train.py", "scripts/evaluate.py", "scripts/smoke_audio.py",
                "scripts/check_generation.py", "scripts/check_audio_components.py",
                "scripts/diagnose_prompt.py", "scripts/run_startup_queue.py")
    queues = ("scripts/run_development_comparisons.sh", "scripts/run_B_seed184_replication.sh",
              "scripts/run_baseline_and_training.sh", "scripts/resume_B_memory075.sh",
              "scripts/resume_B_after_oom.sh", "scripts/probe_B_recovery_memory.sh")
    matches = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        if (any(pattern in fields[1] for pattern in patterns) and re.search(r"\bpython[\d.]*\b", fields[1])) or \
                any(queue in fields[1] for queue in queues):
            matches.append(line.strip())
    return matches


def inventory(folder):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != ".lock"}


def run(directory, *, resume, confirm_gpu_idle):
    require(confirm_gpu_idle, "Run requires --confirm-gpu-idle after coordinator releases the GPU")
    require(not active_gpu_candidates(), "A known training/evaluation process is still active; do not overlap GPU work")
    # This lock serializes this runner only; legacy scripts do not obey it.
    with (ROOT / ".final-evaluation.gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, digest = load_frozen(directory, bindings=True)
        stages = commands(protocol)
        out = local(protocol["candidate"]["output"])
        if out.exists():
            require(resume, "Output exists; overwrite refused. Use explicit --resume for the same frozen run")
            require(read(out / "run-binding.json")["protocol_sha256"] == digest, "Resume protocol mismatch")
        else:
            require(not resume, "--resume requires an existing bound run")
            out.mkdir(parents=True, exist_ok=False)
            write_new(out / "run-binding.json", dict(
                protocol_sha256=digest, protocol=str(directory.resolve()), created_utc=utc(),
                expected_attempts_per_cell=120, stage_plan=stages))
            write_new(out / "protocol.json", protocol)
        require(not (out / "completion.json").exists(), "Run already complete; refuses rerun")
        (out / "logs").mkdir(exist_ok=True)
        (out / "stage-markers").mkdir(exist_ok=True)
        # Persist an authorization audit BEFORE the very first final-file open.
        event = out / "logs" / f"run-{time.time_ns()}.jsonl"
        with event.open("x") as stream:
            def log(value):
                stream.write(json.dumps(dict(utc=utc(), **value), ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            log(dict(event="frozen_protocol_verified_before_final_read", protocol_sha256=digest,
                     GPU_coordination="coordinator confirmed idle; known-process scan + runner lock",
                     resume=resume))
            final_bytes = FINAL.read_bytes()  # First permitted final read; gate/receipt above.
            require(hashlib.sha256(final_bytes).hexdigest() ==
                    protocol["final_input"]["sha256_from_preexisting_metadata"], "Final source hash changed")
            texts = [json.loads(line) for line in final_bytes.splitlines() if line.strip()]
            require(len(texts) == len({r["id"] for r in texts}) == 60, "Final needs 60 unique IDs")
            require(Counter(r["length_group"] for r in texts) == protocol["expected"]["length_counts"],
                    "Final length counts changed")
            expected = {(str(row["id"]), seed) for row in texts for seed in (17, 42)}
            log(dict(event="final_input_verified", texts=60, expected_attempts_per_cell=120,
                     final_sha256=hashlib.sha256(final_bytes).hexdigest()))
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               TOKENIZERS_PARALLELISM="false", HF_HOME=str(ROOT / ".cache/huggingface"),
                               XDG_CACHE_HOME=str(ROOT / ".cache"), MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
                               PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT / "upstream/eecs183-hw4"))
            for stage in stages:
                marker = out / "stage-markers" / (stage["name"] + ".json")
                folder = Path(stage["output"])
                if marker.exists():
                    previous = read(marker)
                    require(previous["stage"] == stage and previous["artifacts"] == inventory(folder),
                            f"Completed stage changed: {stage['name']}")
                    log(dict(event="verified_completed_stage", stage=stage["name"]))
                    continue
                require(stage["kind"] == "evaluation" or not folder.exists(),
                        f"Partial offline output preserved: {folder}. Move it aside explicitly before resuming; never overwrite.")
                require(not active_gpu_candidates(), "Another known GPU process appeared; stopping before next stage")
                logfile = out / "logs" / f"{stage['name']}-{time.time_ns()}.log"
                log(dict(event="stage_start", stage=stage["name"], argv=stage["argv"], log=str(logfile)))
                started = time.perf_counter()
                with logfile.open("x") as child_log:
                    result = subprocess.run(stage["argv"], cwd=ROOT, env=environment,
                                            stdout=child_log, stderr=subprocess.STDOUT)
                elapsed = time.perf_counter() - started
                log(dict(event="stage_exit", stage=stage["name"], returncode=result.returncode,
                         subprocess_wall_seconds=elapsed))
                require(result.returncode == 0, f"Stage failed; evidence retained: {stage['name']}, see {logfile}")
                if stage["kind"] == "evaluation":
                    manifest, rows = terminal_records(folder, expected, verify_wav=True)
                    require(manifest["texts_sha256"] == hashlib.sha256(final_bytes).hexdigest(),
                            "Evaluation source differs from frozen final")
                    cell = next(c for c in protocol["cells"] if c["name"] == stage["system"])
                    require(manifest["checkpoint_sha256"] == protocol["checkpoints"][cell["checkpoint"]]["sha256"],
                            "Evaluation checkpoint differs from frozen protocol")
                    log(dict(event="terminal_attempts_verified", system=stage["system"], attempts=len(rows),
                             status_counts=dict(Counter(row["status"] for row in rows.values()))))
                elif stage["name"].startswith("analysis-"):
                    verify_analysis(folder)
                elif stage["name"] == "runtime-resources":
                    require(read(folder / "completion.json")["status"] == "resource_audit_complete",
                            "Runtime resource audit did not complete")
                write_new(marker, dict(stage=stage, completed_utc=utc(), artifacts=inventory(folder),
                                       subprocess_wall_seconds=elapsed))
            # Classification is copied from the frozen selection, never recomputed on final results.
            comparison_roles = dict(
                primary_comparison=protocol["primary_comparison"],
                primary_outcomes=protocol["candidate"]["policy"]["primary_outcomes"],
                always_report_training_comparison=["A", "B"],
                training_comparison_role=protocol["training_comparison_role"],
                development_fallback=protocol["primary_comparison"] is None,
                other_contrasts=protocol["all_other_contrasts"],
                no_reselection=True, human_listening_performed=False)
            roles_path = out / "comparison-roles.json"
            if roles_path.exists():
                require(read(roles_path) == comparison_roles,
                        "Existing comparison roles differ from the frozen protocol; refusing overwrite")
                log(dict(event="verified_existing_comparison_roles"))
            else:
                write_new(roles_path, comparison_roles)
            log(dict(event="all_stages_completed", protocol_sha256=digest, stages=len(stages)))
        write_new(out / "completion.json", dict(
            status="complete", protocol_sha256=digest, cells=len(protocol["cells"]),
            expected_attempts_per_cell=120, completed_utc=utc(),
            stage_markers={p.name: sha(p) for p in sorted((out / "stage-markers").glob("*.json"))},
            cost_note="Stage subprocess wall times include loading and hashing; failed/resumed invocations remain in logs. "
                      "Per-item generation/ASR totals have separate denominators; neither is whole-project wall time."))
        print(json.dumps(dict(status="complete", output=str(out), cells=len(protocol["cells"]))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("template-check", help="Validate shape only; no prerequisite or final reads")
    check.add_argument("--candidate", type=Path, default=TEMPLATE)
    freezing = sub.add_parser("freeze", help="Verify completed dev/replication; create new frozen protocol and receipt")
    freezing.add_argument("--candidate", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True, help="New protocol directory inside project")
    planning = sub.add_parser("plan", help="Print commands from an existing frozen receipt; no final reads or GPU")
    planning.add_argument("--protocol-dir", type=Path, required=True)
    running = sub.add_parser("run", help="Explicit final read/generation, only after frozen receipt validation")
    running.add_argument("--protocol-dir", type=Path, required=True)
    running.add_argument("--confirm-gpu-idle", action="store_true")
    running.add_argument("--resume", action="store_true", help="Same binding only; preserve errors and original WAVs")
    args = parser.parse_args(argv)
    try:
        if args.command == "template-check":
            validate_candidate(read(args.candidate), allow_unselected=True)
            print(json.dumps(dict(status="template_valid_NOT_frozen", final_read=False, GPU_used=False)))
        elif args.command == "freeze":
            freeze(local(args.candidate), local(args.output))
        elif args.command == "plan":
            protocol, digest = load_frozen(local(args.protocol_dir))
            print(f"# Frozen protocol SHA256: {digest}; primary: {protocol['primary_comparison']}")
            for stage in commands(protocol):
                print(shlex.join(stage["argv"]))
        else:
            run(local(args.protocol_dir), resume=args.resume, confirm_gpu_idle=args.confirm_gpu_idle)
    except (ValueError, KeyError, FileNotFoundError, BlockingIOError) as error:
        parser.exit(2, f"Refused: {error}\n")


if __name__ == "__main__":
    main()
