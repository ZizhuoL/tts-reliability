#!/bin/zsh
# Prepared serial replication; this file's existence is NOT evidence of execution.
# Default: CPU-only prerequisite check. Launch explicitly with --run after the
# coordinator has confirmed no other GPU job and sufficient remaining budget.
# No automatic restart: preserve failures and resume the individual stage manually.
# Full-epoch latest.pt is fixed in advance; never choose the better seed/best.pt.
# Periodic validation uses seed+1 (185 here vs 184 for training seed183), so those
# curves do NOT compare the same subset. Full course-dev validation is identical.
set -euo pipefail
setopt noclobber
cd "${0:A:h:h}"
source scripts/env.sh
replication_mode="${1:---check-only}"
if [[ "$#" -gt 1 || ( "$replication_mode" != "--run" && "$replication_mode" != "--check-only" ) ]]; then
  print -u2 "Usage: zsh scripts/run_B_seed184_replication.sh [--check-only|--run]"
  exit 2
fi

# Standard-library-only preflight: no torch import or checkpoint deserialization.
# Large checkpoints are streamed for SHA256 only, after small completion gates.
.venv/bin/python - "$replication_mode" <<'PY'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

def require(condition, message):
    if not condition:
        raise SystemExit("Replication prerequisite failed: " + message)

def read(path):
    path = Path(path)
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

evidence = {}
def remember(path):
    evidence[str(path)] = sha(path)

run = Path("checkpoints/B-ordinary-seed183")
summary = read(run / "summary.json")
require(summary["stop_reason"] == "token_budget" and summary["completed_epochs"] == 1,
        "seed183 has not finished its full epoch")
require(summary["target_tokens"] == 50450231 and
        summary["unique_examples"] == summary["valid_examples"] == 354780,
        "seed183 coverage/target count differs from preregistered full epoch")
remember(run / "summary.json")
reference = read(run / "config.json")
expected = dict(batch_size=8, pad_multiple=64, sequence_implementation="packed",
                empty_cache_every=1, epochs=1, steps=0, token_budget=0,
                seed=183, long_weight=1.0, learning_rate=0.0002,
                weight_decay=0.01, grad_clip=1.0, warmup_fraction=0.1,
                val_every=2000, val_max_batches=128, gradient_checkpointing=False,
                device="mps", dtype="float32", effective_token_budget=50450231)
for key, value in expected.items():
    require(reference.get(key) == value, f"seed183 configuration changed: {key}")
for filename, digest in reference["implementation_sha256"].items():
    require(sha(Path("src/tts_reliability") / filename) == digest,
            f"core implementation differs: {filename}")
for key, path in (("data_identity", Path("data/processed/train.pt")),
                  ("validation", Path("data/processed/dev.pt"))):
    identity = reference[key]
    require(str(path.resolve()) == identity["path"] and
            path.stat().st_size == identity["bytes"] and
            path.stat().st_mtime_ns == identity["mtime_ns"],
            f"prepared data identity changed: {path}")
remember(run / "config.json")
for name in ("B-full-validation", "starter-full-validation"):
    result = read(f"results/{name}/validation.json")
    config = read(f"results/{name}/config.json")
    require(result["examples"] == 9756 and config["val_max_batches"] == 0,
            f"{name} is not full course-dev validation")
    if name == "starter-full-validation":
        require(config["checkpoint_sha256"] == reference["initial_checkpoint_sha256"],
                "starter full-validation checkpoint changed")
    remember(f"results/{name}/validation.json")
    remember(f"results/{name}/config.json")
b_digest = read("results/B-full-validation/config.json")["checkpoint_sha256"]

# Real analysis completion markers are emitted only after every export succeeds.
analyses = ("development-comparisons", "development-within-B",
            "development-training-no-lm", "development-training-split")
for name in analyses:
    folder = Path("results/analysis") / name
    completion = read(folder / "completion.json")
    require(completion.get("status") == "complete" and completion.get("artifacts"),
            f"incomplete analysis: {name}")
    for filename, digest in completion["artifacts"].items():
        require(Path(filename).name == filename and sha(folder / filename) == digest,
                f"analysis artifact hash mismatch: {name}/{filename}")
    remember(folder / "completion.json")
    manifest = read(folder / "analysis_manifest.json")
    for source in manifest["inputs"].values():
        require(not source["ignored_torn_final_line"] and source["deduplicated_items"] == 90,
                f"partial analysis source: {name}")
        require(sha(Path(source["directory"]) / "raw.jsonl") == source["raw_sha256"],
                f"analysis source changed after completion: {name}")
        require(read(Path(source["directory"]) / "manifest.json") == source["manifest"],
                f"analysis input manifest changed after completion: {name}")

texts = [json.loads(line) for line in Path("data/eval/dev.jsonl").read_text().splitlines() if line.strip()]
require(len(texts) == 45 and len({str(row["id"]) for row in texts}) == 45,
        "development text set must contain 45 unique IDs")
jobs = {(str(row["id"]), seed) for row in texts for seed in (17, 42)}
common_manifest = None
for prefix in ("starter", "B"):
    for strategy in ("standard", "no-lm", "split"):
        folder = Path(f"results/{prefix}-{strategy}-dev")
        manifest = read(folder / "manifest.json")
        settings = manifest["settings"]
        require(settings["seeds"] == [17, 42] and
                settings["no_lm_prompt"] == (strategy == "no-lm") and
                settings["split_sentences"] == (strategy == "split"),
                f"strategy/seeds changed: {folder}")
        events = {}
        for line in (folder / "raw.jsonl").read_text().splitlines():
            row = json.loads(line)
            events[row["key"]] = row
        require(len(events) == 90 and
                {(str(r["id"]), r["seed"]) for r in events.values()} == jobs and
                all(r["status"] in ("complete", "error") and r["run_id"] == manifest["run_id"]
                    for r in events.values()), f"unfinished jobs: {folder}")
        # Errors are valid completed attempts, not silently filtered successes.
        require(manifest["texts_sha256"] == sha("data/eval/dev.jsonl") and
                manifest["prompt_metadata_sha256"] == sha("configs/reference_prompt.json") and
                manifest["prompt_sha256"] == sha("data/reference/prompt.wav"),
                f"development/prompt inputs differ: {folder}")
        for key, value in dict(length_budget="adaptive", max_length=500,
                               tokens_per_text=11.0, temperature=1.0, top_k=50,
                               min_length_ratio=2.0, pause_seconds=0.15,
                               no_cache=False, limit=None, lm_device="mps",
                               flow_device="mps", vocoder_device="cpu",
                               asr_device="cpu", audio_device=None, threads=8).items():
            require(settings[key] == value, f"generation setting changed: {folder}/{key}")
        if prefix == "B":
            require(manifest["checkpoint_sha256"] == b_digest,
                    f"seed183 generation differs from full-validation checkpoint: {folder}")
        else:
            require(manifest["checkpoint_sha256"] == reference["initial_checkpoint_sha256"],
                    f"starter generation checkpoint changed: {folder}")
        if common_manifest is None:
            common_manifest = manifest
        for key in ("implementation", "cosy_assets", "whisper_assets", "packages"):
            require(manifest[key] == common_manifest[key], f"mixed evaluation provenance: {key}")
        remember(folder / "manifest.json")
        remember(folder / "raw.jsonl")
for filename, digest in common_manifest["implementation"].items():
    require(sha(filename) == digest, f"evaluation implementation changed: {filename}")

# These are later pipeline stages. Require their real outputs as well as the
# hash-verified analysis markers; no claim of human listening is made.
for name in ("development-strategy-costs", "development-within-B-costs"):
    report = read(f"results/analysis/{name}/summary.json")
    require(all(item["status"] == "complete_observed_attempts"
                for item in report["systems"].values()), f"incomplete costs: {name}")
    remember(f"results/analysis/{name}/summary.json")
for name in ("A", "ANoLM", "ASplit", "B", "BNoLM", "BSplit"):
    folder = Path("results/analysis/development-failure-profiles") / name
    for filename in ("summary.json", "bootstrap.json", "manifest.json"):
        read(folder / filename)
        remember(folder / filename)
for name in ("development-comparisons", "development-within-B"):
    folder = Path("results/listening") / name
    manifest = read(folder / "manifest.json")
    require(manifest.get("available_input_items") and
            all(count == 90 for count in manifest["available_input_items"].values()) and
            (folder / "INDEX.md").is_file(), f"listening export incomplete: {name}")
    remember(folder / "manifest.json")
    remember(folder / "INDEX.md")

outputs = [Path("checkpoints/B-ordinary-seed184"),
           Path("results/B184-full-validation"), Path("results/B-seed184-execution")]
outputs += [Path(f"results/B184-{strategy}-dev") for strategy in ("standard", "no-lm", "split")]
outputs += [Path(f"results/analysis/replication-{strategy}") for strategy in ("standard", "no-lm", "split")]
logs = [Path("logs/train-B-seed184.log"), Path("logs/B184-full-validation.log")]
logs += [Path(f"logs/B184-{strategy}-dev.log") for strategy in ("standard", "no-lm", "split")]
for path in outputs + logs:
    require(not path.exists(), f"existing output is protected: {path}; resume stages manually")
starter = "models/hw4-tts-starter.pt"
require(sha(starter) == reference["initial_checkpoint_sha256"] ==
        "4cb40a7140b1055b7e077232a202f5600ce4a33eb955e8ff8ff1825b5fd54463",
        "starter checkpoint identity changed")
require(sha(run / "latest.pt") == b_digest,
        "seed183 latest checkpoint differs from completed validation/development evidence")
protocol = dict(
    status="prerequisites_verified_before_execution", created_utc=datetime.now(timezone.utc).isoformat(),
    purpose="Prespecified second training seed; does not replace seed183 or select a winning seed.",
    training_seed=184, primary_training_seed=183, sampling_seeds=[17, 42],
    checkpoint_policy="Complete one full ordinary epoch; evaluate latest.pt, never best.pt.",
    expected_examples=354780, expected_target_tokens=50450231, expected_course_dev_examples=9756,
    initial_checkpoint_sha256=reference["initial_checkpoint_sha256"],
    primary_full_epoch_checkpoint_sha256=b_digest,
    immutable_training_reference=expected,
    runtime=dict(mps_memory_fraction=0.75, empty_cache_every=1, save_every=500, save_minutes=10),
    periodic_validation=dict(seed183_subset_seed=184, seed184_subset_seed=185,
                             max_batches=128, every_steps=2000,
                             limitation="Different subsets: pointwise periodic CE/accuracy differences are not same-set comparisons. Full 9756-example validation is shared."),
    limitations=["Two training seeds do not establish a population-level training-variance estimate.",
                 "Seed183 used memory fraction0.6 then0.75 and OOM recovery; seed184 starts at0.75. Record runtime/cost differences and do not claim bitwise MPS reproducibility.",
                 "All generation settings/prompt/texts and final full-epoch checkpoint policy are fixed before seed184 metrics.",
                 "Course dev was used for checkpoint monitoring; these are development results, not independent final-test results.",
                 "12-hour train time cap is per invocation; if stopped early this pipeline refuses evaluation as a completed epoch."],
    prerequisite_sha256=evidence, script_sha256=sha("scripts/run_B_seed184_replication.sh"))
if sys.argv[1] == "--run":
    folder = Path("results/B-seed184-execution")
    folder.mkdir()  # Exclusive claim; concurrent invocation must fail.
    (folder / "replication_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
print("Prerequisites verified. " + ("Launching serial seed184 pipeline." if sys.argv[1] == "--run"
                                   else "Check only: no training, model loading, or output creation."))
PY
[[ "$replication_mode" == "--run" ]] || exit 0

caffeinate -i .venv/bin/python -u scripts/train.py \
  --mode train --checkpoint models/hw4-tts-starter.pt \
  --data data/processed/train.pt --output checkpoints/B-ordinary-seed184 \
  --device mps --batch-size 8 --pad-multiple 64 --sequence-implementation packed \
  --empty-cache-every 1 --mps-memory-fraction 0.75 \
  --epochs 1 --steps 0 --token-budget 0 --long-weight 1 --seed 184 \
  --learning-rate 0.0002 --weight-decay 0.01 --grad-clip 1 --warmup-fraction 0.1 \
  --max-hours 12 --val-data data/processed/dev.pt --val-every 2000 --val-max-batches 128 \
  --save-every 500 --save-minutes 10 --log-every 50 \
  > logs/train-B-seed184.log 2>&1

.venv/bin/python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path("checkpoints/B-ordinary-seed184/summary.json").read_text())
assert result["stop_reason"] == "token_budget" and result["completed_epochs"] == 1, result
assert result["unique_examples"] == result["valid_examples"] == 354780, result
assert result["target_tokens"] == 50450231, result
PY

caffeinate -i .venv/bin/python -u scripts/train.py \
  --mode validate --checkpoint checkpoints/B-ordinary-seed184/latest.pt \
  --val-data data/processed/dev.pt --val-max-batches 0 \
  --output results/B184-full-validation --device mps --batch-size 8 --pad-multiple 64 \
  --sequence-implementation packed --mps-memory-fraction 0.75 --seed 184 \
  > logs/B184-full-validation.log 2>&1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path("results/B184-full-validation/validation.json").read_text())
assert result["examples"] == 9756, result
PY

for strategy in standard no-lm split; do
  strategy_flags=()
  [[ "$strategy" != no-lm ]] || strategy_flags+=(--no-lm-prompt)
  [[ "$strategy" != split ]] || strategy_flags+=(--split-sentences)
  caffeinate -i .venv/bin/python -u scripts/evaluate.py \
    --checkpoint checkpoints/B-ordinary-seed184/latest.pt \
    --texts data/eval/dev.jsonl --output "results/B184-${strategy}-dev" \
    --prompt data/reference/prompt.wav --prompt-metadata configs/reference_prompt.json \
    --seeds 17 42 --lm-device mps --flow-device mps --vocoder-device cpu --asr-device cpu \
    --threads 8 --length-budget adaptive --max-length 500 --tokens-per-text 11 \
    --temperature 1 --top-k 50 --min-length-ratio 2 --pause-seconds 0.15 \
    "${strategy_flags[@]}" > "logs/B184-${strategy}-dev.log" 2>&1
  .venv/bin/python - "$strategy" <<'PY'
import json
from pathlib import Path
import sys
folder = Path(f"results/B184-{sys.argv[1]}-dev")
rows = {}
for line in (folder / "raw.jsonl").read_text().splitlines():
    row = json.loads(line)
    rows[row["key"]] = row
assert len(rows) == 90 and len({r["id"] for r in rows.values()}) == 45
assert all(r["status"] in ("complete", "error") for r in rows.values())
assert {r["seed"] for r in rows.values()} == {17, 42}
PY
  .venv/bin/python scripts/analyze_results.py \
    --system "B183=results/B-${strategy}-dev" --system "B184=results/B184-${strategy}-dev" \
    --baseline B183 --output "results/analysis/replication-${strategy}" \
    --bootstrap 2000 --seed 1832026 --plots
done

.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
artifacts = [Path("checkpoints/B-ordinary-seed184/summary.json"),
             Path("results/B184-full-validation/validation.json")]
artifacts += [Path(f"results/analysis/replication-{name}/completion.json")
              for name in ("standard", "no-lm", "split")]
for path in artifacts[2:]:
    assert json.loads(path.read_text())["status"] == "complete"
with Path("results/B-seed184-execution/completion.json").open("x") as stream:
    json.dump(dict(status="complete", scope="full ordinary epoch + full course-dev + three development strategies",
                   artifacts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}),
              stream, indent=2)
    stream.write("\n")
PY
