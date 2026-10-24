#!/bin/zsh
# Execute only after ordinary training and full validation finish successfully
# (including explicit recovery pipelines when the initial process failed).
# All jobs are sequential; final.sealed.jsonl is never opened here.
set -euo pipefail
source scripts/env.sh

.venv/bin/python - <<'PY'
import json
from pathlib import Path

training = json.loads(Path("checkpoints/B-ordinary-seed183/summary.json").read_text())
assert training["stop_reason"] == "token_budget", training
assert training["completed_epochs"] == 1, training
assert training["unique_examples"] == training["valid_examples"] == 354780, training
assert training["target_tokens"] == 50450231, training
validation = json.loads(Path("results/B-full-validation/validation.json").read_text())
assert validation["examples"] == 9756, validation
baseline = json.loads(Path("results/starter-standard-dev/summary.json").read_text())["overall"]
assert baseline["items"] == baseline["scored_items"] == 90, baseline
assert baseline["exceptional_items"] == 0, baseline
print("Full ordinary epoch, full course-dev validation and standardized A baseline verified.", flush=True)
PY

if [[ ! -f results/starter-full-validation/validation.json ]]; then
  caffeinate -i .venv/bin/python -u scripts/train.py \
    --mode validate --checkpoint models/hw4-tts-starter.pt \
    --val-data data/processed/dev.pt --val-max-batches 0 \
    --output results/starter-full-validation --batch-size 8 --pad-multiple 64 \
    --sequence-implementation packed --mps-memory-fraction 0.6 \
    > logs/starter-full-validation.log 2>&1
fi
.venv/bin/python - <<'PY'
import json
from pathlib import Path
folder = Path("results/starter-full-validation")
validation = json.loads((folder / "validation.json").read_text())
config = json.loads((folder / "config.json").read_text())
assert validation["examples"] == 9756, validation
assert config["checkpoint_sha256"] == "4cb40a7140b1055b7e077232a202f5600ce4a33eb955e8ff8ff1825b5fd54463"
print("Starter full course-dev teacher-forced reference verified.", flush=True)
PY

run_eval() {
  local run_label="$1"
  local run_checkpoint="$2"
  shift 2
  caffeinate -i .venv/bin/python -u scripts/evaluate.py \
    --checkpoint "$run_checkpoint" --texts data/eval/dev.jsonl \
    --output "results/$run_label" --prompt-metadata configs/reference_prompt.json \
    --seeds 17 42 "$@" > "logs/$run_label.log" 2>&1
  .venv/bin/python - "$run_label" <<'PY'
import json, sys
from pathlib import Path
name = sys.argv[1]
summary = json.loads((Path("results") / name / "summary.json").read_text())["overall"]
assert summary["items"] == 90, summary
print(json.dumps({"run": name, "summary": summary}), flush=True)
# Exceptions are retained for analysis, never silently converted to successes.
PY
}

run_eval starter-no-lm-dev models/hw4-tts-starter.pt --no-lm-prompt
run_eval starter-split-dev models/hw4-tts-starter.pt --split-sentences
run_eval B-standard-dev checkpoints/B-ordinary-seed183/latest.pt
run_eval B-no-lm-dev checkpoints/B-ordinary-seed183/latest.pt --no-lm-prompt
run_eval B-split-dev checkpoints/B-ordinary-seed183/latest.pt --split-sentences

# Fresh output names are intentional: offline analyses refuse silent overwrite.
.venv/bin/python scripts/analyze_results.py \
  --system A=results/starter-standard-dev \
  --system ANoLM=results/starter-no-lm-dev \
  --system ASplit=results/starter-split-dev \
  --system B=results/B-standard-dev \
  --system BNoLM=results/B-no-lm-dev \
  --system BSplit=results/B-split-dev \
  --baseline A --output results/analysis/development-comparisons \
  --bootstrap 2000 --seed 1832026 --plots

# Separate one-factor contrasts: the six-system overview also contains combined
# changes, which must not be described as isolated training or inference effects.
.venv/bin/python scripts/analyze_results.py \
  --system B=results/B-standard-dev \
  --system BNoLM=results/B-no-lm-dev --system BSplit=results/B-split-dev \
  --baseline B --output results/analysis/development-within-B \
  --bootstrap 2000 --seed 1832026
.venv/bin/python scripts/analyze_results.py \
  --system ANoLM=results/starter-no-lm-dev --system BNoLM=results/B-no-lm-dev \
  --baseline ANoLM --output results/analysis/development-training-no-lm \
  --bootstrap 2000 --seed 1832026
.venv/bin/python scripts/analyze_results.py \
  --system ASplit=results/starter-split-dev --system BSplit=results/B-split-dev \
  --baseline ASplit --output results/analysis/development-training-split \
  --bootstrap 2000 --seed 1832026

.venv/bin/python scripts/analyze_strategy_costs.py \
  --system A=results/starter-standard-dev \
  --system ANoLM=results/starter-no-lm-dev \
  --system ASplit=results/starter-split-dev \
  --system B=results/B-standard-dev \
  --system BNoLM=results/B-no-lm-dev \
  --system BSplit=results/B-split-dev \
  --baseline A --expected-items 90 \
  --output results/analysis/development-strategy-costs
.venv/bin/python scripts/analyze_strategy_costs.py \
  --system B=results/B-standard-dev \
  --system BNoLM=results/B-no-lm-dev \
  --system BSplit=results/B-split-dev \
  --baseline B --expected-items 90 \
  --output results/analysis/development-within-B-costs

for system_label in A ANoLM ASplit B BNoLM BSplit; do
  .venv/bin/python scripts/analyze_failure_profiles.py \
    --input "results/analysis/development-comparisons/$system_label.rescored.jsonl" \
    --output "results/analysis/development-failure-profiles/$system_label" \
    --expected-items 90 --bootstrap 10000 --seed 20261020
done

.venv/bin/python scripts/prepare_listening_bundle.py \
  --system A=results/analysis/development-comparisons \
  --system ANoLM=results/analysis/development-comparisons \
  --system ASplit=results/analysis/development-comparisons \
  --system B=results/analysis/development-comparisons \
  --system BNoLM=results/analysis/development-comparisons \
  --system BSplit=results/analysis/development-comparisons \
  --output results/listening/development-comparisons --per-category 1

.venv/bin/python scripts/prepare_listening_bundle.py \
  --system B=results/analysis/development-within-B \
  --system BNoLM=results/analysis/development-within-B \
  --system BSplit=results/analysis/development-within-B \
  --output results/listening/development-within-B --per-category 1
