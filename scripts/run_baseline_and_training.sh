#!/bin/zsh
# Run from the project root. Serialized local compute; no scheduler/system edits.
set -euo pipefail
source scripts/env.sh

caffeinate -i .venv/bin/python -u scripts/evaluate.py \
  --checkpoint models/hw4-tts-starter.pt \
  --texts data/eval/dev.jsonl --output results/starter-standard-dev \
  --prompt-metadata configs/reference_prompt.json --seeds 17 42 \
  > logs/starter-standard-dev.log 2>&1

.venv/bin/python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path("results/starter-standard-dev/summary.json").read_text())["overall"]
assert result["items"] == 90 and result["scored_items"] == 90, result
assert result["exceptional_items"] == 0, result
print("Standardized development baseline completed; starting full-data ordinary continuation.", flush=True)
PY

caffeinate -i .venv/bin/python -u scripts/train.py \
  --mode train --checkpoint models/hw4-tts-starter.pt \
  --data data/processed/train.pt --output checkpoints/B-ordinary-seed183 \
  --batch-size 8 --pad-multiple 64 --sequence-implementation packed \
  --empty-cache-every 1 --mps-memory-fraction 0.6 \
  --epochs 1 --seed 183 --learning-rate 0.0002 --weight-decay 0.01 \
  --grad-clip 1 --warmup-fraction 0.1 --max-hours 12 \
  --val-data data/processed/dev.pt --val-every 2000 --val-max-batches 128 \
  --save-every 2000 --save-minutes 20 --log-every 50 \
  > logs/train-B-seed183.log 2>&1

caffeinate -i .venv/bin/python -u scripts/train.py \
  --mode validate --checkpoint checkpoints/B-ordinary-seed183/latest.pt \
  --val-data data/processed/dev.pt --val-max-batches 0 \
  --output results/B-full-validation --batch-size 8 --pad-multiple 64 \
  --sequence-implementation packed --mps-memory-fraction 0.6 \
  > logs/B-full-validation.log 2>&1
