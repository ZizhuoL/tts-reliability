#!/bin/zsh
# Measured runtime adjustment after two terminal OOM attempts.
# Keep training semantics fixed; record the memory ceiling and save cadence.
set -euo pipefail
source scripts/env.sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("results/B-recovery-memory075/benchmark.json").read_text())
assert report["config"]["mps_memory_fraction"] == 0.75
assert report["worst_case"]["status"] == "passed"
assert report["worst_case"]["parameters_updated"]
assert report["worst_case"]["max_sequence"] == 1216
PY

caffeinate -i .venv/bin/python -u scripts/train.py \
  --mode train --checkpoint models/hw4-tts-starter.pt \
  --resume checkpoints/B-ordinary-seed183/recovery/oom-35592/step34000.pt \
  --data data/processed/train.pt --output checkpoints/B-ordinary-seed183 \
  --batch-size 8 --pad-multiple 64 --sequence-implementation packed \
  --empty-cache-every 1 --mps-memory-fraction 0.75 \
  --epochs 1 --seed 183 --learning-rate 0.0002 --weight-decay 0.01 \
  --grad-clip 1 --warmup-fraction 0.1 --max-hours 12 \
  --val-data data/processed/dev.pt --val-every 2000 --val-max-batches 128 \
  --save-every 500 --save-minutes 10 --log-every 50 \
  > logs/train-B-seed183-resume2.log 2>&1

caffeinate -i .venv/bin/python -u scripts/train.py \
  --mode validate --checkpoint checkpoints/B-ordinary-seed183/latest.pt \
  --val-data data/processed/dev.pt --val-max-batches 0 \
  --output results/B-full-validation --batch-size 8 --pad-multiple 64 \
  --sequence-implementation packed --mps-memory-fraction 0.75 \
  > logs/B-full-validation.log 2>&1
