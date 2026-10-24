#!/bin/zsh
# Isolated memory probe; its parameter updates never enter the B training run.
set -euo pipefail
source scripts/env.sh
caffeinate -i .venv/bin/python -u scripts/train.py \
  --mode benchmark --checkpoint models/hw4-tts-starter.pt \
  --resume checkpoints/B-ordinary-seed183/recovery/oom-35592/step34000.pt \
  --data data/processed/train.pt --output results/B-recovery-memory075 \
  --batch-size 8 --pad-multiple 64 --sequence-implementation packed \
  --empty-cache-every 1 --mps-memory-fraction 0.75 \
  --epochs 1 --seed 183 --learning-rate 0.0002 --weight-decay 0.01 \
  --grad-clip 1 --warmup-fraction 0.1 --max-hours 12 \
  --val-data data/processed/dev.pt --val-every 2000 --val-max-batches 128 \
  --save-every 500 --save-minutes 10 --log-every 50 \
  --steps 8 --warmup-steps 1 --benchmark-repeats 1
