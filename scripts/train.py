"""Train/benchmark locally. See --help; no implicit CPU fallback."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream/eecs183-hw4"))

from tts_reliability.training import run


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("benchmark", "train", "validate"), default="benchmark")
    p.add_argument("--checkpoint", default=str(ROOT / "models/hw4-tts-starter.pt"))
    p.add_argument("--data", default=str(ROOT / "data/processed/train.pt"))
    p.add_argument("--tokenizer-dir", default=str(ROOT / "models/cosy"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("mps", "cpu"), default="mps")
    p.add_argument("--mps-memory-fraction", type=float, default=0.9,
                   help="fraction of Metal recommended working set; bounds process allocations")
    p.add_argument("--steps", type=int, default=0, help="0 = token/epoch budget (benchmark: 8)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--pad-multiple", type=int, default=64,
                   help="round training sequences to reduce MPS shape compilation")
    p.add_argument("--sequence-implementation", choices=("original", "packed"), default="original",
                   help="optional training-only fixed-grid embedding preparation")
    p.add_argument("--empty-cache-every", type=int, default=0,
                   help="clear unused MPS cache every N training updates; 0 disables")
    p.add_argument("--resume", help="project checkpoint; restores optimizer/RNG/sampler")
    p.add_argument("--val-data", help="prepared validation cache")
    p.add_argument("--val-every", type=int, default=0, help="steps between validation; 0 disables")
    p.add_argument("--val-max-batches", type=int, default=64,
                   help="fixed seeded validation subset; 0 validates the entire cache")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--token-budget", type=int, default=0,
                   help="supervised speech targets including EOS; final batch may overshoot")
    p.add_argument("--max-hours", type=float, default=0)
    p.add_argument("--seed", type=int, default=183)
    p.add_argument("--long-weight", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-fraction", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--benchmark-repeats", type=int, default=3,
                   help="steady repeated updates per representative real sampler batch")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--save-minutes", type=float, default=20)
    p.add_argument("--log-every", type=int, default=10)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.epochs < 1 or arguments.steps < 0 or arguments.token_budget < 0:
        raise SystemExit("epochs must be positive; steps/token-budget nonnegative")
    if arguments.save_every < 1 or arguments.log_every < 1:
        raise SystemExit("save-every/log-every must be positive")
    if not 0 <= arguments.warmup_fraction < 1:
        raise SystemExit("warmup-fraction must lie in [0,1)")
    if not 0 < arguments.mps_memory_fraction <= 1:
        raise SystemExit("mps-memory-fraction must lie in (0,1]")
    if arguments.val_every < 0 or arguments.val_max_batches < 0:
        raise SystemExit("val-every and val-max-batches must be nonnegative")
    if arguments.pad_multiple < 1 or arguments.benchmark_repeats < 1 or arguments.warmup_steps < 0:
        raise SystemExit("pad-multiple/repeats must be positive; warmup-steps nonnegative")
    if arguments.empty_cache_every < 0:
        raise SystemExit("empty-cache-every must be nonnegative")
    run(arguments)
