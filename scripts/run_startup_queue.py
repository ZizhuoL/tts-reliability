"""Serialize GPU startup checks after the current 90-item diagnostic run."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = ROOT / "results/starter-dev/summary.json"
    deadline = time.monotonic() + 3600
    while True:
        if summary.exists():
            state = json.loads(summary.read_text())
            if state["overall"]["items"] == 90:
                break
        if time.monotonic() > deadline:
            raise TimeoutError("Initial development evaluation did not finish within queue wait")
        time.sleep(15)
    stages = [
        ("generation-checks", ["scripts/check_generation.py"], 600),
        ("benchmark-b8-warm", ["scripts/train.py", "--mode", "benchmark",
          "--data", "data/processed/train.pt", "--output", "results/benchmark-b8-warm",
          "--batch-size", "8", "--pad-multiple", "64", "--steps", "8",
          "--benchmark-repeats", "3"], 1200),
        ("prompt-diagnosis", ["scripts/diagnose_prompt.py"], 3600),
    ]
    progress = []
    for name, command, timeout in stages:
        started = time.time()
        print(f"Starting {name}", flush=True)
        with (ROOT / "logs" / f"{name}.log").open("w") as log:
            completed = subprocess.run([sys.executable, "-u", *command],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        progress.append(dict(stage=name, command=command, started_unix=started,
                             elapsed_seconds=time.time() - started, returncode=completed.returncode))
        (ROOT / "results/startup_queue.json").write_text(json.dumps(progress, indent=2))
        if completed.returncode:
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}; inspect logs")
        print(f"Completed {name}", flush=True)


if __name__ == "__main__":
    main()
