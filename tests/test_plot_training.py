"""Focused checks for token-window math and immutable rollback-aware snapshots."""
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLOTS = runpy.run_path(str(ROOT / "scripts/plot_training.py"))


def test_trailing_token_window_prorates_boundary_batch():
    mean, span = PLOTS["trailing_token_mean"]([2.0, 6.0, 10.0], [2, 3, 2], 4)
    # End 5: one token from the first batch + three from the second.
    # End 7: two tokens from the second batch + two from the third.
    np.testing.assert_allclose(mean, [2.0, 5.0, 8.0])
    np.testing.assert_array_equal(span, [2, 4, 4])


def test_snapshot_rollback_validation_and_partial_status(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = tmp_path / "plots"
    output.mkdir()
    (run_dir / "config.json").write_text(json.dumps(dict(effective_token_budget=100)))
    rows = [dict(step=step, target_tokens=10, total_target_tokens=step * 10,
                 loss=float(step), accuracy=0.1, learning_rate=1e-4,
                 epoch=0, epoch_batch_cursor=step, padding_fraction=0.2)
            for step in (1, 2, 3, 2)]
    original = "".join(json.dumps(row) + "\n" for row in rows)
    (run_dir / "train.jsonl").write_text(original)
    (run_dir / "validation.jsonl").write_text("".join(
        json.dumps(dict(step=step, loss=2.5, accuracy=0.2)) + "\n" for step in (2, 3)))
    result = PLOTS["load_run"]("fixture", run_dir, output, 15, False, 0)
    assert result["manifest"]["status"] == "PARTIAL_SNAPSHOT"
    assert result["manifest"]["observed_steps"] == 2
    assert result["manifest"]["observed_target_tokens"] == 20
    assert result["manifest"]["integrity"]["discarded_training_update_rows"] == 2
    assert result["manifest"]["integrity"]["ignored_validation_steps_outside_effective_train_branch"] == [3]
    assert len(result["validation"]) == 1
    snap = Path(result["manifest"]["input_snapshots"]["train.jsonl"]["snapshot_path"])
    assert snap.read_text() == original
    (run_dir / "train.jsonl").write_text(original + json.dumps(rows[-1]) + "\n")
    assert snap.read_text() == original


def test_missing_steps_fail_integrity_check(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = tmp_path / "plots"
    output.mkdir()
    (run_dir / "config.json").write_text(json.dumps(dict(effective_token_budget=100)))
    (run_dir / "train.jsonl").write_text('{"step":1}\n{"step":3}\n')
    with pytest.raises(ValueError, match="missing steps"):
        PLOTS["load_run"]("fixture", run_dir, output, 15, False, 0)
