"""Check analysis against actual tiny CPU updates and rollback/corruption cases."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

from tts_reliability.model import TextToSpeechLM
from tts_reliability.training import run

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = runpy.run_path(str(ROOT / "scripts/analyze_training_exposure.py"))


def test_actual_update_reconstruction_checkpoint_and_corrupt_log(tmp_path, monkeypatch):
    model = TextToSpeechLM(17, 11, d_model=16, n_heads=4, n_layers=1, max_seq_len=64)
    torch.save(dict(model_state_dict=model.state_dict(), model_config=model.model_config,
                    text_vocab_size=17, speech_vocab_size=11), tmp_path / "base.pt")
    rows = [dict(utt=str(index), text="fixture", text_token=torch.tensor([1, 2]),
                 speech_token=torch.tensor([3, 4, 5][:1 + index % 3]),
                 speaker_idx=0, speaker_id="test") for index in range(8)]
    torch.save(dict(samples=rows), tmp_path / "data.pt")
    np.save(tmp_path / "lengths.npy", np.array([[2, len(row["speech_token"])] for row in rows]))
    parser = runpy.run_path(str(ROOT / "scripts/train.py"))["parser"]
    run(parser().parse_args([
        "--mode", "train", "--device", "cpu", "--checkpoint", str(tmp_path / "base.pt"),
        "--data", str(tmp_path / "data.pt"), "--output", str(tmp_path / "run"),
        "--steps", "4", "--batch-size", "2"]))
    common = [
        "analyze_training_exposure.py", "--config", str(tmp_path / "run/config.json"),
        "--lengths", str(tmp_path / "lengths.npy"),
        "--checkpoint", str(tmp_path / "run/latest.pt")]
    monkeypatch.setattr(sys, "argv", common + ["--output", str(tmp_path / "actual.json")])
    assert ANALYSIS["analyze"](ANALYSIS["arguments"]()) == 0
    result = json.loads((tmp_path / "actual.json").read_text())
    assert result["totals"]["examples"] == 8
    assert result["totals"]["unique_coverage"] == 1
    assert result["verification"]["checkpoint"]["matched"]
    assert result["verification"]["all_logged_fields_match"]
    logs = [json.loads(line) for line in (tmp_path / "run/train.jsonl").read_text().splitlines()]
    logs[0]["total_target_tokens"] += 1
    (tmp_path / "corrupted.jsonl").write_text("".join(json.dumps(row) + "\n" for row in logs))
    monkeypatch.setattr(sys, "argv", common + [
        "--log", str(tmp_path / "corrupted.jsonl"), "--output", str(tmp_path / "bad.json")])
    assert ANALYSIS["analyze"](ANALYSIS["arguments"]()) == 2
    result = json.loads((tmp_path / "bad.json").read_text())
    assert result["verification"]["mismatch_count"] == 1
    assert result["verification"]["mismatches"][0]["field"] == "total_target_tokens"


def test_rollback_discards_unrerun_old_tail_and_partial_line(tmp_path):
    path = tmp_path / "rollback.jsonl"
    rows = [dict(step=step, target_tokens=7, examples=2) for step in (1, 2, 3, 2)]
    path.write_bytes(("".join(json.dumps(row) + "\n" for row in rows) + '{"step":').encode())
    logs, sources, tails, replay = ANALYSIS["read_logs"]([path])
    assert list(logs) == [1, 2]
    assert len(replay["all_rows"]) == 4
    assert len(replay["discarded_rows"]) == 2
    assert replay["rollback_events"][0]["discarded_steps"] == [2, 3]
    assert tails == [str(path)]
    assert sources[0]["snapshot_bytes"] == path.stat().st_size
