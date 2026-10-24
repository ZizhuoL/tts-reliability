"""Small offline checks: actual segmentation, missingness and fair cost accounting."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_strategy_costs.py"
spec = importlib.util.spec_from_file_location("strategy_costs", SCRIPT)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)


def record(seed=17, segments=2, **changes):
    value = dict(id="same", seed=seed, text="Red green blue.", asr_transcript="Red green blue.",
        status="complete", stop_reason="eos", length_group="short", split_sentences=True,
        wav="does-not-exist.wav", audio_seconds=4.15, generation_seconds=2.0,
        generated_tokens=100, sample_rate=24000,
        segments=[dict(generated_tokens=50, requested_output_budget=500,
                       effective_output_budget=500, stop_reason="eos") for _ in range(segments)])
    value.update(changes)
    return value


def test_pause_rounding_limits_and_rtf_are_distinguished():
    row = strategy.describe(record(), {"pause_seconds": .15002})
    expected = round(.15002*24000)/24000
    assert row["split_state"] == "actually_split"
    assert row["inserted_pause_seconds"] == expected
    assert row["inserted_pause_basis"].startswith("accounted_from_completed_segments_and_config")
    assert row["completed_requested_budget_sum"] == 1000
    assert row["completed_generated_tokens"] == 100
    assert row["generation_rtf_excluding_configured_pauses"] > row["generation_rtf"]


def test_missing_config_or_sample_rate_keeps_pause_unknown():
    assert strategy.describe(record(), {})["inserted_pause_seconds"] is None
    assert strategy.describe(record(sample_rate=None), {"pause_seconds": .15})["inserted_pause_seconds"] is None


def test_partial_synthesis_is_unknown_not_unsplit_or_full_budget():
    row = strategy.describe(record(status="error", error_stage="synthesis", segments=1), {"pause_seconds": .15})
    assert row["split_state"] == "unknown"
    assert row["completed_waveform_segments"] is None
    assert row["completed_requested_budget_sum"] is None
    assert row["partial_or_complete_recorded_requested_budget_sum"] == 500
    assert row["inserted_pause_seconds"] is None
    assert row["metrics"] is None


def test_asr_error_keeps_actual_split_but_does_not_become_zero_wer():
    base = strategy.describe(record(segments=1, split_sentences=False), {})
    failed = strategy.describe(record(status="error", error_stage="asr", asr_transcript=None), {"pause_seconds": .15})
    comparison = strategy.compare([base], [failed])
    subset = comparison["method_actually_split"]["overall"]
    assert subset["shared_attempts"] == 1
    assert subset["common_scored_pairs"] == 0
    assert subset["missing_score_pairs"] == 1
    assert subset["delta_wer"] is None
    assert failed["inserted_pause_seconds"] == .15


def test_paired_subsets_seeds_and_empty_asr():
    base = [strategy.describe(record(seed=s, segments=1, split_sentences=False), {}) for s in [17,42]]
    method = [strategy.describe(record(seed=17, asr_transcript="Red blue."), {"pause_seconds": .15}),
              strategy.describe(record(seed=42, segments=1), {"pause_seconds": .15})]
    result = strategy.compare(base, method)
    assert result["method_actually_split"]["overall"]["shared_attempts"] == 1
    assert result["method_not_split"]["overall"]["shared_attempts"] == 1
    assert result["all_shared"]["by_seed"]["17"]["delta_wer"] == pytest.approx(1/3)
    assert result["all_shared"]["by_seed"]["42"]["delta_wer"] == 0
    assert strategy.describe(record(asr_transcript=""), {})["metrics"] is None


def test_latest_events_torn_tail_and_absent_input(tmp_path):
    path = tmp_path/"raw.jsonl"
    path.write_text(json.dumps(record(status="synthesized"))+"\n"+json.dumps(record())+"\n"+'{"torn":')
    rows, manifest, provenance = strategy.read_input("A", tmp_path)
    assert len(rows)==1 and rows[0]["status"]=="complete"
    assert provenance["ignored_torn_final_line"] is True
    rows, _, provenance = strategy.read_input("B", tmp_path/"not-yet-started")
    assert rows==[] and provenance["input_status"]=="unavailable"
    assert strategy.analyze_system(rows,90)["status"]=="incomplete_or_unavailable"
