"""CPU-only checks of missing-score denominators and clustered inference."""
import importlib.util
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "analyze_results", Path(__file__).resolve().parents[1] / "scripts/analyze_results.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def event(identifier, text, hypothesis=None, *, seed=17, status="complete", length="short"):
    row = dict(key=f"{identifier}:{seed}", id=identifier, seed=seed, run_id="test-run",
               text=text, status=status, length_group=length, stop_reason="eos",
               metrics={"legacy": "preserved"}, wav="wav/example.wav")
    if hypothesis is not None:
        row["asr_transcript"] = hypothesis
    if status == "error":
        row["error_stage"] = "asr"
    return row


def saved_run(tmp_path, name, rows, tail=b""):
    path = tmp_path / name
    path.mkdir()
    raw = b"".join(json.dumps(row).encode()+b"\n" for row in rows)+tail
    (path / "raw.jsonl").write_bytes(raw)
    (path / "manifest.json").write_text(json.dumps(
        dict(run_id="test-run", normalizer="legacy-normalizer")))
    return path


def test_last_event_unicode_rescore_and_nonmutating_torn_tail(tmp_path):
    old = event("one", "I’ll return.", status="synthesized")
    final = event("one", "I’ll return.", "I will return.")
    path = saved_run(tmp_path, "run", [old, final], b'{"key":')
    before = (path / "raw.jsonl").read_bytes()
    rows, provenance = analysis.load_run("run", path)
    assert len(rows) == 1
    assert rows[0]["metrics"]["errors"] == 0
    assert rows[0]["original_metrics"] == {"legacy": "preserved"}
    assert rows[0]["original_normalizer"] == "legacy-normalizer"
    assert provenance["ignored_torn_final_line"]
    assert (path / "raw.jsonl").read_bytes() == before


def test_failure_sensitivity_and_common_success_denominators(tmp_path):
    left = saved_run(tmp_path, "left", [
        event("one", "red blue", "red"),
        event("two", "green yellow orange", status="error"),
    ])
    right = saved_run(tmp_path, "right", [
        event("one", "red blue", "red blue"),
        event("two", "green yellow orange", "green yellow orange"),
    ])
    a, _ = analysis.load_run("baseline", left)
    b, _ = analysis.load_run("method", right)
    summary = analysis.aggregate(a)
    assert summary["scored"]["reference_words"] == 2
    assert summary["scored"]["wer"] == .5
    assert summary["unscored_as_full_deletions"]["reference_words"] == 5
    assert summary["unscored_as_full_deletions"]["deletions"] == 4
    assert summary["unscored_as_full_deletions"]["wer"] == .8
    assert summary["exception_stages"] == {"asr": 1}
    pair, examples = analysis.compare_runs(a, b, 100, 183)
    assert pair["overall"]["common_success_pairs"] == 1
    assert pair["overall"]["common_success"]["delta_wer"] == -.5
    assert pair["overall"]["unscored_as_full_deletions"]["delta_wer"] == -.8
    assert pair["overall"]["common_success"]["delta_wer_ci95"] is None
    assert not examples[1]["common_success"]


def test_cluster_bootstrap_keeps_duplicate_seeds_together(tmp_path):
    # Duplicating each identical seed observation must not narrow the CI.
    raw_a = [event("one", "red blue", "red"),
             event("two", "green yellow", "green yellow"),
             event("three", "orange purple", "orange", length="long")]
    raw_b = [event("one", "red blue", "red blue"),
             event("two", "green yellow", "green"),
             event("three", "orange purple", "orange purple", length="long")]
    a, _ = analysis.load_run("a", saved_run(tmp_path, "a", raw_a))
    b, _ = analysis.load_run("b", saved_run(tmp_path, "b", raw_b))
    pairs = list(zip(a, b))
    single = analysis.paired_bootstrap(pairs, "metrics", 1000, 19)
    double = analysis.paired_bootstrap(pairs+pairs, "metrics", 1000, 19)
    assert single["clusters_per_length"] == {"short": 2, "long": 1}
    assert single["text_clusters"] == double["text_clusters"] == 3
    assert double["paired_items"] == 6
    assert single["delta_wer_ci95"] == double["delta_wer_ci95"]
    assert single["delta_wer"] == double["delta_wer"]


def test_pairing_rejects_changed_reference_and_counts_unmatched(tmp_path):
    a, _ = analysis.load_run("a", saved_run(tmp_path, "a", [
        event("one", "red blue", "red blue"),
        event("extra", "green yellow", "green yellow"),
    ]))
    b, _ = analysis.load_run("b", saved_run(tmp_path, "b", [
        event("one", "red blue", "red blue"),
    ]))
    result, _ = analysis.compare_runs(a, b, 10, 1)
    assert result["baseline_only_items"] == 1
    b[0]["text"] = "different reference"
    with pytest.raises(ValueError, match="different reference"):
        analysis.compare_runs(a, b, 10, 1)


def test_cli_writes_inspectable_results_and_refuses_overwrite(tmp_path):
    path = saved_run(tmp_path, "input", [event("one", "red blue", "red blue")])
    output = tmp_path / "analysis"
    args = ["--system", f"starter={path}", "--output", str(output), "--bootstrap", "10"]
    report = analysis.main(args)
    assert report["systems"]["starter"]["overall"]["scored"]["wer"] == 0
    assert (output / "starter.rescored.jsonl").exists()
    assert (output / "metrics.csv").exists()
    with pytest.raises(SystemExit):
        analysis.main(args)


def test_three_system_paired_cli_finishes_all_exports(tmp_path):
    """Exercise the actual CLI, including both paired CSV and optional plots."""
    pytest.importorskip("matplotlib")
    raw_a, raw_b, raw_c = [], [], []
    for index, text in enumerate(("red blue", "green yellow", "orange purple", "silver gold")):
        for seed in (17, 42):
            common = dict(seed=seed, length="short" if index < 2 else "long")
            raw_a.append(event(str(index), text, text.split()[0], **common))
            raw_b.append(event(str(index), text, text, **common))
            raw_c.append(event(str(index), text, text if index % 2 else "", **common))
    paths = [saved_run(tmp_path, name, rows) for name, rows in
             (("original", raw_a), ("improved", raw_b), ("mixed", raw_c))]
    output = tmp_path / "paired-analysis"
    command = [sys.executable, str(Path(analysis.__file__)), "--baseline", "Original",
               "--output", str(output), "--bootstrap", "100", "--plots"]
    for name, path in zip(("Original", "Improved", "Mixed"), paths):
        command += ["--system", f"{name}={path}"]
    process = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    expected = {
        "summary.json", "analysis_manifest.json", "metrics.csv", "paired.csv",
        "Original.rescored.jsonl", "Improved.rescored.jsonl", "Mixed.rescored.jsonl",
        "Original_vs_Improved.examples.jsonl", "Original_vs_Mixed.examples.jsonl",
        "wer_by_length.png", "wer_by_length.pdf", "completion.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with (output / "paired.csv").open() as stream:
        paired = list(csv.DictReader(stream))
    assert len(paired) == 12  # two methods × overall/two lengths × two scoring rules
    assert {row["method"] for row in paired} == {"Improved", "Mixed"}
    assert all(row["method_system"] == row["method"] for row in paired)
    assert all(row["bootstrap_method"].startswith("Percentile bootstrap") for row in paired)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["paired"]["Improved"]["overall"]["common_success"]["delta_wer"] == -.5
    assert summary["paired"]["Mixed"]["overall"]["common_success"]["delta_wer"] == 0
    assert len((output / "Original_vs_Mixed.examples.jsonl").read_text().splitlines()) == 8
    completion = json.loads((output / "completion.json").read_text())
    assert completion["status"] == "complete"
    assert expected - {"completion.json"} <= set(completion["artifacts"])
    for filename, checksum in completion["artifacts"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == checksum
