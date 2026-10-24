#!/usr/bin/env python3
"""CPU/stdlib regression checks; fixtures are synthetic, real final is never opened."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("final_runner", HERE / "run_final_evaluation.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
REAL_ROOT = runner.ROOT
TEMPLATE = runner.read(runner.TEMPLATE)


class FinalRunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.patches = [patch.object(runner, "ROOT", self.root),
                        patch.object(runner, "FINAL", self.root / "data/eval/final.sealed.jsonl"),
                        patch.object(runner, "TEMPLATE", self.root / "configs/final_evaluation.template.json")]
        for item in self.patches:
            item.start()
        self.write("configs/final_evaluation.template.json", TEMPLATE)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        elif isinstance(data, str):
            path.write_text(data)
        else:
            path.write_text(json.dumps(data))
        return path

    def report(self, winner="B"):
        systems = {}
        for name in runner.DEV_SYSTEMS:
            wer = 0.1 if name == winner else 0.2
            scope = dict(items=90, scored_items=90, scored={"wer": wer},
                         truncation_rate=0, exception_rate=0, generation_seconds=100)
            systems[name] = dict(overall=scope, by_length={g: copy.deepcopy(scope) for g in ("short", "medium", "long")})
        return {"systems": systems}

    def candidate(self, winner="B"):
        candidate = copy.deepcopy(TEMPLATE)
        candidate["selection"].update(
            selected_system=winner, rationale="Synthetic fixture checks conservative development selection only.",
            missingness_review="Synthetic fixture has all scored observations and no missingness.")
        return candidate

    def fixture(self, winner="B"):
        for name in runner.CODE:
            self.write(name, "# synthetic implementation\n")
        self.write("configs/development_selection_rule.json", {})
        self.write("configs/protocol.json", {})
        self.write("configs/reference_prompt.json", {"text": "synthetic prompt"})
        self.write("data/reference/prompt.wav", b"synthetic waveform")
        dev = [{"id": str(i), "text": "Synthetic development example.", "length_group": "short"} for i in range(45)]
        dev_path = self.write("data/eval/dev.jsonl", "".join(json.dumps(x) + "\n" for x in dev))
        self.write("data/eval/manifest.json", dict(
            counts={"final": {"short": 20, "medium": 20, "long": 20}},
            dev_sha256=runner.sha(dev_path), final_sha256="0" * 64))
        # Existence never authorizes reading; guarded below throughout freeze.
        self.write("data/eval/final.sealed.jsonl", b"must never be opened during preparation")
        checkpoint_hashes = {}
        for name, path in runner.CHECKPOINTS.items():
            checkpoint_hashes[name] = runner.sha(self.write(path, name.encode()))
            validation_name = {"A": "starter-full-validation", "B": "B-full-validation",
                               "B184": "B184-full-validation"}[name]
            self.write(f"results/{validation_name}/validation.json", dict(examples=9756, target_tokens=1362834))
            self.write(f"results/{validation_name}/config.json", dict(
                val_max_batches=0, checkpoint_sha256=checkpoint_hashes[name],
                validation=dict(path="same-data", bytes=1, mtime_ns=1, batches=[[1, 2]], subset_seed=184 if name != "B184" else 185)))
            if name != "A":
                folder = Path(path).parent
                self.write(str(folder / "summary.json"), dict(
                    stop_reason="token_budget", completed_epochs=1, unique_examples=354780,
                    valid_examples=354780, target_tokens=50450231, steps=44348))
                self.write(str(folder / "config.json"), dict(
                    seed=183 if name == "B" else 184, long_weight=1.0,
                    initial_checkpoint_sha256=checkpoint_hashes["A"]))
        overview_inputs = {}
        for name, (folder, model, strategy) in dict(runner.DEV_SYSTEMS, **runner.REPLICATION_SYSTEMS).items():
            generation = dict(TEMPLATE["generation"])
            generation["no_cache"] = not generation.pop("use_cache")
            generation.update(no_lm_prompt=strategy == "no-lm", split_sentences=strategy == "split")
            manifest = dict(run_id=name, checkpoint_sha256=checkpoint_hashes[model],
                            texts_sha256=runner.sha(dev_path), settings=generation,
                            prompt_sha256=runner.sha(self.root / "data/reference/prompt.wav"),
                            prompt_metadata_sha256=runner.sha(self.root / "configs/reference_prompt.json"),
                            implementation={}, cosy_assets={}, whisper_assets={}, packages={})
            self.write(folder + "/manifest.json", manifest)
            wav = self.write(folder + "/wav/one.wav", b"synthetic waveform")
            rows = [dict(key=f"{i}-{seed}", id=str(i), seed=seed, run_id=name, status="complete",
                         wav="wav/one.wav", wav_sha256=runner.sha(wav))
                    for i in range(45) for seed in (17, 42)]
            raw = self.write(folder + "/raw.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
            self.write(folder + "/summary.json", {})
            if name in runner.DEV_SYSTEMS:
                overview_inputs[name] = dict(directory=str(self.root / folder), raw_sha256=runner.sha(raw),
                                             deduplicated_items=90, ignored_torn_final_line=False)
        for name in ("development-comparisons", "development-within-B", "development-training-no-lm",
                     "development-training-split", "replication-standard", "replication-no-lm", "replication-split"):
            folder = "results/analysis/" + name
            summary = self.write(folder + "/summary.json", self.report(winner))
            artifacts = {"summary.json": runner.sha(summary)}
            if name == "development-comparisons":
                manifest = self.write(folder + "/analysis_manifest.json", dict(inputs=overview_inputs))
                artifacts["analysis_manifest.json"] = runner.sha(manifest)
            self.write(folder + "/completion.json", dict(status="complete", artifacts=artifacts))
        self.write("results/B-seed184-execution/completion.json", {"status": "complete"})
        return self.write("configs/candidate.json", self.candidate(winner))

    def test_template_cannot_freeze_unselected(self):
        runner.validate_candidate(TEMPLATE, allow_unselected=True)
        with self.assertRaisesRegex(ValueError, "Select one"):
            runner.validate_candidate(TEMPLATE)

    def test_final_read_and_hash_are_denied(self):
        self.write("data/eval/final.sealed.jsonl", b"protected")
        for func in (runner.read, runner.sha):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                func(runner.FINAL)

    def test_complete_freeze_does_not_open_final_and_detects_tampering(self):
        candidate = self.fixture()
        original_open = Path.open
        opened = []

        def guarded(path, *args, **kwargs):
            if path.resolve() == runner.FINAL.resolve():
                raise AssertionError("pre-freeze final access")
            opened.append(str(path))
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            runner.freeze(candidate, self.root / "frozen")
            protocol, _ = runner.load_frozen(self.root / "frozen", bindings=True)
        self.assertEqual(protocol["primary_comparison"], ["A", "B"])
        self.assertEqual(len(protocol["cells"]), 9)
        with self.assertRaisesRegex(ValueError, "must be new"):
            runner.freeze(candidate, self.root / "frozen")
        self.write("scripts/evaluate.py", "changed")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            runner.load_frozen(self.root / "frozen", bindings=True)
        protocol["primary_comparison"] = ["A", "B184"]
        self.write("frozen/protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            runner.load_frozen(self.root / "frozen")

    def test_fallback_does_not_invent_self_comparison(self):
        candidate = self.fixture(winner="A")
        runner.freeze(candidate, self.root / "frozen")
        protocol, _ = runner.load_frozen(self.root / "frozen")
        self.assertIsNone(protocol["primary_comparison"])
        self.assertEqual(protocol["always_report_training_comparison"], ["A", "B"])
        self.assertEqual(protocol["training_comparison_role"], "prespecified_research_comparison_exploratory")

    def test_all_planned_cli_options_exist_and_cells_have_fixed_seeds(self):
        candidate = self.fixture()
        runner.freeze(candidate, self.root / "frozen")
        protocol, _ = runner.load_frozen(self.root / "frozen")
        stages = runner.commands(protocol)
        evaluations = [s for s in stages if s["kind"] == "evaluation"]
        self.assertEqual(len(evaluations), 9)
        self.assertEqual([s["system"] for s in evaluations], list(runner.DEV_SYSTEMS) + list(runner.REPLICATION_SYSTEMS))
        for stage in stages:
            argv = stage["argv"]
            script = next(Path(x) for x in argv if x.endswith(".py"))
            source = (REAL_ROOT / "scripts" / script.name).read_text()
            flags = {arg.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call) and
                     isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                     for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
            self.assertTrue(set(x for x in argv if x.startswith("--")).issubset(flags), stage["name"])
            self.assertNotIn("--retry-errors", argv)
            if stage["kind"] == "evaluation":
                at = argv.index("--seeds")
                self.assertEqual(argv[at + 1:at + 3], ["17", "42"])
                self.assertNotIn("--limit", argv)
        self.assertTrue(any(s["name"] == "analysis-overview" and "--plots" in s["argv"] for s in stages))
        self.assertTrue(any(s["name"] == "listening" and "--pair" in s["argv"] for s in stages))

    def test_incomplete_attempts_block_freeze(self):
        candidate = self.fixture()
        self.write("results/B-split-dev/raw.jsonl", "")
        with self.assertRaisesRegex(ValueError, "Attempt identity mismatch"):
            runner.freeze(candidate, self.root / "frozen")
        self.assertFalse((self.root / "frozen").exists())

    def test_wrong_development_choice_and_unreviewed_missingness_fail(self):
        candidate = self.candidate(winner="ANoLM")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            runner.selection_check(candidate, self.report())
        report = self.report()
        report["systems"]["B"]["overall"]["scored_items"] = 89
        with self.assertRaisesRegex(ValueError, "Missing ASR"):
            runner.selection_check(self.candidate(), report)
        candidate = self.candidate(winner="A")
        candidate["selection"]["failure_robust_eligible_systems"] = []
        self.assertEqual(runner.selection_check(candidate, report)["selected_system"], "A")

    def test_run_requires_explicit_idle_and_known_processes_block_before_read(self):
        with self.assertRaisesRegex(ValueError, "confirm-gpu-idle"):
            runner.run(self.root / "nonexistent", resume=False, confirm_gpu_idle=False)
        with patch.object(runner, "active_gpu_candidates", return_value=["active train.py"]):
            with self.assertRaisesRegex(ValueError, "still active"):
                runner.run(self.root / "nonexistent", resume=False, confirm_gpu_idle=True)

    def interrupted_finalization_fixture(self):
        """All outputs are synthetic temp files; every subprocess is forbidden."""
        candidate = self.fixture()
        final_bytes = "".join(json.dumps(dict(
            id=f"synthetic-{i}", text="Synthetic finalization fixture.",
            length_group=("short", "medium", "long")[i // 20])) + "\n"
            for i in range(60)).encode()
        metadata_path = self.root / "data/eval/manifest.json"
        metadata = runner.read(metadata_path)
        metadata["final_sha256"] = hashlib.sha256(final_bytes).hexdigest()
        self.write("data/eval/manifest.json", metadata)
        self.write("data/eval/final.sealed.jsonl", final_bytes)
        frozen = self.root / "frozen"
        runner.freeze(candidate, frozen)
        protocol, digest = runner.load_frozen(frozen)
        out = runner.local(protocol["candidate"]["output"])
        runner.write_new(out / "run-binding.json", dict(protocol_sha256=digest))
        runner.write_new(out / "protocol.json", protocol)
        for stage in runner.commands(protocol):
            folder = Path(stage["output"])
            runner.write_new(folder / "synthetic-artifact.json", {"fixture": stage["name"]})
            runner.write_new(out / "stage-markers" / (stage["name"] + ".json"),
                             dict(stage=stage, artifacts=runner.inventory(folder)))
        original_write = runner.write_new

        def interrupt_completion(path, value):
            if path == out / "completion.json":
                raise RuntimeError("simulated interruption after roles persisted")
            return original_write(path, value)

        with patch.object(runner, "active_gpu_candidates", return_value=[]), \
                patch.object(runner.subprocess, "run", side_effect=AssertionError("No subprocess permitted")), \
                patch.object(runner, "write_new", side_effect=interrupt_completion):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                runner.run(frozen, resume=True, confirm_gpu_idle=True)
        self.assertTrue((out / "comparison-roles.json").exists())
        self.assertFalse((out / "completion.json").exists())
        return frozen, out

    def test_resume_after_roles_written_before_completion_preserves_roles(self):
        frozen, out = self.interrupted_finalization_fixture()
        roles_path = out / "comparison-roles.json"
        original_bytes, original_mtime = roles_path.read_bytes(), roles_path.stat().st_mtime_ns
        with patch.object(runner, "active_gpu_candidates", return_value=[]), \
                patch.object(runner.subprocess, "run", side_effect=AssertionError("Completed stages must not rerun")):
            runner.run(frozen, resume=True, confirm_gpu_idle=True)
        self.assertEqual(roles_path.read_bytes(), original_bytes)
        self.assertEqual(roles_path.stat().st_mtime_ns, original_mtime)
        self.assertEqual(runner.read(out / "completion.json")["status"], "complete")
        events = [json.loads(line) for path in (out / "logs").glob("run-*.jsonl")
                  for line in path.read_text().splitlines()]
        self.assertTrue(any(row["event"] == "verified_existing_comparison_roles" for row in events))

    def test_resume_refuses_tampered_roles_without_overwriting(self):
        frozen, out = self.interrupted_finalization_fixture()
        roles_path = out / "comparison-roles.json"
        tampered = runner.read(roles_path)
        tampered["primary_comparison"] = ["A", "B184"]
        roles_path.write_text(json.dumps(tampered))
        original_bytes = roles_path.read_bytes()
        with patch.object(runner, "active_gpu_candidates", return_value=[]), \
                patch.object(runner.subprocess, "run", side_effect=AssertionError("Completed stages must not rerun")):
            with self.assertRaisesRegex(ValueError, "comparison roles differ"):
                runner.run(frozen, resume=True, confirm_gpu_idle=True)
        self.assertEqual(roles_path.read_bytes(), original_bytes)
        self.assertFalse((out / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()
