"""End-to-end tiny CPU checks; no course-data or model downloads."""
import contextlib
import io
from pathlib import Path
import runpy
import tempfile

import pytest
import torch

from tts_reliability.model import TextToSpeechLM
from tts_reliability.training import (
    BucketSampler, PreparedDataset, run, validate, validation_batches)


ROOT = Path(__file__).resolve().parents[1]


def fixture_data():
    return [dict(utt=str(i), text="synthetic correctness fixture",
                 text_token=torch.tensor([1, 2]),
                 speech_token=torch.tensor([3, 4, 5][:1 + i % 3]),
                 speaker_idx=0, speaker_id="test") for i in range(8)]


@pytest.mark.parametrize("implementation", ["original", "packed"])
def test_exact_resume_and_validation_checkpoint(implementation):
    parser = runpy.run_path(str(ROOT / "scripts/train.py"))["parser"]
    with tempfile.TemporaryDirectory(prefix="tts-training-test-") as temporary:
        path = Path(temporary)
        torch.manual_seed(18)
        model = TextToSpeechLM(17, 11, d_model=16, n_heads=4, n_layers=2, max_seq_len=32)
        torch.save(dict(model_state_dict=model.state_dict(), model_config=model.model_config,
                        text_vocab_size=17, speech_vocab_size=11), path / "base.pt")
        torch.save(dict(samples=fixture_data()), path / "data.pt")
        common = ["--device", "cpu", "--checkpoint", str(path / "base.pt"),
                  "--data", str(path / "data.pt"), "--batch-size", "2",
                  "--token-budget", "100", "--val-data", str(path / "data.pt"),
                  "--val-every", "2", "--val-max-batches", "2",
                  "--sequence-implementation", implementation, "--empty-cache-every", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            bench = run(parser().parse_args(common + [
                "--mode", "benchmark", "--output", str(path / "bench"),
                "--steps", "2", "--warmup-steps", "1"]))
            run(parser().parse_args(common + [
                "--mode", "train", "--output", str(path / "full"), "--steps", "4"]))
            run(parser().parse_args(common + [
                "--mode", "train", "--output", str(path / "resume"), "--steps", "2"]))
            run(parser().parse_args(common + [
                "--mode", "train", "--output", str(path / "resume"), "--steps", "4",
                "--resume", str(path / "resume/latest.pt")]))
            evaluated = run(parser().parse_args(common + [
                "--mode", "validate", "--output", str(path / "validation"),
                "--val-max-batches", "0"]))
        assert bench["parameters_updated"]
        assert bench["worst_case"]["status"] == "passed"
        assert bench["worst_case"]["parameters_updated"]
        assert bench["worst_case"]["raw_max_sequence"] == 7
        assert bench["measured_steps"] == 2 * 3
        assert bench["worst_case"]["phase"] == "worst_case"
        assert bench["worst_case"]["sequence_implementation"] == implementation
        assert bench["worst_case"]["empty_cache_requested"]
        assert not bench["worst_case"]["empty_cache_executed"]  # CPU test
        assert evaluated["examples"] == 8
        full = torch.load(path / "full/latest.pt", weights_only=True)
        resumed = torch.load(path / "resume/latest.pt", weights_only=True)
        for key, value in full["model_state_dict"].items():
            torch.testing.assert_close(value, resumed["model_state_dict"][key], rtol=0, atol=0)
        assert full["sampler"] == resumed["sampler"]
        assert full["counters"]["seen_indices"] == resumed["counters"]["seen_indices"]
        assert torch.equal(full["rng"]["torch"], resumed["rng"]["torch"])
        assert full["config"]["initial_checkpoint_sha256"] == \
            resumed["config"]["initial_checkpoint_sha256"]
        best = torch.load(path / "resume/best.pt", weights_only=True)
        assert best["kind"] == "tts_reliability_model"
        assert "optimizer" not in best
        preserved = {name: (path / "full" / name).read_bytes()
                     for name in ("latest.pt", "config.json", "train.jsonl")}
        with pytest.raises(FileExistsError, match="already contains run artifacts"):
            run(parser().parse_args(common + [
                "--mode", "train", "--output", str(path / "full"), "--steps", "4"]))
        with pytest.raises(ValueError, match="resume configuration mismatch: grad_clip"):
            run(parser().parse_args(common + [
                "--mode", "train", "--output", str(path / "resume"), "--steps", "4",
                "--resume", str(path / "resume/latest.pt"), "--grad-clip", "0.5"]))
        for name, content in preserved.items():
            assert (path / "full" / name).read_bytes() == content


def test_validation_is_token_weighted_and_preserves_model():
    torch.manual_seed(8)
    model = TextToSpeechLM(17, 11, d_model=16, n_heads=4, n_layers=2, max_seq_len=32)
    data = PreparedDataset(fixture_data())
    state = {key: value.clone() for key, value in model.state_dict().items()}
    before_rng = torch.get_rng_state()
    singleton = validate(model, data, [[i] for i in range(8)], torch.device("cpu"))
    batched = validate(model, data, [list(range(8))], torch.device("cpu"))
    assert abs(singleton["loss"] - batched["loss"]) < 1e-6
    assert abs(singleton["accuracy"] - batched["accuracy"]) < 1e-6
    assert model.training
    assert torch.equal(before_rng, torch.get_rng_state())
    for key, value in state.items():
        assert torch.equal(value, model.state_dict()[key])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_sampler_coverage_and_cursor_for_next_batch():
    lengths = [(i, 2 + i % 3, 3 + i % 5) for i in range(23)]
    sampler = BucketSampler(lengths, 4)
    seen = []
    while sampler.epoch == 0:
        seen.extend(sampler.peek())
        sampler.advance()
        clone = BucketSampler(lengths, 4)
        clone.load_state_dict(sampler.state_dict())
        assert clone.peek() == sampler.peek()
    assert sorted(seen) == list(range(23))
    first = validation_batches(lengths, 4, 2, 184)
    assert first == validation_batches(lengths, 4, 2, 184)
    assert len({i for batch in first for i in batch}) == 8
