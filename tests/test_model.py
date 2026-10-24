"""Small deterministic checks of model semantics, independent of downloads."""
import json
from pathlib import Path
import sys
import unittest

import torch

from tts_reliability.data import CosyVoiceDataset, cosyvoice_collate_fn
from tts_reliability.model import TextToSpeechLM


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(183)
        self.model = TextToSpeechLM(17, 11, d_model=16, n_heads=4,
                                    n_layers=2, max_seq_len=32)
        self.text = torch.tensor([[3, 4, 0], [5, 6, 7]])
        self.text_len = torch.tensor([2, 3])
        self.speech = torch.tensor([[8, 9, 10], [2, 0, 0]])
        self.speech_len = torch.tensor([3, 1])

    def test_exact_shift_padding_and_embedding_gradients(self):
        embeddings, targets, padding = self.model.prepare_sequence(
            self.text, self.text_len, self.speech, self.speech_len)
        self.assertEqual(targets.tolist(),
                         [[-1, -1, -1, 8, 9, 10, 11],
                          [-1, -1, -1, -1, 2, 11, -1]])
        self.assertEqual(padding.tolist(),
                         [[False] * 7, [False] * 6 + [True]])
        expected = torch.cat((
            self.model.special_embedding.weight[:1],
            self.model.text_embedding(self.text[0, :2]),
            self.model.special_embedding.weight[1:2],
            self.model.speech_embedding(self.speech[0])))
        torch.testing.assert_close(embeddings[0], expected)
        loss, accuracy = self.model(
            self.text, self.text_len, self.speech, self.speech_len)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(0 <= accuracy <= 1)
        loss.backward()
        for table in (self.model.text_embedding, self.model.speech_embedding,
                      self.model.special_embedding):
            self.assertGreater(table.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.model.text_embedding.weight.grad[0].abs().sum(), 0)

    def test_causality_and_batch_padding_do_not_change_valid_logits(self):
        self.model.eval()
        inputs, _, padding = self.model.prepare_sequence(
            self.text, self.text_len, self.speech, self.speech_len)
        with torch.no_grad():
            original = self.model._logits(inputs, padding)
            changed = inputs.clone()
            changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 10
            altered = self.model._logits(changed, padding)
            torch.testing.assert_close(original[:, :5], altered[:, :5])
            single = self.model._logits(inputs[1:2, :6])
            torch.testing.assert_close(original[1:2, :6], single, atol=1e-6, rtol=1e-5)

    def test_kv_cache_matches_full_history_including_chunked_decode(self):
        self.model.eval()
        inputs = torch.randn(1, 9, 16)
        with torch.no_grad():
            expected = self.model._logits(inputs)
            initial, cache = self.model._cached_logits(inputs[:, :4])
            middle, cache = self.model._cached_logits(inputs[:, 4:7], cache)
            last_one, cache = self.model._cached_logits(inputs[:, 7:8], cache)
            last_two, _ = self.model._cached_logits(inputs[:, 8:], cache)
        actual = torch.cat((initial, middle, last_one, last_two), dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_generation_eos_minimum_prefix_and_context_boundaries(self):
        with torch.no_grad():
            self.model.output_projection.weight.zero_()
            self.model.output_projection.bias.fill_(-100)
            self.model.output_projection.bias[11] = 100
            self.model.output_projection.bias[3] = 50
        args = dict(text_tokens=torch.tensor([[4, 5]]),
                    prompt_text_tokens=torch.tensor([[1]]),
                    prompt_speech_token=torch.tensor([[6, 7]]),
                    min_length=2, max_length=5, top_k=1)
        for use_cache in (False, True):
            generated, meta = self.model.generate_with_metadata(
                **args, use_cache=use_cache)
            self.assertEqual(generated.tolist(), [[3, 3]])
            self.assertEqual(meta["prefix_tokens"], 7)
            self.assertEqual(meta["stop_reason"], "eos")
            self.assertFalse(self.model.training)
            self.assertFalse(generated.requires_grad)
        generated, meta = self.model.generate_with_metadata(
            torch.tensor([[2] * 30]), max_length=4, min_length=4, top_k=1)
        self.assertEqual(generated.shape, (1, 1))
        self.assertEqual(meta["stop_reason"], "context_limit")
        generated, meta = self.model.generate_with_metadata(
            torch.tensor([[2] * 31]), max_length=4, top_k=1)
        self.assertEqual(generated.shape, (1, 0))
        self.assertEqual(meta["stop_reason"], "context_limit")
        generated, meta = self.model.generate_with_metadata(
            torch.tensor([[2]]), max_length=2, min_length=2, top_k=1)
        self.assertEqual(generated.tolist(), [[3, 3]])
        self.assertEqual(meta["stop_reason"], "output_limit")

    def test_optimizer_and_state_restore(self):
        before = self.model.output_projection.weight.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        loss, _ = self.model(self.text, self.text_len, self.speech, self.speech_len)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.output_projection.weight))
        clone = TextToSpeechLM(17, 11, d_model=16, n_heads=4,
                              n_layers=2, max_seq_len=32)
        clone.load_state_dict(self.model.state_dict(), strict=True)
        self.model.eval()
        clone.eval()
        with torch.no_grad():
            expected = self.model(self.text, self.text_len, self.speech, self.speech_len)
            actual = clone(self.text, self.text_len, self.speech, self.speech_len)
        torch.testing.assert_close(actual, expected)

    def test_checkpointed_gradients_match_regular_training(self):
        clone = TextToSpeechLM(17, 11, d_model=16, n_heads=4,
                              n_layers=2, max_seq_len=32)
        clone.load_state_dict(self.model.state_dict())
        clone.gradient_checkpointing = True
        self.model(self.text, self.text_len, self.speech, self.speech_len)[0].backward()
        clone(self.text, self.text_len, self.speech, self.speech_len)[0].backward()
        for left, right in zip(self.model.parameters(), clone.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=1e-5)

    def test_starter_architecture_metadata_matches(self):
        upstream = Path(__file__).resolve().parents[1] / "upstream/eecs183-hw4"
        sys.path.insert(0, str(upstream))
        from tts_checkpoint import describe_architecture
        manifest = json.loads((upstream / "tts_starter_manifest.json").read_text())
        # Meta device validates full starter shapes without allocating 0.8 GB.
        with torch.device("meta"):
            full = TextToSpeechLM(151663, 6561)
        self.assertEqual(describe_architecture(full), manifest["architecture"])


class DatasetTests(unittest.TestCase):
    def test_validation_pairing_and_sorted_collation(self):
        class Tokenizer:
            vocab_size = 20

            def encode(self, text):
                return [int(x) for x in text.split()]

        rows = [
            dict(utt_id="short", text="1 2", speech_tokens=[3], speaker_id="known"),
            dict(utt_id="long", text="4", speech_tokens=[2, 3, 4], speaker_id="new"),
            dict(utt_id="too_long", text="1 2 3 4", speech_tokens=[2], speaker_id="known"),
            dict(utt_id="invalid", text="2", speech_tokens=[1.5], speaker_id="known"),
            dict(utt_id="eos_in_audio", text="2", speech_tokens=[11], speaker_id="known"),
        ]
        dataset = CosyVoiceDataset(rows, Tokenizer(), {"known": 5},
                                   max_text_len=3, speech_vocab_size=11)
        batch = cosyvoice_collate_fn([dataset[i] for i in range(len(dataset))])
        self.assertEqual(batch["utts"], ["long", "short"])
        self.assertEqual(batch["text"], ["4", "1 2"])
        self.assertEqual(batch["text_tokens"].tolist(), [[4, 0], [1, 2]])
        self.assertEqual(batch["speech_lengths"].tolist(), [3, 1])
        self.assertEqual(batch["speaker_indices"].tolist(), [-1, 5])
        self.assertIsNone(cosyvoice_collate_fn([None, None]))


if __name__ == "__main__":
    unittest.main()
def test_shape_bucketing_preserves_supervised_loss_and_gradients():
    import copy
    import torch
    from tts_reliability.model import TextToSpeechLM
    torch.manual_seed(25)
    original = TextToSpeechLM(20, 10, d_model=16, n_heads=2, n_layers=2, max_seq_len=64)
    bucketed = copy.deepcopy(original)
    bucketed.pad_multiple = 16
    text = torch.tensor([[1, 2, 0], [3, 4, 5]])
    speech = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    lengths, speech_lengths = torch.tensor([2, 3]), torch.tensor([4, 2])
    for model in (original, bucketed):
        model.train()
    a, aa = original(text, lengths, speech, speech_lengths)
    b, ba = bucketed(text, lengths, speech, speech_lengths)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(aa, ba)
    a.backward()
    b.backward()
    for p, q in zip(original.parameters(), bucketed.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
