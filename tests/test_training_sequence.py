"""Original versus packed semantics and complete parameter-gradient equivalence."""
import pytest
import torch

from tts_reliability.model import TextToSpeechLM
from tts_reliability.training_sequence import install_sequence_implementation


@pytest.mark.parametrize("pad_multiple", [0, 8, 64])
def test_packed_embeddings_logits_loss_and_all_gradients_match(pad_multiple):
    torch.manual_seed(29)
    original = TextToSpeechLM(17, 11, d_model=16, n_heads=4, n_layers=2, max_seq_len=32)
    packed = TextToSpeechLM(17, 11, d_model=16, n_heads=4, n_layers=2, max_seq_len=32)
    packed.load_state_dict(original.state_dict())
    original.pad_multiple = packed.pad_multiple = pad_multiple
    install_sequence_implementation(packed, "packed")
    # Out-of-vocabulary IDs only in padding must never reach an embedding table.
    text = torch.tensor([[3, 4, 999], [5, 6, 7], [999, 999, 999]])
    text_lengths = torch.tensor([2, 3, 0])
    speech = torch.tensor([[8, 9, 10], [2, 999, 999], [999, 999, 999]])
    speech_lengths = torch.tensor([3, 1, 0])
    arguments = (text, text_lengths, speech, speech_lengths)
    first = original.prepare_sequence(*arguments)
    second = packed.prepare_sequence(*arguments)
    for expected, actual in zip(first, second):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_logits = original._logits(first[0], first[2])
    actual_logits = packed._logits(second[0], second[2])
    torch.testing.assert_close(actual_logits[~second[2]], expected_logits[~first[2]],
                               rtol=0, atol=0)
    expected_loss, expected_accuracy = original(*arguments)
    actual_loss, actual_accuracy = packed(*arguments)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual_accuracy, expected_accuracy, rtol=0, atol=0)
    expected_loss.backward()
    actual_loss.backward()
    for (name, parameter), (packed_name, packed_parameter) in zip(
            original.named_parameters(), packed.named_parameters()):
        assert name == packed_name
        assert parameter.grad is not None
        assert packed_parameter.grad is not None
        torch.testing.assert_close(packed_parameter.grad, parameter.grad,
                                   rtol=1e-5, atol=1e-7, msg=lambda message: f"{name}: {message}")
    assert set(original.state_dict()) == set(packed.state_dict())
    install_sequence_implementation(packed, "original")
    assert "prepare_sequence" not in packed.__dict__
    for expected, restored in zip(first, packed.prepare_sequence(*arguments)):
        torch.testing.assert_close(expected, restored, rtol=0, atol=0)


def test_packed_text_only_and_zero_width_inputs():
    torch.manual_seed(30)
    original = TextToSpeechLM(17, 11, d_model=16, n_heads=4, n_layers=2, max_seq_len=32)
    packed = TextToSpeechLM(17, 11, d_model=16, n_heads=4, n_layers=2, max_seq_len=32)
    packed.load_state_dict(original.state_dict())
    install_sequence_implementation(packed, "packed")
    for text, lengths in [(torch.tensor([[2, 3], [4, 0]]), torch.tensor([2, 1])),
                          (torch.empty((2, 0), dtype=torch.long), torch.tensor([0, 0]))]:
        for speech, speech_lengths in [(None, None),
                                       (torch.empty((2, 0), dtype=torch.long), torch.tensor([0, 0]))]:
            expected = original.prepare_sequence(text, lengths, speech, speech_lengths)
            actual = packed.prepare_sequence(text, lengths, speech, speech_lengths)
            for left, right in zip(expected, actual):
                torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_packed_rejects_over_capacity():
    model = TextToSpeechLM(17, 11, d_model=16, n_heads=4, n_layers=2, max_seq_len=8)
    install_sequence_implementation(model, "packed")
    with pytest.raises(ValueError, match="position capacity"):
        model.prepare_sequence(torch.ones(1, 5, dtype=torch.long), torch.tensor([5]),
                               torch.ones(1, 3, dtype=torch.long), torch.tensor([3]))
