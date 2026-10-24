"""Validated, pairing-preserving access to the course LibriTTS token cache."""
from __future__ import annotations

import numbers

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CosyVoiceDataset(Dataset):
    def __init__(self, samples_list, text_tokenizer, speaker_to_idx_dict,
                 max_text_len=512, max_speech_len=1200, speech_vocab_size=6561):
        self.samples = samples_list
        self.text_tokenizer = text_tokenizer
        self.speaker_to_idx = speaker_to_idx_dict
        self.max_text_len = max_text_len
        self.max_speech_len = max_speech_len
        self.speech_vocab_size = speech_vocab_size

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _ids(value):
        if isinstance(value, torch.Tensor):
            if value.ndim != 1 or value.dtype not in (
                    torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
                return None
            return value.to(dtype=torch.long)
        if not isinstance(value, (list, tuple)) or any(
                not isinstance(x, numbers.Integral) or isinstance(x, bool) for x in value):
            return None
        return torch.tensor(value, dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not isinstance(sample, dict):
            return None
        text = sample.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        if not sample.get("utt_id") or "speaker_id" not in sample:
            return None
        speech = self._ids(sample.get("speech_tokens"))
        if speech is None or not 0 < len(speech) <= self.max_speech_len:
            return None
        if (speech < 0).any() or (speech >= self.speech_vocab_size).any():
            return None
        text_ids = self._ids(self.text_tokenizer.encode(text))
        if text_ids is None or not 0 < len(text_ids) <= self.max_text_len:
            return None
        if (text_ids < 0).any():
            return None
        vocab_size = getattr(self.text_tokenizer, "vocab_size", None)
        if vocab_size is not None and (text_ids >= vocab_size).any():
            return None
        speaker = sample["speaker_id"]
        return dict(utt=sample["utt_id"], text=text, text_token=text_ids,
                    speech_token=speech,
                    speaker_idx=self.speaker_to_idx.get(speaker, -1),
                    speaker_id=speaker)


def cosyvoice_collate_fn(batch):
    valid = sorted((item for item in batch if item is not None),
                   key=lambda item: len(item["speech_token"]), reverse=True)
    if not valid:
        return None
    return dict(
        utts=[item["utt"] for item in valid],
        text=[item["text"] for item in valid],
        text_tokens=pad_sequence([item["text_token"] for item in valid],
                                 batch_first=True, padding_value=0),
        text_lengths=torch.tensor([len(item["text_token"]) for item in valid],
                                  dtype=torch.long),
        speech_tokens=pad_sequence([item["speech_token"] for item in valid],
                                   batch_first=True, padding_value=0),
        speech_lengths=torch.tensor([len(item["speech_token"]) for item in valid],
                                    dtype=torch.long),
        speaker_indices=torch.tensor([item["speaker_idx"] for item in valid],
                                     dtype=torch.long))
