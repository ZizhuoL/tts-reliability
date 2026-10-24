"""Optional training-only sequence packing; leaves the model's source/weights intact.

Build token-index grids and masks on CPU, then perform one embedding lookup per
table on a shared [B, padded_length] shape. This avoids a separate embedding
autograd branch and MPS graph shape for every variable-length sample. Memory and
speed benefits must be measured; equivalence is tested independently.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
from torch.nn import functional as F

IGNORE_ID = -1


def prepare_sequence_packed(self, text_tokens, text_lengths, speech_tokens=None,
                            speech_lengths=None):
    text_lens = self._lengths(text_tokens, text_lengths, "text")
    if not text_lens:
        raise ValueError("empty batch")
    if speech_tokens is not None:
        if speech_lengths is None or len(speech_tokens) != len(text_tokens):
            raise ValueError("speech tokens require matching lengths and batch")
        speech_lens = self._lengths(speech_tokens, speech_lengths, "speech")
    else:
        if speech_lengths is not None:
            raise ValueError("speech lengths require speech tokens")
        speech_lens = [0] * len(text_lens)
    text_lens = torch.tensor(text_lens, dtype=torch.long)[:, None]
    speech_lens = torch.tensor(speech_lens, dtype=torch.long)[:, None]
    lengths = text_lens + speech_lens + 2
    length = int(lengths.max())
    capacity = self.position_embedding.num_embeddings
    if length > capacity:
        raise ValueError("sequence exceeds learned position capacity")
    multiple = getattr(self, "pad_multiple", 0)
    if multiple:
        length = min(capacity, math.ceil(length / multiple) * multiple)
    batch_size = len(text_lens)
    position = torch.arange(length, dtype=torch.long)[None].expand(batch_size, -1)
    text_mask = (position >= 1) & (position <= text_lens)
    speech_mask = ((position >= text_lens + 2)
                   & (position < text_lens + speech_lens + 2))
    special_mask = (position == 0) | (position == text_lens + 1)
    padding = position >= lengths

    # Append a sentinel so even zero-width token tensors have a safe gather
    # index. Unused source padding is masked before any embedding lookup.
    text_source = F.pad(text_tokens.detach().cpu(), (0, 1))
    text_indices = (position - 1).clamp(min=0, max=text_tokens.shape[1])
    text_ids = text_source.gather(1, text_indices).masked_fill(~text_mask, 0)
    special_ids = (position == text_lens + 1).long()
    speech_ids = torch.zeros_like(text_ids)
    targets = torch.full_like(text_ids, IGNORE_ID)
    if speech_tokens is not None:
        speech_source = F.pad(speech_tokens.detach().cpu(), (0, 1))
        speech_indices = (position - text_lens - 2).clamp(
            min=0, max=speech_tokens.shape[1])
        speech_ids = speech_source.gather(1, speech_indices).masked_fill(~speech_mask, 0)
        next_indices = (position - text_lens - 1).clamp(
            min=0, max=speech_tokens.shape[1])
        predicts_speech = ((position >= text_lens + 1)
                           & (position < text_lens + speech_lens + 1))
        targets = speech_source.gather(1, next_indices).masked_fill(
            ~predicts_speech, IGNORE_ID)
        targets = targets.masked_fill(
            position == text_lens + speech_lens + 1, self.speech_vocab_size)

    device = self.text_embedding.weight.device
    # Each lookup has exactly the same batch/sequence shape, irrespective of
    # individual sample lengths. Masks preserve gradients only for actual tokens.
    inputs = (
        self.text_embedding(text_ids.to(device)) * text_mask.to(device)[..., None]
        + self.speech_embedding(speech_ids.to(device)) * speech_mask.to(device)[..., None]
        + self.special_embedding(special_ids.to(device)) * special_mask.to(device)[..., None])
    return inputs, targets.to(device), padding.to(device)


def install_sequence_implementation(model, implementation="original"):
    if implementation == "packed":
        model.prepare_sequence = MethodType(prepare_sequence_packed, model)
    elif implementation == "original":
        if "prepare_sequence" in model.__dict__:
            delattr(model, "prepare_sequence")
    else:
        raise ValueError(f"unknown sequence implementation: {implementation}")
    model._sequence_implementation = implementation
    return model
