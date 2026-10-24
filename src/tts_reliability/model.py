"""Course-compatible causal text-to-speech LM and measured generation boundaries.

Architecture/module names follow upstream ``hw4-c.ipynb``. The implementation of
sequence preparation, forward, generation and the optional KV cache is original
to this project. No extra learned parameters are introduced.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

IGNORE_ID = -1


class TextToSpeechLM(nn.Module):
    def __init__(self, text_vocab_size, speech_vocab_size, d_model=768,
                 n_heads=12, n_layers=12, max_seq_len=2048):
        super().__init__()
        self.model_config = dict(d_model=d_model, n_heads=n_heads,
                                 n_layers=n_layers, max_seq_len=max_seq_len)
        self.d_model = d_model
        self.speech_vocab_size = speech_vocab_size
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.speech_embedding = nn.Embedding(speech_vocab_size + 1, d_model)
        self.special_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, n_layers, enable_nested_tensor=False)
        self.output_projection = nn.Linear(d_model, speech_vocab_size + 1)
        # Runtime-only option: state dict and starter compatibility are unchanged.
        self.gradient_checkpointing = False
        # Bound MPS training graph shapes. Zero leaves course behavior unchanged.
        self.pad_multiple = 0

    @staticmethod
    def _lengths(tokens, lengths, name):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError(f"{name} tokens must be int64 [batch, length]")
        if lengths.ndim != 1 or len(lengths) != len(tokens):
            raise ValueError(f"{name} lengths must be [batch]")
        if lengths.dtype != torch.long:
            raise ValueError(f"{name} lengths must be int64")
        values = lengths.detach().cpu().tolist()
        if any(n < 0 or n > tokens.shape[1] for n in values):
            raise ValueError(f"{name} length outside padded tensor")
        return values

    def prepare_sequence(self, text_tokens, text_lengths, speech_tokens=None,
                         speech_lengths=None):
        text_lens = self._lengths(text_tokens, text_lengths, "text")
        if len(text_lens) == 0:
            raise ValueError("empty batch")
        speech_lens = None
        if speech_tokens is not None:
            if speech_lengths is None or len(speech_tokens) != len(text_tokens):
                raise ValueError("speech tokens require matching lengths and batch")
            speech_lens = self._lengths(speech_tokens, speech_lengths, "speech")
        elif speech_lengths is not None:
            raise ValueError("speech lengths require speech tokens")
        sequences, targets = [], []
        for i, text_len in enumerate(text_lens):
            prefix = torch.cat((
                self.special_embedding.weight[0:1],
                self.text_embedding(text_tokens[i, :text_len]),
                self.special_embedding.weight[1:2]), dim=0)
            target = torch.full((len(prefix),), IGNORE_ID, dtype=torch.long,
                                device=text_tokens.device)
            if speech_lens is not None:
                speech = speech_tokens[i, :speech_lens[i]]
                prefix = torch.cat((prefix, self.speech_embedding(speech)), dim=0)
                # TASK predicts speech[0]; final speech input predicts EOS.
                target = torch.cat((
                    target[:-1], speech,
                    speech.new_tensor([self.speech_vocab_size])), dim=0)
            if len(prefix) > self.position_embedding.num_embeddings:
                raise ValueError("sequence exceeds learned position capacity")
            sequences.append(prefix)
            targets.append(target)
        inputs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        target = pad_sequence(targets, batch_first=True, padding_value=IGNORE_ID)
        lens = torch.tensor([len(x) for x in sequences], device=inputs.device)
        padding = torch.arange(inputs.shape[1], device=inputs.device)[None] >= lens[:, None]
        if self.pad_multiple:
            padded_length = min(
                self.position_embedding.num_embeddings,
                math.ceil(inputs.shape[1] / self.pad_multiple) * self.pad_multiple)
            extra = padded_length - inputs.shape[1]
            if extra:
                inputs = F.pad(inputs, (0, 0, 0, extra))
                target = F.pad(target, (0, extra), value=IGNORE_ID)
                padding = F.pad(padding, (0, extra), value=True)
        return inputs, target, padding

    def _logits(self, inputs, padding=None):
        length = inputs.shape[1]
        if length > self.position_embedding.num_embeddings:
            raise ValueError("sequence exceeds learned position capacity")
        positions = torch.arange(length, device=inputs.device)
        hidden = inputs + self.position_embedding(positions)[None]
        causal = torch.ones((length, length), device=inputs.device,
                            dtype=torch.bool).triu(diagonal=1)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            for layer in self.transformer.layers:
                hidden = checkpoint(
                    layer, hidden, src_mask=causal, src_key_padding_mask=padding,
                    is_causal=True, use_reentrant=False)
            if self.transformer.norm is not None:
                hidden = self.transformer.norm(hidden)
        else:
            hidden = self.transformer(
                hidden, mask=causal, src_key_padding_mask=padding, is_causal=True)
        return self.output_projection(hidden)

    def forward(self, text_tokens, text_lengths, speech_tokens, speech_lengths):
        inputs, targets, padding = self.prepare_sequence(
            text_tokens, text_lengths, speech_tokens, speech_lengths)
        logits = self._logits(inputs, padding)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            ignore_index=IGNORE_ID)
        valid = targets != IGNORE_ID
        accuracy = (logits.argmax(-1)[valid] == targets[valid]).float().mean()
        return loss, accuracy

    def _cached_logits(self, inputs, cache=None):
        """Eval-only dense prefix / incremental decode using original weights.

        Cache is a list of (key, value) tensors, one per layer. There is no padding
        in generation. This intentionally does not replace the training path.
        """
        if self.training:
            raise ValueError("KV cache is only valid in eval mode")
        offset = 0 if cache is None else cache[0][0].shape[-2]
        length = inputs.shape[1]
        if offset + length > self.position_embedding.num_embeddings:
            raise ValueError("sequence exceeds learned position capacity")
        positions = torch.arange(offset, offset + length, device=inputs.device)
        hidden = inputs + self.position_embedding(positions)[None]
        next_cache = []
        for index, layer in enumerate(self.transformer.layers):
            attn = layer.self_attn
            batch, size, width = hidden.shape
            heads, head_dim = attn.num_heads, width // attn.num_heads
            q, k, v = F.linear(
                hidden, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            q, k, v = [x.reshape(batch, size, heads, head_dim).transpose(1, 2)
                       for x in (q, k, v)]
            if cache is not None:
                k = torch.cat((cache[index][0], k), dim=-2)
                v = torch.cat((cache[index][1], v), dim=-2)
            next_cache.append((k, v))
            # SDPA bool masks use True=allowed, opposite Transformer mask.
            allowed = None
            if offset and size > 1:
                allowed = (torch.arange(k.shape[-2], device=hidden.device)[None]
                           <= positions[:, None])
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed,
                is_causal=(offset == 0), dropout_p=0.0)
            attended = attended.transpose(1, 2).reshape(batch, size, width)
            attended = F.linear(attended, attn.out_proj.weight, attn.out_proj.bias)
            hidden = layer.norm1(hidden + attended)
            feedforward = layer.linear2(layer.activation(layer.linear1(hidden)))
            hidden = layer.norm2(hidden + feedforward)
        if self.transformer.norm is not None:
            hidden = self.transformer.norm(hidden)
        return self.output_projection(hidden), next_cache

    @torch.no_grad()
    def generate(self, text_tokens, max_length=500, temperature=1.0, top_k=50,
                 prompt_speech_token=None, prompt_text_tokens=None,
                 min_length=None, *, use_cache=False, generator=None,
                 return_metadata=False):
        """Generate only new speech IDs; never include reference prefix or EOS.

        A fixed explicit generator gives reproducible sampling on its device.
        ``min_length=None`` means zero; upstream synthesis passes its own 2x-text
        heuristic. An EOS is allowed after exactly min_length speech tokens.
        """
        self.eval()
        started = time.perf_counter()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not isinstance(max_length, int) or max_length < 0:
            raise ValueError("max_length must be a nonnegative integer")
        min_length = 0 if min_length is None else min_length
        if not isinstance(min_length, int) or not 0 <= min_length <= max_length:
            raise ValueError("min_length must be between zero and max_length")
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")

        def tokens(value, name):
            if value is None:
                return text_tokens.new_empty((1, 0))
            if value.dtype != torch.long or value.ndim != 2 or value.shape[0] != 1:
                raise ValueError(f"{name} must be int64 [1, length]")
            if value.device != text_tokens.device:
                raise ValueError(f"{name} must be on the text token device")
            return value

        text_tokens = tokens(text_tokens, "text_tokens")
        prompt_text_tokens = tokens(prompt_text_tokens, "prompt_text_tokens")
        prompt_speech_token = tokens(prompt_speech_token, "prompt_speech_token")
        all_text = torch.cat((prompt_text_tokens, text_tokens), dim=1)
        prefix = torch.cat((
            self.special_embedding.weight[0:1][None],
            self.text_embedding(all_text),
            self.special_embedding.weight[1:2][None],
            self.speech_embedding(prompt_speech_token)), dim=1)
        prefix_length = prefix.shape[1]
        capacity = self.position_embedding.num_embeddings
        generated, cache = [], None
        reason = "output_limit"
        eos_emitted = False
        # A full-capacity prefix can still predict one new token. Feeding that
        # token back would exceed capacity, so the next iteration is blocked.
        current = prefix
        if prefix_length > capacity:
            reason = "context_limit"
        else:
            for index in range(max_length):
                if prefix_length + index > capacity:
                    reason = "context_limit"
                    break
                if use_cache:
                    logits, cache = self._cached_logits(current, cache)
                else:
                    logits = self._logits(current)
                scores = logits[:, -1, :] / temperature
                if index < min_length:
                    scores[:, self.speech_vocab_size] = -torch.inf
                if top_k:
                    k = min(top_k, scores.shape[-1])
                    values, ids = torch.topk(scores, k, dim=-1)
                    filtered = torch.full_like(scores, -torch.inf)
                    scores = filtered.scatter(-1, ids, values)
                probabilities = F.softmax(scores, dim=-1)
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError("nonfinite generation probabilities")
                token = torch.multinomial(probabilities, 1, generator=generator)
                if token.item() == self.speech_vocab_size:
                    reason, eos_emitted = "eos", True
                    break
                generated.append(token)
                embedding = self.speech_embedding(token)
                current = embedding if use_cache else torch.cat((current, embedding), dim=1)
        output = (torch.cat(generated, dim=1) if generated
                  else text_tokens.new_empty((1, 0)))
        metadata = dict(
            stop_reason=reason, eos_emitted=eos_emitted,
            generated_tokens=output.shape[1], prefix_tokens=prefix_length,
            target_text_tokens=text_tokens.shape[1],
            prompt_text_tokens=prompt_text_tokens.shape[1],
            prompt_speech_tokens=prompt_speech_token.shape[1],
            max_length=max_length, min_length=min_length, temperature=temperature,
            top_k=top_k, use_cache=use_cache, context_capacity=capacity,
            elapsed_seconds=time.perf_counter() - started)
        return (output, metadata) if return_metadata else output

    def generate_with_metadata(self, *args, **kwargs):
        kwargs["return_metadata"] = True
        return self.generate(*args, **kwargs)
