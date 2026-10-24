"""Local training, representative throughput measurements, and exact batch resume."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import psutil
import torch

from .data import CosyVoiceDataset, cosyvoice_collate_fn
from .model import TextToSpeechLM
from .training_sequence import install_sequence_implementation


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_record(device):
    result = {"rss_bytes": psutil.Process().memory_info().rss}
    try:
        result.update(system_available_bytes=psutil.virtual_memory().available,
                      system_swap_used_bytes=psutil.swap_memory().used)
    except (OSError, psutil.AccessDenied):
        # Sandboxed CPU checks may not have access to host-wide counters.
        result["system_memory_counters_unavailable"] = True
    if device.type == "mps":
        result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    return result


class PreparedDataset:
    """Accept canonical collator records or common plural token field names."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        if row is None:
            return None
        return dict(
            utt=row.get("utt", row.get("utt_id", str(index))),
            text=row.get("text", ""),
            text_token=torch.as_tensor(row.get("text_token", row.get("text_tokens")),
                                       dtype=torch.long),
            speech_token=torch.as_tensor(row.get("speech_token", row.get("speech_tokens")),
                                         dtype=torch.long),
            speaker_idx=row.get("speaker_idx", -1),
            speaker_id=row.get("speaker_id", "unknown"))


def load_dataset(path, tokenizer_dir):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    samples = cache["samples"] if isinstance(cache, dict) else cache
    metadata = cache.get("metadata", {}) if isinstance(cache, dict) else {}
    first = next(row for row in samples if row is not None)
    if "text_token" in first or "text_tokens" in first:
        return PreparedDataset(samples), metadata
    from hw4_util import load_text_tokenizer
    tokenizer = load_text_tokenizer(str(tokenizer_dir))
    return CosyVoiceDataset(
        samples, tokenizer, metadata.get("speaker_to_idx", {}),
        speech_vocab_size=metadata.get("speech_vocab_size", 6561)), metadata


def valid_lengths(dataset):
    """Materialize only index/length arrays; never silently truncate samples."""
    records = []
    for index in range(len(dataset)):
        row = dataset[index]
        if row is not None:
            records.append((index, len(row["text_token"]), len(row["speech_token"])))
    if not records:
        raise ValueError("no valid examples")
    return records


class BucketSampler:
    """Deterministic shuffled epochs; cursor advances only after an update.

    ``long_weight > 1`` draws N examples with replacement using increased weight
    for the longest speech-token quartile. Ordinary sampling covers every valid
    example exactly once per epoch, including the final partial batch.
    """
    def __init__(self, lengths, batch_size, seed=183, long_weight=1.0,
                 bucket_batches=50):
        if batch_size < 1 or long_weight < 1:
            raise ValueError("batch_size >= 1 and long_weight >= 1 are required")
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.long_weight = long_weight
        self.bucket_batches = bucket_batches
        self.epoch, self.cursor = 0, 0
        self._batches = None

    def _build(self):
        rng = random.Random(self.seed + self.epoch)
        rows = list(self.lengths)
        if self.long_weight > 1:
            boundary = sorted(row[2] for row in rows)[int(0.75 * (len(rows) - 1))]
            weights = [self.long_weight if row[2] >= boundary else 1 for row in rows]
            rows = rng.choices(rows, weights=weights, k=len(rows))
        else:
            rng.shuffle(rows)
        width = self.batch_size * self.bucket_batches
        batches = []
        for start in range(0, len(rows), width):
            bucket = sorted(rows[start:start + width], key=lambda row: row[1] + row[2])
            batches.extend([[row[0] for row in bucket[i:i + self.batch_size]]
                            for i in range(0, len(bucket), self.batch_size)])
        rng.shuffle(batches)
        self._batches = batches

    def peek(self):
        if self._batches is None:
            self._build()
        return self._batches[self.cursor]

    def advance(self):
        self.cursor += 1
        if self.cursor == len(self._batches):
            self.epoch += 1
            self.cursor = 0
            self._batches = None

    def state_dict(self):
        return dict(epoch=self.epoch, cursor=self.cursor, seed=self.seed,
                    batch_size=self.batch_size, long_weight=self.long_weight,
                    bucket_batches=self.bucket_batches)

    def load_state_dict(self, state):
        for key in ("seed", "batch_size", "long_weight", "bucket_batches"):
            if state[key] != getattr(self, key):
                raise ValueError(f"sampler resume mismatch: {key}")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        self._batches = None


class TokenScheduler:
    """Warmup and cosine decay by supervised speech targets, including EOS."""
    def __init__(self, optimizer, learning_rate, budget, warmup_fraction=0.1):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.budget = max(1, budget)
        self.warmup_fraction = warmup_fraction
        self.tokens = 0
        self._apply()

    def _apply(self):
        fraction = min(1.0, self.tokens / self.budget)
        if fraction < self.warmup_fraction:
            # A nonzero first learning rate avoids a no-op first update.
            scale = max(0.01, fraction / self.warmup_fraction)
        else:
            progress = ((fraction - self.warmup_fraction)
                        / max(1e-12, 1 - self.warmup_fraction))
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * scale

    def step(self, target_tokens):
        self.tokens += target_tokens
        self._apply()

    def state_dict(self):
        return dict(tokens=self.tokens, learning_rate=self.learning_rate,
                    budget=self.budget, warmup_fraction=self.warmup_fraction)

    def load_state_dict(self, state):
        self.tokens = state["tokens"]
        for name in ("learning_rate", "budget", "warmup_fraction"):
            if state[name] != getattr(self, name):
                raise ValueError(f"scheduler resume mismatch: {name}")
        self._apply()


def train_step(model, batch, optimizer, device, grad_clip=1.0, empty_cache=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    input_device = "cpu" if getattr(model, "_sequence_implementation", "original") == "packed" else device
    moved = {name: batch[name].to(input_device) for name in
             ("text_tokens", "text_lengths", "speech_tokens", "speech_lengths")}
    synchronize(device)
    started = time.perf_counter()
    loss, accuracy = model(**moved)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(norm).item():
        raise FloatingPointError("nonfinite training gradients")
    optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    text_tokens = int(batch["text_lengths"].sum())
    speech_tokens = int(batch["speech_lengths"].sum())
    batch_size = len(batch["utts"])
    effective = text_tokens + speech_tokens + 2 * batch_size
    padded_length = int((batch["text_lengths"] + batch["speech_lengths"] + 2).max())
    multiple = getattr(model, "pad_multiple", 1) or 1
    padded_length = min(model.position_embedding.num_embeddings,
                        math.ceil(padded_length / multiple) * multiple)
    padded = batch_size * padded_length
    result = dict(loss=loss.item(), accuracy=accuracy.item(), grad_norm=norm.item(),
                  compute_seconds=elapsed, examples=batch_size, text_tokens=text_tokens,
                  speech_tokens=speech_tokens, target_tokens=speech_tokens + batch_size,
                  effective_input_tokens=effective, padded_input_tokens=padded,
                  padding_fraction=1 - effective / padded, max_sequence=padded_length,
                  learning_rate=optimizer.param_groups[0]["lr"],
                  sequence_implementation=getattr(model, "_sequence_implementation", "original"),
                  **memory_record(device))
    del moved, loss, accuracy, norm
    result.update(empty_cache_requested=empty_cache, empty_cache_executed=False,
                  empty_cache_seconds=0.0)
    if empty_cache and device.type == "mps":
        result["memory_before_empty_cache"] = memory_record(device)
        started_clear = time.perf_counter()
        torch.mps.empty_cache()
        synchronize(device)
        result["empty_cache_seconds"] = time.perf_counter() - started_clear
        result["empty_cache_executed"] = True
        result["memory_after_empty_cache"] = memory_record(device)
        result.update(result["memory_after_empty_cache"])
    return result


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, scheduler, sampler, counters, config, device):
    synchronize(device)
    rng = dict(python=random.getstate(), torch=torch.get_rng_state())
    if device.type == "mps":
        rng["mps"] = torch.mps.get_rng_state().cpu()
    atomic_save(path, dict(
        format_version=1, kind="tts_reliability_training",
        model_config=model.model_config,
        text_vocab_size=model.text_embedding.num_embeddings,
        speech_vocab_size=model.speech_vocab_size,
        model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), sampler=sampler.state_dict(),
        counters=counters, config=config, rng=rng))


def load_model(checkpoint_path, device, tokenizer_dir=None):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = state.get("architecture", {})
    config = state.get("model_config", {})
    text_vocab = state.get("text_vocab_size", architecture.get("text_vocab_size"))
    speech_vocab = state.get("speech_vocab_size", architecture.get("speech_vocab_size"))
    if text_vocab is None or speech_vocab is None:
        weights = state["model_state_dict"]
        text_vocab = weights["text_embedding.weight"].shape[0]
        speech_vocab = weights["speech_embedding.weight"].shape[0] - 1
    model = TextToSpeechLM(text_vocab, speech_vocab, **config)
    if state.get("kind") == "hw4_tts_starter" and tokenizer_dir is not None:
        from hw4_util import load_text_tokenizer
        from tts_checkpoint import load_tts_starter
        # Reuse upstream's full SHA, architecture and token-to-ID checks.
        load_tts_starter(model, load_text_tokenizer(str(tokenizer_dir)), checkpoint_path)
    else:
        model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32)
    return model, state


@torch.no_grad()
def validate(model, dataset, batches, device):
    """Exact token-weighted teacher-forced metrics; no parameter/optimizer updates."""
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    loss_sum, correct_sum, target_count, examples = 0.0, 0.0, 0, 0
    try:
        for ids in batches:
            batch = cosyvoice_collate_fn([dataset[i] for i in ids])
            if batch is None:
                continue
            input_device = "cpu" if getattr(model, "_sequence_implementation", "original") == "packed" else device
            moved = {key: batch[key].to(input_device) for key in
                     ("text_tokens", "text_lengths", "speech_tokens", "speech_lengths")}
            loss, accuracy = model(**moved)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite validation loss")
            count = int(batch["speech_lengths"].sum()) + len(batch["utts"])
            loss_sum += loss.item() * count
            correct_sum += accuracy.item() * count
            target_count += count
            examples += len(batch["utts"])
        synchronize(device)
    finally:
        model.train(was_training)
    if not target_count:
        raise ValueError("validation has no supervised targets")
    return dict(loss=loss_sum / target_count, accuracy=correct_sum / target_count,
                target_tokens=target_count, examples=examples,
                seconds=time.perf_counter() - started, **memory_record(device))


def validation_batches(lengths, batch_size, max_batches, seed):
    rows = list(lengths)
    if max_batches and len(rows) > max_batches * batch_size:
        rows = random.Random(seed).sample(rows, max_batches * batch_size)
    rows.sort(key=lambda row: row[1] + row[2])
    return [[row[0] for row in rows[i:i + batch_size]]
            for i in range(0, len(rows), batch_size)]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_model_checkpoint(path, model, counters, config, validation):
    atomic_save(path, dict(
        format_version=1, kind="tts_reliability_model",
        model_config=model.model_config,
        text_vocab_size=model.text_embedding.num_embeddings,
        speech_vocab_size=model.speech_vocab_size,
        model_state_dict=model.state_dict(), counters=counters,
        config=config, validation=validation))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(args):
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; no automatic CPU fallback")
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(getattr(args, "mps_memory_fraction", 0.9))
    if device.type not in ("cpu", "mps"):
        raise ValueError("this project supports local CPU/MPS only")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protected_artifacts = (
        "config.json", "latest.pt", "latest.pt.tmp", "best.pt", "train.jsonl",
        "validation.jsonl", "validation.json", "summary.json", "benchmark.json",
        "benchmark.jsonl")
    existing = [name for name in protected_artifacts if (output / name).exists()]
    if existing and not args.resume:
        raise FileExistsError(
            f"Output already contains run artifacts {existing}; use a fresh --output "
            "or explicitly --resume an existing training checkpoint")
    config = vars(args).copy()
    config.update(torch_version=str(torch.__version__), dtype="float32")
    config["implementation_sha256"] = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("model.py", "data.py", "training.py", "training_sequence.py")
    }
    if args.mode == "validate" and not args.val_data:
        raise ValueError("--mode validate requires --val-data")
    data_path = Path(args.val_data if args.mode == "validate" else args.data)
    config["data_identity"] = dict(path=str(data_path.resolve()),
                                  bytes=data_path.stat().st_size,
                                  mtime_ns=data_path.stat().st_mtime_ns)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)
    dataset, metadata = load_dataset(data_path, args.tokenizer_dir)
    lengths = valid_lengths(dataset)
    full_epoch_tokens = sum(row[2] + 1 for row in lengths)
    budget = args.token_budget or full_epoch_tokens * args.epochs
    config.update(valid_examples=len(lengths), full_epoch_target_tokens=full_epoch_tokens,
                  effective_token_budget=budget)
    val_dataset, val_batches = None, None
    if args.val_data:
        val_path = Path(args.val_data)
        val_dataset = dataset if val_path.resolve() == data_path.resolve() else \
            load_dataset(val_path, args.tokenizer_dir)[0]
        val_batches = validation_batches(valid_lengths(val_dataset), args.batch_size,
                                         args.val_max_batches, args.seed + 1)
        config["validation"] = dict(
            path=str(val_path.resolve()), bytes=val_path.stat().st_size,
            mtime_ns=val_path.stat().st_mtime_ns, batches=val_batches,
            subset_seed=args.seed + 1,
            purpose="course dev; checkpoint selection, not independent final test")
    sampler = BucketSampler(lengths, args.batch_size, args.seed, args.long_weight)
    source = args.resume or args.checkpoint
    config["checkpoint_sha256"] = file_sha256(source)
    model, state = load_model(source, device, args.tokenizer_dir)
    config["initial_checkpoint_sha256"] = state.get("config", {}).get(
        "initial_checkpoint_sha256", config["checkpoint_sha256"])
    model.gradient_checkpointing = args.gradient_checkpointing
    model.pad_multiple = args.pad_multiple
    install_sequence_implementation(model, args.sequence_implementation)
    if args.mode == "validate":
        if val_dataset is None:
            raise ValueError("--mode validate requires --val-data")
        result = validate(model, val_dataset, val_batches, device)
        write_json(output / "config.json", config)
        write_json(output / "validation.json", result)
        print(json.dumps(result), flush=True)
        return result
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = TokenScheduler(optimizer, args.learning_rate, budget, args.warmup_fraction)
    counters = dict(steps=0, target_tokens=0, examples=0, compute_seconds=0.0,
                    effective_input_tokens=0, padded_input_tokens=0, seen_indices=[])
    if args.resume:
        if state.get("kind") != "tts_reliability_training":
            raise ValueError("--resume requires this project's training checkpoint")
        for key in ("data_identity", "batch_size", "seed", "long_weight",
                    "learning_rate", "weight_decay", "gradient_checkpointing",
                    "effective_token_budget", "warmup_fraction", "validation",
                    "pad_multiple", "grad_clip", "device", "sequence_implementation",
                    "empty_cache_every", "implementation_sha256"):
            if state["config"].get(key) != config.get(key):
                raise ValueError(f"resume configuration mismatch: {key}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        counters = state["counters"]
        random.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"])
        if device.type == "mps" and "mps" in state["rng"]:
            torch.mps.set_rng_state(state["rng"]["mps"])
    del state
    write_json(output / "config.json", config)
    print(json.dumps(dict(event="ready", device=str(device), valid_examples=len(lengths),
                          full_epoch_target_tokens=full_epoch_tokens,
                          model_parameters=sum(p.numel() for p in model.parameters()),
                          **memory_record(device))), flush=True)
    if args.mode == "benchmark":
        steps = args.steps or 8
        # Select actual training batches, retaining their real within-batch
        # variation, instead of fabricating homogeneous quantile batches.
        length_lookup = {row[0]: row[1] + row[2] + 2 for row in lengths}
        sampler._build()
        actual_batches = sorted(sampler._batches,
                                key=lambda ids: max(length_lookup[i] for i in ids))
        quantiles = [(i + 0.5) / steps for i in range(steps)]
        quantiles[-1] = 0.99
        selected = [(q, actual_batches[min(len(actual_batches) - 1,
                                         int(q * len(actual_batches)))])
                    for q in quantiles]
        median = actual_batches[len(actual_batches) // 2]
        measurements = []
        seen_shapes = set()
        initial = model.output_projection.weight.detach().cpu().clone()
        schedule = [(0.5, median, "global_warmup", i) for i in range(args.warmup_steps)]
        for quantile, ids in selected:
            schedule.append((quantile, ids, "first_for_batch", 0))
            schedule.extend((quantile, ids, "steady", i + 1)
                            for i in range(args.benchmark_repeats))
        # Separate safety probe: use the real batch containing the maximum
        # sequence, preferring the fuller batch on ties. Never include this
        # extreme batch in the representative steady-throughput average.
        worst_ids = max(actual_batches,
                        key=lambda ids: (max(length_lookup[i] for i in ids), len(ids)))
        schedule.append((1.0, worst_ids, "worst_case", 0))
        log_path = output / "benchmark.jsonl"
        with log_path.open("w") as stream:
            stream.write("")
        for index, (quantile, ids, phase, repetition) in enumerate(schedule):
            started = time.perf_counter()
            load_started = time.perf_counter()
            batch = cosyvoice_collate_fn([dataset[i] for i in ids])
            load_seconds = time.perf_counter() - load_started
            raw_length = max(length_lookup[i] for i in ids)
            shape = (len(ids), min(model.position_embedding.num_embeddings,
                                  math.ceil(raw_length / args.pad_multiple) * args.pad_multiple))
            before_probe = None
            if phase == "worst_case":
                before_probe = model.output_projection.bias.detach().cpu().clone()
                before_probe_memory = memory_record(device)
            try:
                clear_cache = bool(args.empty_cache_every and (index + 1) % args.empty_cache_every == 0)
                row = train_step(model, batch, optimizer, device, args.grad_clip,
                                 empty_cache=clear_cache)
                if phase == "worst_case":
                    updated = not torch.equal(
                        before_probe, model.output_projection.bias.detach().cpu())
                    if not updated:
                        raise RuntimeError("worst-case optimizer did not change output bias")
                    row.update(status="passed", parameters_updated=True,
                               loss_finite=True, gradient_finite=True,
                               memory_before=before_probe_memory)
            except Exception as error:
                if phase == "worst_case":
                    failure = dict(
                        status="error", error_type=type(error).__name__, error=str(error),
                        sample_indices=ids, raw_max_sequence=raw_length,
                        padded_sequence=shape[1], examples=len(ids),
                        memory_before=before_probe_memory, **memory_record(device))
                    write_json(output / "benchmark.json", dict(
                        mode="benchmark", status="worst_case_failed", config=config,
                        measurements=measurements, worst_case=failure))
                raise
            row.update(step=index, quantile=quantile, sample_indices=ids,
                       phase=phase, repetition=repetition, raw_max_sequence=raw_length,
                       shape_cold=shape not in seen_shapes,
                       warmup=phase == "global_warmup", load_seconds=load_seconds,
                       total_seconds=time.perf_counter() - started)
            seen_shapes.add(shape)
            measurements.append(row)
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        measured = [row for row in measurements if row["phase"] == "steady"]
        seconds = sum(row["total_seconds"] for row in measured)
        changed = not torch.equal(initial, model.output_projection.weight.detach().cpu())
        if not changed:
            raise RuntimeError("benchmark optimizer did not change output projection weights")
        # Estimate every real epoch batch from measured steady timings at nearby
        # padded lengths. This is explicitly an interpolation, not a measured epoch.
        duration_by_length = {}
        for row in measured:
            duration_by_length.setdefault(row["max_sequence"], []).append(
                row["total_seconds"] * args.batch_size / row["examples"])
        points = sorted((size, sum(values) / len(values))
                        for size, values in duration_by_length.items())

        def estimate_batch(ids):
            size = min(model.position_embedding.num_embeddings,
                       math.ceil(max(length_lookup[i] for i in ids) / args.pad_multiple)
                       * args.pad_multiple)
            prediction = points[-1][1]
            if size <= points[0][0]:
                prediction = points[0][1]
            else:
                for (left, left_time), (right, right_time) in zip(points, points[1:]):
                    if left <= size <= right:
                        prediction = left_time + (right_time - left_time) * (
                            size - left) / (right - left)
                        break
            return prediction * len(ids) / args.batch_size

        epoch_seconds = sum(estimate_batch(ids) for ids in actual_batches)
        summary = dict(
            mode="benchmark", config=config, measurements=measurements,
            parameters_updated=changed, measured_steps=len(measured),
            mean_step_seconds=seconds / len(measured), steps_per_second=len(measured) / seconds,
            target_tokens_per_second=sum(row["target_tokens"] for row in measured) / seconds,
            estimated_full_epoch_hours=epoch_seconds / 3600,
            epoch_estimation_method="linear interpolation of steady step times by padded length over actual sampler batches; excludes validation/checkpoint saving",
            actual_epoch_batches=len(actual_batches),
            steady_timing_points=[dict(padded_length=size, seconds=duration)
                                  for size, duration in points],
            cold_shape_seconds=sum(row["total_seconds"] for row in measurements
                                   if row["shape_cold"]),
            unique_shapes=len(seen_shapes),
            steady_padding_fraction=1 - sum(row["effective_input_tokens"] for row in measured)
            / sum(row["padded_input_tokens"] for row in measured),
            worst_case=next(row for row in measurements if row["phase"] == "worst_case"),
            steady_empty_cache_seconds=sum(row["empty_cache_seconds"] for row in measured),
            total_seconds=seconds, **memory_record(device))
        write_json(output / "benchmark.json", summary)
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("config", "measurements")}), flush=True)
        return summary
    seen = set(counters.pop("seen_indices", []))
    last_save = time.monotonic()
    started_run = time.monotonic()
    log = output / "train.jsonl"
    stop_reason = "token_budget"
    while counters["target_tokens"] < budget:
        if args.steps and counters["steps"] >= args.steps:
            stop_reason = "step_limit"
            break
        if args.max_hours and time.monotonic() - started_run >= args.max_hours * 3600:
            stop_reason = "time_limit"
            break
        ids = sampler.peek()
        started = time.perf_counter()
        batch = cosyvoice_collate_fn([dataset[i] for i in ids])
        if batch is None:
            raise RuntimeError("previously validated batch became empty")
        clear_cache = bool(args.empty_cache_every
                           and (counters["steps"] + 1) % args.empty_cache_every == 0)
        row = train_step(model, batch, optimizer, device, args.grad_clip,
                         empty_cache=clear_cache)
        scheduler.step(row["target_tokens"])
        sampler.advance()
        seen.update(ids)
        counters["steps"] += 1
        for key in ("target_tokens", "examples", "compute_seconds",
                    "effective_input_tokens", "padded_input_tokens"):
            counters[key] += row[key]
        row.update(step=counters["steps"], epoch=sampler.epoch,
                   epoch_batch_cursor=sampler.cursor,
                   total_target_tokens=counters["target_tokens"],
                   unique_examples=len(seen), total_seconds=time.perf_counter() - started)
        with log.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if counters["steps"] % args.log_every == 0 or counters["steps"] == 1:
            print(json.dumps(row), flush=True)
        if val_dataset is not None and args.val_every and counters["steps"] % args.val_every == 0:
            metrics = validate(model, val_dataset, val_batches, device)
            metrics.update(step=counters["steps"], event="validation")
            with (output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if metrics["loss"] < counters.get("best_validation_loss", float("inf")):
                counters["best_validation_loss"] = metrics["loss"]
                counters["best_validation_step"] = counters["steps"]
                save_model_checkpoint(output / "best.pt", model, counters, config, metrics)
        if (counters["steps"] % args.save_every == 0
                or time.monotonic() - last_save >= args.save_minutes * 60):
            save_checkpoint(output / "latest.pt", model, optimizer, scheduler, sampler,
                            dict(counters, seen_indices=sorted(seen)), config, device)
            last_save = time.monotonic()
            print(json.dumps(dict(event="checkpoint", steps=counters["steps"],
                                  path=str(output / "latest.pt"))), flush=True)
    save_checkpoint(output / "latest.pt", model, optimizer, scheduler, sampler,
                    dict(counters, seen_indices=sorted(seen)), config, device)
    summary = dict(counters, stop_reason=stop_reason, unique_examples=len(seen),
                   valid_examples=len(lengths), completed_epochs=sampler.epoch,
                   epoch_batch_cursor=sampler.cursor,
                   wall_seconds_this_run=time.monotonic() - started_run,
                   checkpoint=str(output / "latest.pt"))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary
