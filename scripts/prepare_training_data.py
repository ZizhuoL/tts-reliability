"""Validate every paired sample, tokenize once, and retain original indices."""
import collections
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from hw4_util import load_text_tokenizer
from tts_reliability.data import CosyVoiceDataset


def main():
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_text_tokenizer("models/cosy")
    report = {}
    for split, file in (("train", "libritts_train.pt"), ("dev", "libritts_dev.pt")):
        started = time.monotonic()
        raw = torch.load(Path("data/tts") / file, map_location="cpu", weights_only=True)
        metadata, samples = raw["metadata"], raw["samples"]
        valid, rejected = [], collections.Counter()
        lengths, speakers = [], collections.Counter()
        for offset in range(0, len(samples), 1024):
            group = samples[offset:offset + 1024]
            texts = [x.get("text", "") if isinstance(x.get("text"), str) else "" for x in group]
            encoded = tokenizer.tokenizer(texts, add_special_tokens=True)["input_ids"]
            for j, (sample, text_ids) in enumerate(zip(group, encoded)):
                speech = CosyVoiceDataset._ids(sample.get("speech_tokens"))
                if not texts[j].strip() or not sample.get("utt_id") or "speaker_id" not in sample:
                    rejected["missing_fields_or_empty_text"] += 1
                elif speech is None or len(speech) == 0:
                    rejected["invalid_or_empty_speech"] += 1
                elif (speech < 0).any() or (speech >= metadata["speech_vocab_size"]).any():
                    rejected["speech_id_range"] += 1
                elif len(speech) > 1200:
                    rejected["speech_length_above_1200"] += 1
                elif not 0 < len(text_ids) <= 512:
                    rejected["text_length_not_1_to_512"] += 1
                else:
                    valid.append(dict(
                        utt=sample["utt_id"], text=texts[j],
                        text_token=torch.tensor(text_ids, dtype=torch.long),
                        speech_token=speech, speaker_id=sample["speaker_id"],
                        speaker_idx=metadata["speaker_to_idx"].get(sample["speaker_id"], -1),
                        original_index=offset + j))
                    lengths.append((len(text_ids), len(speech)))
                    speakers[str(sample["speaker_id"])] += 1
            if offset % 20480 == 0:
                print(f"{split} {offset}/{len(samples)} valid={len(valid)}", flush=True)
        array = np.asarray(lengths)
        quantiles = [0, .1, .25, .5, .75, .9, .95, .99, 1]
        stats = dict(raw_count=len(samples), valid_count=len(valid), rejected=dict(rejected),
                     quantile_levels=quantiles,
                     text_token_quantiles=np.quantile(array[:, 0], quantiles).tolist(),
                     speech_token_quantiles=np.quantile(array[:, 1], quantiles).tolist(),
                     speech_per_text_quantiles=np.quantile(array[:, 1] / array[:, 0], quantiles).tolist(),
                     total_speech_targets=int(array[:, 1].sum() + len(valid)),
                     total_text_tokens=int(array[:, 0].sum()), speakers=len(speakers),
                     seconds=time.monotonic() - started)
        torch.save(dict(samples=valid, metadata=metadata, statistics=stats), out / f"{split}.pt")
        np.save(out / f"{split}_lengths.npy", array)
        (out / f"{split}_speakers.json").write_text(json.dumps(speakers, indent=2))
        report[split] = stats
        print(json.dumps({split: stats}), flush=True)
        del raw, samples, valid
    (out / "statistics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
