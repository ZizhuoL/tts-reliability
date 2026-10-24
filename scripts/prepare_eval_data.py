"""Prepare fixed public prompt and reproducible, held-out natural text manifests.

No TTS output is used for selection. Final text is saved without printing it.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import ssl
import sys
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
PARQUET_SHA = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
REFERENCE_ID = "1272-128104-0000"
REFERENCE_TEXT = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"
BOOKS = {
    "alice": ("Lewis Carroll", "Alice's Adventures in Wonderland", 11),
    "holmes": ("Arthur Conan Doyle", "The Adventures of Sherlock Holmes", 1661),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text))


def grams(text: str, n: int = 8) -> set[str]:
    words = normalized(text).split()
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}


def fetch(url: str, dest: Path, expected_sha: str | None = None) -> None:
    import certifi
    if dest.exists():
        if expected_sha and sha(dest.read_bytes()) != expected_sha:
            raise ValueError(f"Existing download checksum mismatch: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    errors = []
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120, context=context) as response:
                data = response.read()
            if expected_sha and sha(data) != expected_sha:
                raise ValueError("Downloaded checksum mismatch")
            temp = dest.with_suffix(dest.suffix + ".partial")
            temp.write_bytes(data)
            temp.replace(dest)
            return
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(f"Failed public download {url}: {errors}")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def prepare_reference() -> None:
    import pyarrow.parquet as pq
    import soundfile as sf
    directory = ROOT / "data/reference"
    source = directory / "librispeech_dummy.parquet"
    url = f"https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/{REVISION}/clean/validation-00000-of-00001.parquet"
    fetch(url, source, PARQUET_SHA)
    rows = pq.read_table(source).to_pylist()
    row = next(r for r in rows if r["id"] == REFERENCE_ID)
    assert row["text"] == REFERENCE_TEXT
    audio_bytes = row["audio"]["bytes"]
    original = directory / f"{REFERENCE_ID}.flac"
    original.write_bytes(audio_bytes)
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    assert sr == 16000 and audio.ndim == 1 and len(audio) > 16000
    wav = directory / "prompt.wav"
    sf.write(wav, audio, sr, subtype="PCM_16")
    write_json(directory / "prompt.json", {
        "id": REFERENCE_ID, "text": REFERENCE_TEXT,
        "audio_path": "data/reference/prompt.wav",
        "sample_rate": sr, "num_samples": len(audio), "duration_seconds": len(audio)/sr,
        "license": "CC BY 4.0", "license_source": "https://www.openslr.org/12",
        "attribution": "LibriSpeech, Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur (ICASSP 2015)",
        "source_url": url, "revision": REVISION, "source_sha256": PARQUET_SHA,
        "original_audio_sha256": sha(audio_bytes), "wav_sha256": sha(wav.read_bytes()),
        "transformation": "Decode FLAC and write mono PCM16 WAV at original 16000 Hz; no trimming or resampling.",
        "transcript_status": "Exact dataset transcript; not manually listening-verified.",
    })
    print(f"Prepared fixed reference: {REFERENCE_ID}, {len(audio)/sr:.3f}s at {sr}Hz")


def sentence_chunks(paragraph: str) -> list[str]:
    # Protect common abbreviations; source segment IDs refer to retained paragraph.
    protected = re.sub(r"\b(Mr|Mrs|Ms|Dr|St|Prof)\.", r"\1<DOT>", paragraph)
    return [s.replace("<DOT>", ".").strip() for s in
            re.split(r'(?<=[.!?])\s+(?=[“"‘A-Z])', protected) if s.strip()]


def book_candidates(directory: Path):
    sources = []
    candidates = []
    for key, (author, title, ebook_id) in BOOKS.items():
        url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        path = directory / "sources" / f"pg{ebook_id}.txt"
        fetch(url, path)
        raw = path.read_text(encoding="utf-8-sig")
        start = re.search(r"\*\*\* START OF .*?\*\*\*", raw)
        end = re.search(r"\*\*\* END OF .*?\*\*\*", raw)
        assert start and end
        body = raw[start.end():end.start()]
        sources.append({"key": key, "author": author, "title": title,
                        "url": url, "sha256": sha(path.read_bytes()),
                        "license": "Public domain in the USA; original Gutenberg license retained in source file.",
                        "metadata_url": f"https://www.gutenberg.org/ebooks/{ebook_id}"})
        for paragraph_index, paragraph in enumerate(re.split(r"\n\s*\n", body)):
            # Exclude headings, verse, and very short fragments before joining lines.
            lines = [s.strip() for s in paragraph.splitlines() if s.strip()]
            if not lines or any("CHAPTER " in s or "[Illustration" in s for s in lines):
                continue
            if len(lines) >= 3 and sum(len(s) < 35 for s in lines) > len(lines)/2:
                continue
            paragraph = " ".join(lines).replace("_", "")
            if not paragraph or not re.search(r"[.!?][”’\"']?$", paragraph):
                continue
            # The grouping key ensures overlapping spans cannot straddle dev/final.
            group_id = f"{key}:paragraph:{paragraph_index}"
            split = "dev" if int(sha(group_id.encode())[:8], 16) % 2 == 0 else "final"
            sentences = sentence_chunks(paragraph)
            for i in range(len(sentences)):
                for width in range(1, min(4, len(sentences)-i)+1):
                    text = " ".join(sentences[i:i+width])
                    if len(normalized(text).split()) < 5 or len(text) > 1100:
                        continue
                    if any(c.isalpha() and not (c.isascii()) for c in text):
                        continue
                    candidates.append({
                        "id": f"{key}-p{paragraph_index:05d}-s{i:02d}-n{width}",
                        "split": split, "text": text, "source_key": key,
                        "source_url": url, "paragraph_index": paragraph_index,
                        "sentence_start": i, "sentence_count": width, "group_id": group_id,
                    })
    return candidates, sources


def prepare_texts(args) -> None:
    import torch
    from transformers import AutoTokenizer
    directory = ROOT / "data/eval"
    dev_path, final_path = directory / "dev.jsonl", directory / "final.sealed.jsonl"
    if dev_path.exists() or final_path.exists():
        raise RuntimeError("Refusing to overwrite existing fixed eval manifests; archive them explicitly before rebuilding.")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/cosy/CosyVoice-BlankEN", local_files_only=True)
    candidates, sources = book_candidates(directory)
    ranges = {"short": (args.min_tokens, args.short_max),
              "medium": (args.short_max+1, args.medium_max),
              "long": (args.medium_max+1, args.max_tokens)}
    candidates = [dict(c, text_tokens=len(tokenizer.encode(c["text"]))) for c in candidates]
    candidates = [c for c in candidates if args.min_tokens <= c["text_tokens"] <= args.max_tokens]
    candidate_grams = set().union(*(grams(c["text"]) for c in candidates))
    contaminated_grams, corpus_texts = set(), set()
    corpus_sources = []
    for filename in ("libritts_train.pt", "libritts_dev.pt"):
        path = ROOT / "data/tts" / filename
        records = torch.load(path, map_location="cpu", weights_only=False)["samples"]
        for row in records:
            text = row["text"]
            corpus_texts.add(normalized(text))
            contaminated_grams.update(candidate_grams.intersection(grams(text)))
        corpus_sources.append({"file": filename, "records": len(records), "sha256": sha(path.read_bytes())})
        del records
    rejected = Counter()
    pools = {}
    for c in candidates:
        if normalized(c["text"]) in corpus_texts:
            rejected["exact_train_or_course_dev"] += 1
            continue
        if any(normalized(s) in corpus_texts for s in sentence_chunks(c["text"])):
            rejected["component_sentence_train_or_course_dev"] += 1
            continue
        if grams(c["text"]) & contaminated_grams:
            rejected["shared_8gram_train_or_course_dev"] += 1
            continue
        group = next(k for k, (lo, hi) in ranges.items() if lo <= c["text_tokens"] <= hi)
        c["length_group"] = group
        pools.setdefault((c["split"], group), []).append(c)
    rng = random.Random(args.seed)
    selections, used_paragraphs, used_normalized, used_grams = {}, set(), set(), set()
    for split, count in (("dev", 15), ("final", 20)):
        selected = []
        for group in ranges:
            pool = sorted(pools.get((split, group), []), key=lambda c: c["id"])
            rng.shuffle(pool)
            chosen = []
            for c in pool:
                norm, eight = normalized(c["text"]), grams(c["text"])
                if c["group_id"] in used_paragraphs or norm in used_normalized or eight & used_grams:
                    continue
                # Alternate sources where available to reduce source/length confounding.
                if len(chosen) < count and sum(x["source_key"] == c["source_key"] for x in chosen) >= (count+1)//2:
                    continue
                chosen.append(c)
                used_paragraphs.add(c["group_id"])
                used_normalized.add(norm)
                used_grams.update(eight)
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise RuntimeError(f"Insufficient {split}/{group}: found {len(chosen)} of {count}; no manifests written.")
            selected.extend(chosen)
        selections[split] = selected
    directory.mkdir(parents=True, exist_ok=True)
    for split, path in (("dev", dev_path), ("final", final_path)):
        path.write_text("".join(json.dumps(c, ensure_ascii=False)+"\n" for c in selections[split]))
    write_json(directory / "manifest.json", {
        "seed": args.seed, "ranges_inclusive_text_tokens": ranges,
        "sources": sources, "dedup_against": corpus_sources,
        "candidate_count_within_length_bounds": len(candidates), "rejected": dict(rejected),
        "selection": "Fixed seed; paragraph-hash dev/final split; at most one span per paragraph; balanced sources per length group.",
        "dedup": "NFKC/casefold ASCII word normalization; exact complete and component sentences; any shared 8-word sequence; also no shared 8-gram across selected eval texts.",
        "counts": {s: dict(Counter(c["length_group"] for c in rows)) for s, rows in selections.items()},
        "source_counts": {s: dict(Counter(c["source_key"] for c in rows)) for s, rows in selections.items()},
        "dev_sha256": sha(dev_path.read_bytes()), "final_sha256": sha(final_path.read_bytes()),
        "final_policy": "Sealed from model selection: do not synthesize or inspect final outcomes until method/checkpoint/configuration are frozen.",
        "limitations": [
            "Deduplication does not establish absence from unknown starter or frozen-component pretraining data.",
            "Lexical 8-gram rejection is conservative but does not detect every paraphrase or overlap shorter than eight words.",
            "Public-domain literary prose is a narrow domain; this does not estimate all practical TTS usage.",
            "Final manifest is a procedural holdout, not an access-control security boundary.",
        ],
    })
    print(json.dumps({"prepared_counts": {k: len(v) for k, v in selections.items()},
                      "ranges": ranges, "rejected": dict(rejected)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--texts-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1832026)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--short-max", type=int, default=20)
    parser.add_argument("--medium-max", type=int, default=45)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()
    if not args.texts_only:
        prepare_reference()
    if not args.reference_only:
        prepare_texts(args)


if __name__ == "__main__":
    main()
