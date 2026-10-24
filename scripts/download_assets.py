"""Download fixed public course/model assets; no credentials or uploads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import ssl

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/eecs183-hw4"
sys.path.insert(0, str(UPSTREAM))


def fetch(url, destination, spec):
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    destination.parent.mkdir(parents=True, exist_ok=True)
    def valid(path):
        if not path.exists() or path.stat().st_size != spec["bytes"]:
            return False
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest() == spec["sha256"]
    if valid(destination):
        print(f"Verified cached {destination.name}", flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120, context=context) as response, temporary.open("wb") as out:
                total = 0
                last = time.monotonic()
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
                    total += len(block)
                    if time.monotonic() - last > 10:
                        print(f"{destination.name}: {total}/{spec['bytes']} bytes", flush=True)
                        last = time.monotonic()
            if not valid(temporary):
                raise ValueError(f"Checksum/size mismatch: {destination.name}")
            temporary.replace(destination)
            print(f"Verified {destination.name}", flush=True)
            return
        except Exception as exc:
            print(f"Attempt {attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    args = parser.parse_args()
    from course_download import asset_spec
    for name in ("hw4-data-c.zip", "hw4-tts-starter.pt"):
        destination = ROOT / ("data" if name.endswith(".zip") else "models") / name
        fetch(f"https://huggingface.co/datasets/Louis0324/eecs183-hw4/resolve/hw4-2026-v4/{name}",
              destination, asset_spec(name))
    if args.models:
        from course_download import ensure_course_data
        ensure_course_data("c", ROOT / "data", ROOT / "data/hw4-data-c.zip")
        from huggingface_hub import snapshot_download
        manifest = json.loads((UPSTREAM / "model_manifest.json").read_text())
        for key, patterns in (
            ("cosy", ["CosyVoice-BlankEN/*.json", "CosyVoice-BlankEN/*.txt",
                      "flow.pt", "hift.pt", "speech_tokenizer_v2.onnx", "campplus.onnx"]),
            ("whisper", ["*.json", "*.txt", "*.safetensors", "*.model"]),
        ):
            spec = manifest[key]
            for attempt in range(3):
                try:
                    path = snapshot_download(spec["repository"], revision=spec["revision"],
                        local_dir=ROOT / "models" / key, allow_patterns=patterns, max_workers=2)
                    print(f"Downloaded {key}: {path}", flush=True)
                    break
                except Exception as exc:
                    print(f"{key} attempt {attempt + 1}: {exc}", flush=True)
                    if attempt == 2:
                        raise
                    time.sleep(5)


if __name__ == "__main__":
    main()
