"""Verify starter compatibility, cached logits, frozen decoding and local ASR."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from hw4_util import (load_text_tokenizer, load_flow_model, load_vocoder,
                      generate_audio_from_tokens, load_speech_tokenizer, prompt_features)
from tts_checkpoint import load_tts_starter
from tts_reliability.model import TextToSpeechLM


def main():
    torch.set_num_threads(6)
    torch.manual_seed(183)
    assert torch.backends.mps.is_available(), "MPS required for LM smoke check"
    out = Path("results/smoke")
    out.mkdir(parents=True, exist_ok=True)
    report = {"seed": 183, "lm_device": "mps", "flow_device": "mps",
              "vocoder_device": "cpu", "asr_device": "cpu"}
    tok = load_text_tokenizer("models/cosy")
    model = TextToSpeechLM(tok.vocab_size, 6561)
    load_tts_starter(model, tok, "models/hw4-tts-starter.pt")
    report["parameters"] = sum(p.numel() for p in model.parameters())
    model = model.to("mps").eval()
    report["text"] = "This is a short speech synthesis test."
    text = torch.tensor([tok.encode(report["text"])], device="mps")
    inputs, _, _ = model.prepare_sequence(text, torch.tensor([text.shape[1]], device="mps"))
    with torch.inference_mode():
        full = model._logits(inputs)
        cached, _ = model._cached_logits(inputs)
        report["cache_max_abs_error"] = (full - cached).abs().max().item()
        torch.testing.assert_close(full, cached, rtol=2e-4, atol=2e-4)
    print("Verified starter and MPS cache", report, flush=True)
    kwargs = {}
    prompt_path = Path("data/reference/prompt.wav")
    prompt = None
    if prompt_path.exists():
        reference = json.loads(prompt_path.with_suffix(".json").read_text())
        audio, rate = sf.read(prompt_path, dtype="float32")
        assert rate == 16000
        speech_tokenizer = load_speech_tokenizer("models/cosy", "cpu")
        # Feature extraction explicitly runs on CPU: ONNX, STFT and resampling.
        prompt = prompt_features(torch.from_numpy(audio)[None], speech_tokenizer, "models/cosy", "cpu")
        kwargs.update(prompt_speech_token=prompt[0].to("mps"),
                      prompt_text_tokens=torch.tensor([tok.encode(reference["text"])], device="mps"))
        report["reference"] = reference
    start = time.monotonic()
    speech, details = model.generate_with_metadata(text, max_length=800,
        min_length=text.shape[1] * 2, use_cache=True, **kwargs)
    torch.mps.synchronize()
    report["generation"] = details
    report["generation_seconds"] = time.monotonic() - start
    torch.save(speech.cpu(), out / "speech_tokens.pt")
    print("Generated", report["generation"], flush=True)
    del model, inputs, full, cached
    gc.collect()
    torch.mps.empty_cache()
    start = time.monotonic()
    flow = load_flow_model("models/cosy", "mps")
    vocoder = load_vocoder("models/cosy", "cpu")
    def length(n):
        return torch.tensor([n], device="mps", dtype=torch.int32)
    with torch.inference_mode():
        if prompt is None:
            ptoken = torch.empty(1, 0, dtype=torch.long, device="mps")
            pmel = torch.empty(1, 0, 80, device="mps")
            embedding = torch.zeros(1, 192, device="mps")
        else:
            ptoken, pmel, embedding = [x.to("mps") for x in prompt]
        mel, _ = flow.inference(token=speech, token_len=length(speech.shape[1]),
            prompt_token=ptoken, prompt_token_len=length(ptoken.shape[1]),
            prompt_feat=pmel, prompt_feat_len=length(pmel.shape[1]), embedding=embedding,
            streaming=False, finalize=True)
        audio, _ = vocoder.inference(speech_feat=mel.cpu(), cache_source=torch.zeros(1, 1, 0))
    audio = audio.flatten().cpu().numpy()
    assert np.isfinite(audio).all() and len(audio)
    sf.write(out / "starter.wav", audio, 24000)
    report["audio_seconds"] = len(audio) / 24000
    report["decoding_seconds"] = time.monotonic() - start
    print("Saved full waveform", report["audio_seconds"], flush=True)
    del flow, vocoder, mel
    gc.collect()
    torch.mps.empty_cache()
    from transformers import pipeline
    import librosa
    import jiwer
    start = time.monotonic()
    asr = pipeline("automatic-speech-recognition", model="models/whisper",
                   device="cpu", torch_dtype=torch.float32)
    hypothesis = asr({"array": librosa.resample(audio, orig_sr=24000, target_sr=16000),
                      "sampling_rate": 16000},
                     generate_kwargs={"language": "english", "task": "transcribe"})["text"]
    from whisper.normalizers import EnglishTextNormalizer
    norm = EnglishTextNormalizer()
    alignment = jiwer.process_words(norm(report["text"]), norm(hypothesis))
    report.update(transcript=hypothesis, wer=alignment.wer, substitutions=alignment.substitutions,
                  deletions=alignment.deletions, insertions=alignment.insertions,
                  asr_seconds=time.monotonic() - start)
    (out / "smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
