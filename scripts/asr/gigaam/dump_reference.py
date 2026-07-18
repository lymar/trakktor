#!/usr/bin/env python3
"""Dump GigaAM reference geometry, vocab, mel filterbank, and golden traces.

Runs the reference `gigaam` package (CPU, fp32) so the Rust port can be
validated against exact intermediates. Writes to an output directory:

  config.json        model geometry (encoder/preprocessor/head/decoding) + vocab
  state_dict.json    every weight tensor name -> shape/dtype
  mel_fbank.bin      torchaudio mel filterbank, raw f32 LE, shape [n_freqs, n_mels]
  <clip>.pcm.bin     the clip's 16 kHz mono f32 samples (feed the Rust model this)
  <clip>.trace.json  manifest: shapes, decoded text/tokens/frames, file names
  <clip>.mel.bin     log-mel features [n_mels, T], raw f32 LE
  <clip>.encoded.bin encoder output [D, T'], raw f32 LE
  <clip>.logprobs.bin CTC log-probs [T', C], raw f32 LE (CTC models only)
  <clip>.labels.bin  argmax labels [T'], raw i32 LE (CTC models only)

Run inside the gigaam venv:
  /Users/sergey/trakktor/reference/.venv-gigaam/bin/python \
      scripts/asr/gigaam/dump_reference.py --model v3_ctc \
      --audio tmp/sample.mp3 --offset 40 --duration 12 --out tmp/gigaam/v3_ctc
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import gigaam


def save_f32(path: Path, arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    arr.tofile(path)
    return path.name


def save_i32(path: Path, arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr, dtype="<i4")
    arr.tofile(path)
    return path.name


def dump_config(model, model_name: str, out: Path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(model.cfg, resolve=True)
    dec = model.decoding
    tok = dec.tokenizer
    vocab = list(tok.vocab) if getattr(tok, "charwise", False) else None
    info = {
        "model_name": model_name,
        "charwise": bool(getattr(tok, "charwise", False)),
        "vocab": vocab,
        "vocab_size": len(tok),
        "blank_id": dec.blank_id,
        "cfg": cfg,
    }
    (out / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"vocab_size={len(tok)} blank_id={dec.blank_id} charwise={info['charwise']}")
    return info


def dump_state_dict(model, out: Path):
    sd = model.state_dict()
    shapes = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in sd.items()}
    (out / "state_dict.json").write_text(json.dumps(shapes, indent=2))
    print(f"state_dict: {len(shapes)} tensors")


def dump_mel_fbank(model, out: Path):
    # torchaudio MelSpectrogram: featurizer[0] is MelSpectrogram, .mel_scale.fb
    # is [n_freqs, n_mels]; MelScale.forward does spec.T @ fb. The fb and the
    # STFT window are CHECKPOINT buffers (fp16-quantized) — the port must use
    # these, not freshly computed ones: the quantization is amplified by the log
    # at quiet mel bins.
    mel = model.preprocessor.featurizer[0]
    fb = mel.mel_scale.fb.detach().cpu().numpy()  # [n_freqs, n_mels]
    window = mel.spectrogram.window.detach().cpu().numpy()  # [n_fft]
    save_f32(out / "mel_fbank.bin", fb)
    save_f32(out / "mel_window.bin", window)
    hop = model.preprocessor.hop_length
    win = model.preprocessor.win_length
    n_fft = model.preprocessor.n_fft
    center = model.preprocessor.center
    print(f"mel_fbank {fb.shape} window {window.shape} n_fft={n_fft} hop={hop} win={win} center={center}")
    return {"mel_fbank_shape": list(fb.shape), "n_fft": n_fft, "hop": hop, "win": win}


def dump_clip(model, info, audio: str, offset: float, duration: float, out: Path):
    sr = 16000
    wav_full = gigaam.load_audio(audio)  # [N] f32 in [-1,1)
    a = int(offset * sr)
    b = a + int(duration * sr) if duration > 0 else wav_full.shape[0]
    clip = wav_full[a:b].contiguous()
    tag = f"clip_{int(offset)}_{int(duration)}"
    pcm_name = save_f32(out / f"{tag}.pcm.bin", clip.numpy())

    wav = clip.unsqueeze(0).float()  # [1, n], cpu fp32
    length = torch.tensor([wav.shape[-1]], dtype=torch.long)

    with torch.inference_mode():
        feats, feat_len = model.preprocessor(wav, length)  # [1, n_mels, T]
        encoded, enc_len = model.encoder(feats, feat_len)  # [1, D, T']

    trace = {
        "model_name": info["model_name"],
        "pcm": pcm_name,
        "pcm_samples": int(clip.shape[0]),
        "offset": offset,
        "duration": duration,
        "feats_shape": list(feats.shape),
        "feat_len": int(feat_len[0].item()),
        "encoded_shape": list(encoded.shape),
        "enc_len": int(enc_len[0].item()),
    }
    trace["mel"] = save_f32(out / f"{tag}.mel.bin", feats[0].cpu().numpy())
    trace["encoded"] = save_f32(out / f"{tag}.encoded.bin", encoded[0].cpu().numpy())

    is_ctc = "ctc" in info["model_name"]
    if is_ctc:
        with torch.inference_mode():
            log_probs = model.head(encoded)  # [1, T', C]
            labels = log_probs.argmax(dim=-1)  # [1, T']
        trace["logprobs_shape"] = list(log_probs.shape)
        trace["logprobs"] = save_f32(out / f"{tag}.logprobs.bin", log_probs[0].cpu().numpy())
        trace["labels"] = save_i32(out / f"{tag}.labels.bin", labels[0].cpu().numpy())

    with torch.inference_mode():
        decoded = model.decoding.decode(model.head, encoded, enc_len)
    text, token_ids, token_frames = decoded[0]
    trace["text"] = text
    trace["token_ids"] = list(token_ids)
    trace["token_frames"] = list(token_frames)
    # Word timestamps (frame-based)
    frame_shift = wav.shape[-1] / sr / int(enc_len[0].item())
    trace["frame_shift"] = frame_shift
    from gigaam.timestamps_utils import frames_to_words

    words = frames_to_words(model.decoding.tokenizer, token_ids, token_frames, frame_shift)
    trace["words"] = [{"text": w.text, "start": w.start, "end": w.end} for w in words]

    (out / f"{tag}.trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2))
    print(f"[{tag}] text={text!r}")
    print(f"[{tag}] encoded={trace['encoded_shape']} enc_len={trace['enc_len']} tokens={len(token_ids)}")
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v3_ctc")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--offset", type=float, default=40.0)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    # CPU + fp32 encoder => deterministic, no autocast; our parity target.
    model = gigaam.load_model(args.model, fp16_encoder=False, device="cpu")
    model = model.eval()

    info = dump_config(model, args.model, out)
    dump_state_dict(model, out)
    dump_mel_fbank(model, out)

    if args.audio:
        dump_clip(model, info, args.audio, args.offset, args.duration, out)

    print(f"wrote reference dump to {out}")


if __name__ == "__main__":
    main()
