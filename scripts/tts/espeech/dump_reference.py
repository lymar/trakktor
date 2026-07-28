#!/usr/bin/env python3
"""Dump ESpeech (F5-TTS) reference intermediates as the golden trace for the port.

Runs the reference stack — F5-TTS + Vocos + torchaudio, CPU, fp32 — on one
reference recording and one sentence, and writes every stage the Rust port is
compared against. Because the model is flow matching rather than sampling, the
run is deterministic once the starting noise is fixed, so the port is fed *this*
noise and every stage has to match.

The solver loop here is re-implemented rather than taken from `CFM.sample`, so
that intermediates are observable; `--check-cfm` verifies the re-implementation
against the reference's own sampler and must report a zero difference. That check
is what catches subtle layout mistakes — for instance that the reference splices
one frame more of the reference mel than it cuts.

Writes to an output directory (all raw little-endian, mels frames-major):

  ref_wave.bin          the reference recording as the model saw it, f32
  ref_mel.bin           its log-mel [frames, 100], f32
  text_ids.bin          character ids of reference transcript + sentence, i32
  y0.bin                the starting noise [duration, 100], f32
  t.bin                 the timestep schedule [steps + 1], f32
  v0.bin                the guided flow at the first step [duration, 100], f32
  mel_gen.bin           the generated mel after the cut [frames, 100], f32
  wave.bin              the synthesized waveform, f32
  vocos_roundtrip.bin   the vocoder applied to ref_mel.bin — an isolated stage
  dims.json             frame counts, step count, text length, reference loudness
  ref.wav, ref.txt, test.txt   copies, so the dump is self-contained

Run inside the F5-TTS venv:
  /Users/sergey/trakktor/reference/.venv-f5/bin/python \
      scripts/tts/espeech/dump_reference.py \
      --model ~/.trakktor/tts/espeech/rl-v2 \
      --ref-audio tmp/espeech/ref.wav --ref-text-file tmp/espeech/ref.txt \
      --text-file tmp/espeech/test.txt --nfe 4 --check-cfm \
      --out tmp/espeech/golden4
"""

import argparse
import importlib.abc
import importlib.machinery
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# The F5-TTS checkout whose `src/` is imported; override with F5_TTS_SRC when
# the reference lives elsewhere.
REFERENCE = Path(
    os.environ.get("F5_TTS_SRC", "/Users/sergey/trakktor/reference/F5-TTS/src")
)

TARGET_SR = 24000
N_FFT = 1024
HOP = 256
WIN = 1024
N_MEL = 100
TARGET_RMS = 0.1

# `f5_tts.model`'s package init pulls the trainer, and with it the training
# stack. Stub what only training needs, so inference can be imported.
STUBBED = {"wandb", "accelerate", "ema_pytorch", "datasets", "bitsandbytes"}


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        return type(name, (), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


def install_stubs():
    for name in list(STUBBED):
        try:
            __import__(name)
            STUBBED.discard(name)
        except ImportError:
            pass
    sys.path.insert(0, str(REFERENCE))
    sys.meta_path.insert(0, _StubFinder())


def f5_mel(wave):
    """F5-TTS's vocos-style mel: torchaudio MelSpectrogram(power=1) then log."""
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        win_length=WIN,
        hop_length=HOP,
        n_mels=N_MEL,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    )
    return mel_stft(wave).clamp(min=1e-5).log()


def read_reference(path):
    """The reference recording, loudness-normalized the way inference does.

    soundfile rather than torchaudio.load: recent torchaudio delegates I/O to
    torchcodec. Both hand back int16/32768 as float32, so the samples match.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T).contiguous()
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms
    if sr != TARGET_SR:
        audio = torchaudio.transforms.Resample(sr, TARGET_SR)(audio)
    return audio, float(rms)


def load_vocoder(model_dir):
    """Load Vocos from whichever form of its weights is on disk.

    A normal trakktor install leaves only the converted `model.safetensors`: the
    published `pytorch_model.bin` is deleted once converted. Both hold the same
    tensors under the same names, so either will do — and accepting the converted
    one is what lets this script run without re-downloading anything.
    """
    from vocos import Vocos

    vocoder_dir = Path(model_dir).parent / "vocos-mel-24khz"
    config = vocoder_dir / "config.yaml"
    if not config.is_file():
        raise SystemExit(f"no {config} — fetch it from charactr/vocos-mel-24khz")

    published = vocoder_dir / "pytorch_model.bin"
    converted = vocoder_dir / "model.safetensors"
    if published.is_file():
        weights = torch.load(published, map_location="cpu", weights_only=True)
    elif converted.is_file():
        from safetensors.torch import load_file

        weights = load_file(str(converted))
    else:
        raise SystemExit(
            f"no vocoder weights in {vocoder_dir} (neither {published.name} nor "
            f"{converted.name})"
        )

    vocoder = Vocos.from_hparams(str(config))
    vocoder.load_state_dict(weights, strict=False)
    return vocoder.eval()


def load_model(model_dir):
    from f5_tts.model.backbones.dit import DiT
    from f5_tts.model.cfm import CFM
    from f5_tts.model.utils import get_tokenizer
    from safetensors.torch import load_file

    vocab_map, vocab_size = get_tokenizer(
        str(Path(model_dir) / "vocab.txt"), "custom"
    )
    model = CFM(
        transformer=DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            text_num_embeds=vocab_size,
            mel_dim=N_MEL,
        ),
        mel_spec_kwargs=dict(
            n_fft=N_FFT,
            hop_length=HOP,
            win_length=WIN,
            n_mel_channels=N_MEL,
            target_sample_rate=TARGET_SR,
            mel_spec_type="vocos",
        ),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_map,
    )
    weights = load_file(str(Path(model_dir) / "model.safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [name for name in missing if not name.startswith("mel_spec.")]
    if missing or unexpected:
        raise SystemExit(f"weights do not fit: missing {missing}, extra {unexpected}")
    return model.eval(), vocab_map


def close_reference_text(text):
    """Close the transcript exactly as inference does — ending in two spaces.

    The reference appends ". " (or just the space, if it already ends in a
    period) in `preprocess_ref_audio_text`, then appends another space in
    `infer_batch_process` because the last character is single-byte.
    """
    if not text.endswith(". ") and not text.endswith("。"):
        text = text + " " if text.endswith(".") else text + ". "
    if len(text[-1].encode("utf-8")) == 1:
        text += " "
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="converted checkpoint directory")
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text-file", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--check-cfm",
        action="store_true",
        help="verify the local solver loop against the reference's CFM.sample",
    )
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    install_stubs()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, array, dtype="<f4"):
        array = np.ascontiguousarray(array.astype(dtype))
        array.tofile(out / name)
        print(f"  {name}: {array.shape} {array.dtype}")

    from f5_tts.model.utils import convert_char_to_pinyin, get_epss_timesteps

    vocoder = load_vocoder(args.model)
    model, vocab_map = load_model(args.model)

    audio, ref_rms = read_reference(args.ref_audio)
    ref_mel = f5_mel(audio)  # [1, 100, frames]
    write("ref_wave.bin", audio[0].numpy())
    write("ref_mel.bin", ref_mel[0].numpy().T)
    write("vocos_roundtrip.bin", vocoder.decode(ref_mel)[0].numpy())

    ref_text = close_reference_text(Path(args.ref_text_file).read_text().strip())
    gen_text = Path(args.text_file).read_text()
    chars = convert_char_to_pinyin([ref_text + gen_text])
    ids = torch.tensor(
        [[vocab_map.get(c, 0) for c in chars[0]]], dtype=torch.long
    )
    write("text_ids.bin", ids[0].numpy(), "<i4")

    # The reference counts the recording in whole hops here, one frame fewer
    # than its mel has; the generated audio is cut at this same frame.
    cut_frames = audio.shape[-1] // HOP
    cond_frames = ref_mel.shape[-1]
    speed = args.speed if len(gen_text.encode("utf-8")) >= 10 else 0.3
    duration = cut_frames + int(
        cut_frames / len(ref_text.encode("utf-8"))
        * len(gen_text.encode("utf-8"))
        / speed
    )

    cond = ref_mel.permute(0, 2, 1)
    cond_pad = torch.nn.functional.pad(
        cond, (0, 0, 0, duration - cond.shape[1]), value=0.0
    )

    torch.manual_seed(args.seed)
    y0 = torch.randn(duration, N_MEL, dtype=cond.dtype).unsqueeze(0)
    write("y0.bin", y0[0].numpy())

    if args.nfe in (5, 6, 7, 10, 12, 16):
        t = get_epss_timesteps(args.nfe, device="cpu", dtype=cond.dtype)
    else:
        t = torch.linspace(0, 1, args.nfe + 1, dtype=cond.dtype)
    t = t + args.sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
    write("t.bin", t.numpy())

    x = y0.clone()
    first_flow = None
    for step in range(args.nfe):
        both = model.transformer(
            x=x,
            cond=cond_pad,
            text=ids,
            time=t[step],
            mask=None,
            cfg_infer=True,
            cache=True,
        )
        conditional, unconditional = torch.chunk(both, 2, dim=0)
        flow = conditional + (conditional - unconditional) * args.cfg
        if step == 0:
            first_flow = flow[0].numpy().copy()
        x = x + (t[step + 1] - t[step]) * flow
    model.transformer.clear_cache()
    write("v0.bin", first_flow)

    # The conditioning is spliced back over its own frames — all of them, one
    # more than the cut below takes away.
    mask = torch.zeros(1, duration, 1, dtype=torch.bool)
    mask[:, :cond_frames] = True
    full = torch.where(mask, cond_pad, x)
    generated = full[:, cut_frames:, :]
    write("mel_gen.bin", generated[0].numpy())

    wave = vocoder.decode(generated.permute(0, 2, 1))
    if ref_rms < TARGET_RMS:
        wave = wave * ref_rms / TARGET_RMS
    write("wave.bin", wave[0].numpy())

    (out / "dims.json").write_text(
        json.dumps(
            {
                "ref_frames": cut_frames,
                "duration": duration,
                "nfe": args.nfe,
                "text_len": len(chars[0]),
                "cond_frames": cond_frames,
                "ref_rms": ref_rms,
            },
            indent=1,
        )
    )
    for source, name in (
        (args.ref_audio, "ref.wav"),
        (args.ref_text_file, "ref.txt"),
        (args.text_file, "test.txt"),
    ):
        shutil.copyfile(source, out / name)
    sf.write(out / "reference.wav", wave[0].numpy(), TARGET_SR)

    if args.check_cfm:
        torch.manual_seed(args.seed)
        sampled, _ = model.sample(
            cond=cond,
            text=chars,
            duration=duration,
            steps=args.nfe,
            cfg_strength=args.cfg,
            sway_sampling_coef=args.sway,
            seed=args.seed,
        )
        difference = float((sampled - full).abs().max())
        print(f"CFM.sample vs the local loop: max abs difference {difference}")
        if difference != 0.0:
            raise SystemExit(
                "the local solver loop no longer matches the reference sampler; "
                "the dump would not be a golden trace"
            )

    print(f"wrote reference dump to {out}")


if __name__ == "__main__":
    main()
