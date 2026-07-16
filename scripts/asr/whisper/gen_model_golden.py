#!/usr/bin/env python3
"""Generate runtime parity goldens from the reference Whisper implementation.

This is a developer tool, not part of the shipped binary. Run it with an
interpreter that has the reference Whisper package (and torch) installed. It
loads a reference model on the CPU in full precision, runs the committed PCM
fixture through the reference feature extraction and network, and dumps:

- the encoder output for one 30 s window, and
- the decoder logits for the start sequence (sot, language, task),

as raw little-endian f32 plus a small JSON manifest. The Rust runtime's
opt-in parity tests compare against these dumps. The outputs land in the
developer-local `tmp/` tree (they are large and machine-generated, so they
are not committed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import whisper
from whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
from whisper.tokenizer import get_tokenizer


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tiny", help="reference model name")
    ap.add_argument(
        "--pcm",
        default=str(
            repo_root
            / "trakktor_core/src/asr/whisper/testdata/sample_2s.pcm.bin"
        ),
        help="path to the f32 PCM fixture",
    )
    ap.add_argument(
        "--out", default=str(repo_root / "tmp/whisper_golden")
    )
    args = ap.parse_args()

    pcm = np.fromfile(args.pcm, dtype="<f4")
    audio = pad_or_trim(torch.from_numpy(pcm), N_SAMPLES)

    model = whisper.load_model(args.model, device="cpu")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="en",
        task="transcribe",
    )
    tokens = list(tokenizer.sot_sequence)

    mel = log_mel_spectrogram(audio, model.dims.n_mels)
    with torch.no_grad():
        features = model.encoder(mel.unsqueeze(0))
        logits = model.logits(torch.tensor([tokens]), features)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The exact reference mel input, so the encoder can be compared in
    # isolation from feature-extraction differences.
    mel_path = out / f"{args.model}_mel.bin"
    mel.numpy().astype("<f4").tofile(mel_path)

    encoder_path = out / f"{args.model}_encoder.bin"
    features.squeeze(0).numpy().astype("<f4").tofile(encoder_path)
    logits_path = out / f"{args.model}_logits_sot.bin"
    logits.squeeze(0).numpy().astype("<f4").tofile(logits_path)

    manifest = {
        "model": args.model,
        "sot_sequence": tokens,
        "encoder_shape": list(features.squeeze(0).shape),
        "logits_shape": list(logits.squeeze(0).shape),
    }
    (out / f"{args.model}_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {encoder_path} {list(features.shape)}")
    print(f"wrote {logits_path} {list(logits.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
