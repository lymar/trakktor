#!/usr/bin/env python3
"""Generate window-decoding parity goldens from the reference implementation.

This is a developer tool, not part of the shipped binary. Run it with an
interpreter that has the reference Whisper package installed. It decodes one
30 s window (the committed PCM fixture, padded) with the reference
implementation on the CPU in full precision, under a greedy and a beam-search
configuration, and also detects the window's language. The Rust decoding
tests replay the same window through the runtime and compare tokens exactly
and diagnostics approximately. Outputs land in the developer-local `tmp/`
tree.
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
from whisper.decoding import DecodingOptions


def result_to_dict(result) -> dict:
    return {
        "language": result.language,
        "tokens": list(result.tokens),
        "text": result.text,
        "avg_logprob": float(result.avg_logprob),
        "no_speech_prob": float(result.no_speech_prob),
        "compression_ratio": float(result.compression_ratio),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tiny")
    ap.add_argument(
        "--pcm",
        default=str(
            repo_root
            / "trakktor_core/src/asr/whisper/testdata/sample_2s.pcm.bin"
        ),
    )
    ap.add_argument("--out", default=str(repo_root / "tmp/whisper_golden"))
    args = ap.parse_args()

    pcm = np.fromfile(args.pcm, dtype="<f4")
    audio = pad_or_trim(torch.from_numpy(pcm), N_SAMPLES)

    model = whisper.load_model(args.model, device="cpu")
    mel = log_mel_spectrogram(audio, model.dims.n_mels)

    common = {"language": "en", "task": "transcribe", "fp16": False}
    greedy = whisper.decode(
        model, mel, DecodingOptions(temperature=0.0, **common)
    )
    beam = whisper.decode(
        model, mel, DecodingOptions(temperature=0.0, beam_size=5, **common)
    )

    lang_tokens, lang_probs = model.detect_language(
        mel.unsqueeze(0).to(torch.float32)
    )
    top_language = max(lang_probs[0], key=lang_probs[0].get)

    golden = {
        "model": args.model,
        "greedy": result_to_dict(greedy),
        "beam5": result_to_dict(beam),
        "detected_language": top_language,
        "detected_language_prob": float(lang_probs[0][top_language]),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model}_decode.json"
    path.write_text(
        json.dumps(golden, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
    print("greedy:", json.dumps(result_to_dict(greedy), ensure_ascii=False))
    print("beam5 :", json.dumps(result_to_dict(beam), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
