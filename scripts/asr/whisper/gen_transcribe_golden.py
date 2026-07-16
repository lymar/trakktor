#!/usr/bin/env python3
"""Generate a transcription decision trace from the reference implementation.

This is a developer tool, not part of the shipped binary. Run it with an
interpreter that has the reference Whisper package installed. It transcribes a
slice of a local audio file on the CPU in full precision, recording every
decode call (the temperature-fallback trajectory), and dumps the calls, the
resulting segments, and the final text. The Rust transcription tests replay
the same slice and must reproduce the same decisions.

The trace is fully comparable only while every window is accepted at
temperature zero (beam search is deterministic; higher temperatures sample).
The dump records the accepted temperatures so the comparing side knows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import whisper
from whisper.decoding import DecodingOptions, DecodingResult


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tiny")
    ap.add_argument(
        "--sample", default=str(repo_root / "tmp/sample.mp3")
    )
    ap.add_argument("--offset", type=int, default=0, help="slice start, s")
    ap.add_argument("--duration", type=int, default=120, help="slice length, s")
    ap.add_argument("--out", default=str(repo_root / "tmp/whisper_golden"))
    args = ap.parse_args()

    sample_rate = whisper.audio.SAMPLE_RATE
    audio = whisper.load_audio(args.sample)
    audio = audio[args.offset * sample_rate : (args.offset + args.duration) * sample_rate]

    model = whisper.load_model(args.model, device="cpu")

    # Record every decode call: the temperature-fallback trajectory.
    calls: list[dict] = []
    original_decode = whisper.model.Whisper.decode

    def recording_decode(self, mel, options=DecodingOptions(), **kwargs):
        result: DecodingResult = original_decode(self, mel, options, **kwargs)
        calls.append(
            {
                "temperature": float(options.temperature),
                "tokens": list(result.tokens),
                "text": result.text,
                "avg_logprob": float(result.avg_logprob),
                "no_speech_prob": float(result.no_speech_prob),
                "compression_ratio": float(result.compression_ratio),
            }
        )
        return result

    whisper.model.Whisper.decode = recording_decode
    try:
        result = whisper.transcribe(
            model,
            audio,
            language=None,
            task="transcribe",
            beam_size=5,
            best_of=5,
            fp16=False,
            verbose=None,
        )
    finally:
        whisper.model.Whisper.decode = original_decode

    segments = [
        {
            "id": s["id"],
            "seek": s["seek"],
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "tokens": s["tokens"],
            "temperature": s["temperature"],
            "avg_logprob": s["avg_logprob"],
            "compression_ratio": s["compression_ratio"],
            "no_speech_prob": s["no_speech_prob"],
        }
        for s in result["segments"]
    ]

    temperatures = sorted({c["temperature"] for c in calls})
    golden = {
        "model": args.model,
        "offset": args.offset,
        "duration": args.duration,
        "language": result["language"],
        "text": result["text"],
        "segments": segments,
        "calls": calls,
        "deterministic": all(t == 0.0 for t in temperatures),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model}_transcribe.json"
    path.write_text(
        json.dumps(golden, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
    print(f"language: {result['language']}")
    print(f"decode calls: {len(calls)}, temperatures used: {temperatures}")
    print(f"segments: {len(segments)}")
    print(f"text: {result['text'][:200]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
