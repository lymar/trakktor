#!/usr/bin/env python3
"""Write miniature checkpoints in the published layout, for the converter tests.

The engine's download path extracts the moving average out of a training
checkpoint and refuses anything it does not recognize. Exercising that on the
real 2.7 GB archive costs an hour of bandwidth, so the tests run against these
three miniatures instead:

  mini.pt        the normal shape: the net twice (live + moving average), plus
                 the average's `initted`/`step` bookkeeping, all f32
  mini_f16.pt    the average stored in half precision — must be refused, since
                 converting it would produce something that loads and sounds wrong
  mini_noema.pt  no moving average at all — must be refused, since that is not
                 the half inference uses

Run inside the F5-TTS venv (any venv with torch will do):
  /Users/sergey/trakktor/reference/.venv-f5/bin/python \
      scripts/tts/espeech/gen_convert_fixtures.py --out tmp/espeech/fixture
"""

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Values are a small ramp so a test can assert it read the average (twice
    # the live weights) rather than the live weights.
    base = {
        "transformer.proj_out.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "transformer.proj_out.bias": torch.zeros(2),
    }
    torch.save(
        {
            "model_state_dict": dict(base),
            "ema_model_state_dict": {
                "initted": torch.tensor(True),
                "step": torch.tensor(84000),
                **{f"ema_model.{name}": value * 2.0 for name, value in base.items()},
            },
        },
        out / "mini.pt",
    )
    torch.save(
        {
            "ema_model_state_dict": {
                "ema_model.transformer.proj_out.weight": base[
                    "transformer.proj_out.weight"
                ].half()
            }
        },
        out / "mini_f16.pt",
    )
    torch.save({"model_state_dict": dict(base)}, out / "mini_noema.pt")

    for name in ("mini.pt", "mini_f16.pt", "mini_noema.pt"):
        print(f"  {name}: {(out / name).stat().st_size} B")
    print(f"wrote converter fixtures to {out}")


if __name__ == "__main__":
    main()
