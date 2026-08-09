"""Measure a reading against the page it was made from.

The corpus is the same typeset page photographed several ways, so the question
it exists to answer is arithmetic: how much *more* wrong is the reading of a
photograph than the reading of the flat render of the same PDF. The number is
the character error rate — Levenshtein distance over the length of the truth —
and the flat render's own rate is the floor everything else is read against.

Both sides are normalized the same way before the distance is taken:
whitespace collapses, and the punctuation a fixed-alphabet recognizer cannot
emit is folded to its ASCII neighbour. Neither is a favour to the engine —
both sides get it — and both remove a constant that has nothing to do with the
geometry being measured.

    python measure.py --truth <ground-truth.txt> --reading <reading.txt>
    python measure.py --truth <t.txt> --dir <readings/> --json
"""

import argparse
import json
import pathlib
import re
import sys
import unicodedata

FOLD = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "−": "-",
    "•": "-",
    " ": " ",
    "ﬁ": "fi",
    "ﬂ": "fl",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for source, target in FOLD.items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def distance(a: str, b: str) -> int:
    """Levenshtein distance, two rows at a time."""
    if a == b:
        return 0
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def rate(truth: str, reading: str) -> dict:
    truth, reading = normalize(truth), normalize(reading)
    characters = distance(truth, reading)
    words = distance(truth.split(), reading.split())
    return {
        "cer": characters / max(len(truth), 1),
        "wer": words / max(len(truth.split()), 1),
        "truth_chars": len(truth),
        "read_chars": len(reading),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--reading")
    parser.add_argument("--dir", help="measure every .txt in this directory")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    truth = pathlib.Path(args.truth).read_text()
    if args.reading:
        result = rate(truth, pathlib.Path(args.reading).read_text())
        print(json.dumps(result, indent=2) if args.json else
              f"CER {result['cer']:.4f}  WER {result['wer']:.4f}")
        return

    if not args.dir:
        sys.exit("one of --reading or --dir is required")

    results = {}
    for path in sorted(pathlib.Path(args.dir).glob("*.txt")):
        results[path.stem] = rate(truth, path.read_text())
    if args.json:
        print(json.dumps(results, indent=2))
        return
    width = max(len(name) for name in results) if results else 0
    for name, result in sorted(results.items(), key=lambda kv: kv[1]["cer"]):
        print(
            f"{name:<{width}}  CER {result['cer']:7.4f}  "
            f"WER {result['wer']:7.4f}  read {result['read_chars']:5d}"
            f" of {result['truth_chars']}"
        )


if __name__ == "__main__":
    main()
