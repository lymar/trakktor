#!/usr/bin/env python3
"""Generate tokenizer golden fixtures from the reference Whisper package.

This is a developer tool, not part of the shipped binary. Run it with an
interpreter that has the reference Whisper package installed. It dumps, for a
set of tokenizer configurations, everything the Rust port must reproduce:
special-token ids, start sequences, the non-speech suppress set, encodings of
sample texts, decodings of a synthetic token stream (with and without
timestamps), and word-splitting results. The Rust tests replay the fixture
hermetically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from whisper.tokenizer import get_tokenizer

SAMPLE_TEXTS = [
    "",
    " ",
    " Hello, world!",
    "Hello world",
    " Привет, мир! Это тест распознавания речи.",
    "こんにちは世界。",
    " 42 apples cost $3.14, right?",
    " -- - ' '' «» 「」 (( ))",
    " I'm sure we can't won't o'clock",
    "🙂 emoji test 🎵",
]

SPLIT_TEXTS = {
    None: " Hello there, how's it going? (Good, thanks!)",
    "en": " Hello there, how's it going? (Good, thanks!)",
    "ru": " Привет, как дела? Хорошо, спасибо!",
    "zh": "这是一个中文测试。",
}


def dump_case(name, multilingual, num_languages, language, task):
    tok = get_tokenizer(
        multilingual, num_languages=num_languages, language=language, task=task
    )

    ts = tok.timestamp_begin
    stream = (
        list(tok.sot_sequence)
        + [ts]
        + tok.encoding.encode(" Hello, world!")
        + [ts + 54, ts + 54]
        + tok.encoding.encode(" Again.")
        + [ts + 150, tok.eot]
    )

    split_text = SPLIT_TEXTS.get(tok.language, SPLIT_TEXTS["en"])
    split_tokens = tok.encoding.encode(split_text)
    words, word_tokens = tok.split_to_word_tokens(list(split_tokens))

    return {
        "name": name,
        "multilingual": multilingual,
        "num_languages": num_languages,
        "language": language,
        "task": task,
        "normalized_language": tok.language,
        "specials": {
            "eot": tok.eot,
            "sot": tok.sot,
            "translate": tok.translate,
            "transcribe": tok.transcribe,
            "sot_lm": tok.sot_lm,
            "sot_prev": tok.sot_prev,
            "no_speech": tok.no_speech,
            "no_timestamps": tok.no_timestamps,
            "timestamp_begin": tok.timestamp_begin,
        },
        "n_vocab": tok.encoding.n_vocab,
        "sot_sequence": list(tok.sot_sequence),
        "sot_sequence_including_notimestamps": list(
            tok.sot_sequence_including_notimestamps
        ),
        # The reference iterates a Python set here, so only the *set* of ids is
        # stable across runs; dump it sorted. (The port returns vocabulary
        # order, which is the same list once sorted.)
        "all_language_tokens_sorted": sorted(tok.all_language_tokens),
        "non_speech_tokens": list(tok.non_speech_tokens),
        "encode": {text: tok.encoding.encode(text) for text in SAMPLE_TEXTS},
        "decode_stream_ids": stream,
        "decode_stream_plain": tok.decode(stream),
        "decode_stream_with_timestamps": tok.decode_with_timestamps(stream),
        "split": {
            "text": split_text,
            "tokens": split_tokens,
            "words": words,
            "word_tokens": word_tokens,
        },
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    out = (
        repo_root
        / "trakktor_core/src/asr/whisper/testdata/tokenizer_golden.json"
    )

    cases = [
        dump_case("gpt2", False, 99, None, None),
        dump_case("multilingual_99_en_transcribe", True, 99, "en", "transcribe"),
        dump_case("multilingual_99_en_translate", True, 99, "en", "translate"),
        dump_case("multilingual_100_ru_transcribe", True, 100, "ru", "transcribe"),
        dump_case("multilingual_100_zh_transcribe", True, 100, "zh", "transcribe"),
        dump_case("multilingual_99_alias_mandarin", True, 99, "Mandarin", "transcribe"),
    ]

    out.write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
