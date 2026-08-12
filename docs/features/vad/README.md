# `vad` — voice-activity audio editing

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

`trakktor vad` finds the speech in an audio file with Silero voice-activity
detection and acts on it — report where the speech is, cut the silence out, or
split a recording into clips. It stands beside [`asr`](../asr/README.md) (this
is editing, not
transcription): the same detector, used to reshape audio rather than feed a
transcription engine.

The key idea: **detection runs on 16 kHz mono (what Silero needs), but cutting
runs on the decoded original at full quality** — its own sample rate, channels,
and bit depth. Nothing is downsampled to the detector's 16 kHz; the speech times
are mapped back to the original samples, and only a few milliseconds of fade at
each join are altered. Output is a self-contained pure-Rust WAV (default) or FLAC
— no ffmpeg required.

The input is any format the built-in decoder reads — mp3, aac (LC), vorbis,
flac, alac, adpcm, and PCM audio in wav/aiff/caf/ogg/mp4/mkv containers; it is
the decoder described under [`asr` audio input](../asr/README.md#audio-input).

```sh
trakktor vad timeline talk.mp3     # JSON: where the speech is, plus stats
trakktor vad cut talk.mp3          # write talk.speech.wav with the silence removed
trakktor vad split talk.mp3        # write one file per utterance
```

## timeline — report the speech, write nothing

```sh
trakktor vad timeline interview.mp3
```

Prints a JSON object: the `duration` — of the audio, or of the working window
when `--start`/`--end` narrow it, in which case a `window` object with the
bounds appears as well — a `speech` array of `{start, end}` intervals
(seconds, original timeline), and a `stats` object: `segments`,
`speech_seconds`, `silence_seconds`, `speech_ratio`, and
`longest_pause_seconds` (leading and trailing silence count as pauses). Safe
and side-effect-free. `--text` prints one tab-separated `start`⇥`end` pair per
line.

```json
{
  "duration": 60.0,
  "speech": [
    { "start": 0.5, "end": 20.0 },
    { "start": 24.0, "end": 49.5 }
  ],
  "stats": {
    "segments": 2,
    "speech_seconds": 45.0,
    "silence_seconds": 15.0,
    "speech_ratio": 0.75,
    "longest_pause_seconds": 10.5
  }
}
```

## cut — remove (or keep) the silence

```sh
trakktor vad cut interview.mp3                     # → interview.speech.wav
trakktor vad cut interview.mp3 --format flac       # smaller, lossless (integer sources)
trakktor vad cut interview.mp3 --keep non-speech   # invert: keep the non-speech
trakktor vad cut podcast.wav --max-silence-ms 400  # shorten long pauses, don't drop them
```

Concatenates the detected speech into one file, dropping non-speech, with a short
fade at each join so edits do not click. `--keep non-speech` inverts it (keep the
silence/music, drop the speech). `--max-silence-ms` collapses long pauses to a
fixed length instead of removing them, keeping the natural rhythm — it applies
with the default `--keep speech` only. The file lands in `--output-dir`
(default: the current directory, shared with `split`), named
`<stem>.<speech|non-speech>.<ext>` after the side kept — `interview.speech.wav`
above, `interview.non-speech.wav` with `--keep non-speech`. The written
path is reported on stderr; stdout carries a JSON summary — `output` (omitted
when nothing was written), `format`, `kept`, `segments`, `source_duration`,
`output_duration`:

```json
{
  "output": "./interview.speech.wav",
  "format": "wav",
  "kept": "speech",
  "segments": 2,
  "source_duration": 60.0,
  "output_duration": 45.2
}
```

## split — one clip per utterance

```sh
trakktor vad split lecture.mp3 --output-dir clips --min-duration-ms 500
```

Writes one file per detected span into `--output-dir` (created if missing),
named as in `cut` with a clip number added —
`<stem>.<speech|non-speech>.<NNN>.<ext>`: `lecture.speech.001.wav`,
`.002.wav`, … — dropping any shorter than `--min-duration-ms` (default 0: keep
everything). `--keep non-speech` splits out the non-speech spans instead. The
JSON result lists each clip with its `path`, `index`, source `start`/`end`,
and `duration`, inside an envelope of `output_dir`, `format`, `kept`, and
`count`:

```json
{
  "output_dir": "clips",
  "format": "wav",
  "kept": "speech",
  "count": 2,
  "clips": [
    {
      "path": "clips/lecture.speech.001.wav",
      "index": 1,
      "start": 0.5,
      "end": 20.0,
      "duration": 19.5
    },
    {
      "path": "clips/lecture.speech.002.wav",
      "index": 2,
      "start": 24.0,
      "end": 49.5,
      "duration": 25.5
    }
  ]
}
```

## Detection tuning and presets

Detection is the canonical Silero algorithm, and its unit is a **speech
probability** (not decibels). `--preset` sets a base that the individual flags
override:

```
--preset tight|asr|natural    # default: tight
```

- **`tight`** (default) — aggressive silence removal (canonical Silero).
- **`asr`** — bridge long pauses, keeping speech in large chunks (a
  transcription front-end).
- **`natural`** — keep more breathing room around speech.

The numbers each preset sets — any flag you pass overrides its column:

| | threshold | min-speech | min-silence | pad | margin\* | merge-gap\* | fade\* |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tight` | 0.5 | 250 ms | 100 ms | 30 ms | 50 ms | 150 ms | 10 ms |
| `asr` | 0.5 | 0 | 2000 ms | 400 ms | 0 | 200 ms | 10 ms |
| `natural` | 0.5 | 250 ms | 100 ms | 30 ms | 200 ms | 400 ms | 15 ms |

\* margin, merge-gap, and fade shape the edit, so they exist on `cut` and
`split` only — passing `--margin-ms`, `--merge-gap-ms`, or `--fade-ms` to
`timeline` is a usage error.

Override any piece directly with `--threshold`, `--min-speech-duration-ms`,
`--min-silence-duration-ms`, `--speech-pad-ms`, `--max-speech-duration-s`. Shape
the edit with `--fade-ms`, `--merge-gap-ms` (merge close spans), and `--margin-ms`
(extra audio kept around each span). `--start`/`--end` (same time format as
[`asr`](../asr/whisper.md#clipping-transcribe-only-part-of-the-audio)) define a
**working window**: cutting, splitting, and inversion all happen
within `[start, end]`, so `--keep non-speech --start 1:00 --end 2:00` yields just
the silence inside that minute.

## Output format

```
--format wav|flac                      # default: wav (built-in encoder)
--audio-encoder auto|builtin|ffmpeg    # default: builtin
--bitrate <rate>                       # e.g. 192k, for lossy ffmpeg formats
```

`--audio-encoder auto` picks by `--format`: the built-in encoder for wav and
flac, ffmpeg for everything else.

With the built-in pure-Rust encoder (the default), `--format wav` reproduces any
source exactly — including the float PCM that mp3/aac/vorbis decode to — so it is
the safe choice for preserving quality; `--format flac` is lossless and about
half the size, but integer-only (up to 24-bit), so a float or >24-bit source is
rejected with a message pointing back at WAV. A third rejection guards 24-bit
material that does not compress: on loud, noise-like audio the encoder can
emit far more bytes than the raw PCM, and the run fails with the same
write-WAV-instead advice rather than leaving an absurdly large file.

For anything else, `--audio-encoder ffmpeg` shells out to an installed `ffmpeg`,
and `--format` then accepts any format it writes by extension — mp3, aac, m4a,
opus, ogg, and more:

```sh
trakktor vad cut talk.mp3 --audio-encoder ffmpeg --format mp3 --bitrate 192k
trakktor vad cut talk.mp3 --audio-encoder ffmpeg --format opus
```

The cut PCM is streamed to ffmpeg and re-encoded (not stream-copied), so cuts
stay sample-accurate for any target; `--bitrate` sets ffmpeg's `-b:a` for lossy
formats. It needs `ffmpeg` on your `PATH` — the built-in path never does.

`--audio-encoder` applies to the **output** only: the input is always decoded
by the built-in decoder — `vad` has no `--audio-decoder`, unlike
[`asr`](../asr/README.md#audio-input) — so `vad` can write opus but cannot
read it.
