# `asr whisper` — Whisper engine

> One of the engines of [trakktor's `asr` command](README.md); the behavior
> shared by all engines — audio input, model storage, device and precision,
> runtime, timestamps, output files — is documented there.

[Whisper](https://github.com/openai/whisper) is the most versatile engine:
many languages, language autodetection, and optional translation into English.
Transcription runs window by window (30 seconds each) with a
temperature-fallback policy that guards against repetition loops, closely
following Whisper's own decoding behavior.

## Models

```
--model <name|dir>      # default: tiny
```

Pass a **published name** — downloaded on first use into the model directory
(`~/.trakktor/asr/whisper/<name>/` by default; see
[Model storage](README.md#model-storage)) — or a **path** to a local checkpoint
directory (one containing `config.json`).

Names, smallest to largest (larger is slower but more accurate):

- **Multilingual:** `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`,
  `large-v3` (alias `large`), `large-v3-turbo` (alias `turbo`).
- **English-only:** `tiny.en`, `base.en`, `small.en`, `medium.en` — slightly
  better on English audio.

## Language and task

```
--language <code|name>        # e.g. en, ru, or russian; autodetected if omitted
--task transcribe|translate   # default: transcribe
```

With no `--language`, the language is detected from the first 30 seconds.
`--task translate` renders the speech as English instead of transcribing it in
the source language.

## Clipping: transcribe only part of the audio

```
--start <time>      # begin at this offset (default: the beginning)
--end <time>        # stop at this offset (default: the end)
```

`--start`/`--end` limit transcription to a time range. Each accepts a plain
number of seconds or a `[[HH:]MM:]SS[.mmm]` clock, to millisecond precision, and
either flag works on its own:

```sh
trakktor asr whisper talk.mp3 --start 0:15 --end 5:30   # from 0:15 to 5:30
trakktor asr whisper talk.mp3 --start 90                # from 1:30 to the end
trakktor asr whisper talk.mp3 --end 1:02:03.250         # from the start to 1:02:03.250
```

They are a convenience over `--clip-timestamps` and cannot be combined with it;
use `--clip-timestamps` directly to transcribe several ranges at once.

## Voice-activity detection (VAD)

`--vad` runs Silero voice-activity detection first and transcribes only the
speech, dropping silence, music, and noise. It is the direct remedy for
Whisper's tendency to hallucinate and loop over long non-speech stretches, and
it is faster on sparse audio. Off by default; the model is built in (nothing to
download) and runs on the CPU, so `--device metal` still accelerates the
transcription itself.

```sh
trakktor asr whisper interview.mp3 --vad
```

The detected speech is glued into one dense buffer, transcribed in a single
pass, and the timestamps are then mapped back to the original timeline — so the
output stays on the source clock while non-speech is never sent to the model.

Detection is tunable (shown with the reference defaults):

```
--vad-threshold 0.5                 # speech-probability cutoff (0..=1)
--vad-min-speech-duration-ms 250    # drop shorter detections
--vad-min-silence-duration-ms 100   # a shorter pause won't split a segment
--vad-speech-pad-ms 30              # padding kept around each speech span
--vad-max-speech-duration-s none    # force-split longer speech (none = never)
```

`--vad` combines with `--start`/`--end` (detect speech within that range) but
not with `--clip-timestamps`. When VAD is active the JSON result gains a
top-level `vad` block listing the detected speech spans (original timeline):

```json
"vad": { "speech": [ { "start": 0.48, "end": 12.3 } ] }
```

## Decoding controls

The decoding defaults mirror the reference behavior and rarely need touching;
every flag below has a sensible default. Run `trakktor asr whisper --help` for
the complete list with defaults and exact value formats. In brief:

- **Temperature fallback** — `--temperature`,
  `--temperature-increment-on-fallback`: the schedule the decoder climbs when a
  window looks like a failure.
- **Sampling and search** — `--best-of` (trajectories at non-zero temperature),
  `--beam-size` (beam width at temperature 0), `--patience`, `--length-penalty`.
- **Failure gates** (each triggers a hotter retry) —
  `--compression-ratio-threshold` (repetition), `--logprob-threshold`
  (confidence), `--no-speech-threshold` (silence).
- **Prompting** — `--initial-prompt` (bias the first window toward domain
  vocabulary or proper nouns), `--carry-initial-prompt`,
  `--condition-on-previous-text`.
- **Token suppression** — `--suppress-tokens` (`-1` expands to a built-in
  non-speech set).
- **Word-timestamp tuning** (with `--timestamps word`) —
  `--prepend-punctuations`, `--append-punctuations`,
  `--hallucination-silence-threshold`.
- **Partial audio** — `--clip-timestamps` to transcribe only selected
  `start,end` second ranges. For a single range, the `--start`/`--end` flags
  above are usually easier.
- **Voice-activity detection** — `--vad` (with the `--vad-*` tuning flags)
  detects speech and skips non-speech before transcribing; see the VAD section
  above.

## Examples

```sh
# Russian interview, best model on the GPU, word-level timings, indented JSON
trakktor asr whisper interview.m4a \
  --model large-v3 --device metal --language ru \
  --timestamps word --pretty

# Translate a lecture to English, readable text output
trakktor asr whisper lecture.mp3 --model medium --task translate --text

# Bias the first window with domain terms
trakktor asr whisper standup.wav \
  --initial-prompt "Kubernetes, Grafana, Prometheus, sharding"

# Subtitle one section: transcribe 0:15–5:30 and write every format into subs/
trakktor asr whisper talk.mp3 --start 0:15 --end 5:30 \
  --output-format all --output-dir subs

# Decode a format the built-in decoder does not cover
trakktor asr whisper voice.opus --audio-decoder ffmpeg --model small
```
