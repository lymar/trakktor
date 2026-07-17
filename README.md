# trakktor

`trakktor` is a Rust command-line utility that exposes helper functions for
coding agents (Claude Code, OpenCode, etc.). It is built to be **predictable and
automation-friendly**: stable commands and flags, machine-readable output, and
meaningful exit codes.

## Install

Requires a stable Rust toolchain **≥ 1.88** (edition 2024; nightly is only
needed for `cargo fmt`, not to build). The binary installs into `~/.cargo/bin`,
which must be on your `PATH`.

From git:

```sh
cargo install --locked --git https://github.com/lymar/trakktor.git \
  --branch agent-cli trakktor
```

- `trakktor` (the trailing word) is the package to install — the workspace root
  is virtual, so it must be named.
- `--branch agent-cli` is required for now: the code is not yet on the default
  `trunk` branch. Drop the flag once it lands there.
- `--locked` builds with the exact dependency versions pinned in `Cargo.lock`
  (reproducible); omit it to pull the latest semver-compatible versions.

From a local clone (installs whatever is checked out):

```sh
git clone -b agent-cli https://github.com/lymar/trakktor.git
cd trakktor
cargo install --locked --path trakktor
```

For GPU-accelerated speech recognition on macOS, add `--features metal` to
either install command; see the `asr` section below.

Verify, then remove if needed:

```sh
trakktor --version
cargo uninstall trakktor
```

On Linux, the build pulls in `reqwest`'s default TLS, which needs OpenSSL
(`pkg-config` plus `libssl-dev`/`openssl-devel`); macOS uses the system TLS and
needs nothing extra.

## Layout

A Cargo workspace with a flat crate layout:

- `trakktor` — the CLI binary: argument parsing, configuration, output
  formatting. A thin layer over the library.
- `trakktor_core` — the library: all functionality, independent of the CLI.

## Build & test

```sh
cargo build --workspace
cargo test --workspace
```

Formatting uses unstable rustfmt features, so it requires nightly:

```sh
cargo +nightly fmt --all
```

## Output and exit codes

The default output is machine-readable JSON (`--pretty` indents it); `--text`
switches to human-readable text. Data is written to stdout, errors to stderr.

- `0` — success.
- `1` — runtime/validation error (network, parsing, I/O, bad value). Formatted
  like normal output: JSON `{ "error": { "code": "…", "message": "…" } }` by
  default, plain text with `--text`.
- `2` — usage error (unknown flag, missing argument); always text, from the
  argument parser.

Global options (usable before or after the command): `--work-dir <path>` (also
`TRAKKTOR_DIR`; default `./.trakktor`, per-project state such as feed
read-state), `--model-dir <path>` (also `TRAKKTOR_MODEL_DIR`; default
`~/.trakktor`, where model weights are cached — see `asr`), `--text`,
`--pretty`.

## `asr` — speech recognition

Transcribe (or translate) speech from an audio file. Speech recognition is
organized as **engines**, each selected as a subcommand with its own model and
flags; one engine ships today, `whisper`:

```sh
trakktor asr whisper talk.mp3            # JSON (default): text + timestamped segments
trakktor asr whisper talk.mp3 --text     # readable [start --> end] lines
trakktor asr whisper talk.mp3 --pretty   # indented JSON
```

The one required argument is the path to an audio file. It is decoded by a
**built-in decoder** — no ffmpeg or other external tool is needed — then
internally downmixed to mono and resampled to 16 kHz. Supported inputs: mp3,
aac (LC), vorbis, flac, alac, and raw PCM, held in wav, aiff, caf, ogg, mp4, or
mkv containers. Formats beyond that set are reachable with `--audio-decoder
ffmpeg` (see [Input decoder](#input-decoder)).

Transcription runs window by window (30 seconds each) with a temperature-fallback
policy that guards against repetition loops, closely following Whisper's own
decoding behavior.

### Models

```
--model <name|dir>      # default: tiny
```

Pass a **published name** — downloaded on first use into the model directory
(`~/.trakktor/asr/whisper/<name>/` by default) and reused on later runs — or a
**path** to a local checkpoint directory (one containing `config.json`).
First-run download progress is printed to stderr.

Model weights are large and **shared across projects**: they live under the
**model directory** — `~/.trakktor` by default — *not* in the per-project
`--work-dir`. Override it with `--model-dir <path>` or `TRAKKTOR_MODEL_DIR`
(precedence: flag > env > `~/.trakktor`). To reuse weights already downloaded
elsewhere, point it at that root — e.g. `TRAKKTOR_MODEL_DIR=/data/models` looks
for `/data/models/asr/whisper/<name>/`.

Names, smallest to largest (larger is slower but more accurate):

- **Multilingual:** `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`,
  `large-v3` (alias `large`), `large-v3-turbo` (alias `turbo`).
- **English-only:** `tiny.en`, `base.en`, `small.en`, `medium.en` — slightly
  better on English audio.

### Device and precision

```
--device cpu|metal      # default: cpu
--precision f16|f32     # default: f16
```

- **`--device metal`** runs on the macOS GPU and is several times faster than
  the CPU on a typical clip. It needs a build with the `metal` feature:

  ```sh
  cargo install --locked --git https://github.com/lymar/trakktor.git \
    --branch agent-cli --features metal trakktor
  ```

  Without that feature, `--device metal` is rejected.

- **`--precision f16`** (the default) uses about half the memory and is faster;
  **`--precision f32`** computes in full precision for reproducible results, at
  twice the weight memory. f16 is what keeps the large models within reach on a
  16 GB machine.

### Input decoder

```
--audio-decoder builtin|ffmpeg   # default: builtin
```

`builtin` (the default) is the pure-Rust decoder described above and needs no
external tools. `ffmpeg` instead shells out to an installed `ffmpeg`, which
decodes many more input formats — opus, HE-AAC, wma, amr, and others the
built-in decoder does not cover:

```sh
trakktor asr whisper voice.opus --audio-decoder ffmpeg
```

It requires `ffmpeg` on your `PATH`; without it the command fails with a clear
error. The default build never needs ffmpeg.

### Language and task

```
--language <code|name>        # e.g. en, ru, or russian; autodetected if omitted
--task transcribe|translate   # default: transcribe
```

With no `--language`, the language is detected from the first 30 seconds.
`--task translate` renders the speech as English instead of transcribing it in
the source language.

### Clipping: transcribe only part of the audio

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

### Voice-activity detection (VAD)

`--vad` runs Silero voice-activity detection first and transcribes only the
speech, dropping silence, music, and noise. It is the direct remedy for
Whisper's tendency to hallucinate and loop over long non-speech stretches, and
it is faster on sparse audio. Off by default; the model is built in (nothing to
download) and runs on the CPU, so `--device metal` still accelerates the
transcription itself.

```sh
trakktor asr whisper interview.mp3 --vad
```

Two modes control how the detected speech reaches the model:

```
--vad-mode collapse|fragments   # default: collapse
```

- `collapse` (default) — the speech is glued into one dense buffer, transcribed
  in a single pass, then the timestamps are mapped back. It packs speech tightly
  and is markedly faster on fragmented speech.
- `fragments` — each speech span is transcribed where it is and the silence
  between spans is skipped; timestamps stay native on the original timeline.

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

### Timestamps and output shape

```
--timestamps none|segment|word   # default: segment
```

- `segment` — per-segment start/end times (the default).
- `word` — segment times plus per-word timings, from an extra alignment pass
  (slower).
- `none` — text only, no segment list.

The default output is a single JSON object: the transcript `text`, the detected
or given `language`, the audio `duration` (seconds), the `engine`, and — unless
`--timestamps none` — a `segments` array. Each segment carries its `id`,
`start`, `end`, `text`, engine-specific quality signals under `whisper`, and
(with `--timestamps word`) a `words` array:

```json
{
  "text": "Ask not what your country can do for you.",
  "language": "en",
  "duration": 11.0,
  "engine": { "name": "whisper", "model": "tiny" },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.6,
      "text": "Ask not what your country can do for you.",
      "whisper": {
        "avg_logprob": -0.29,
        "compression_ratio": 1.15,
        "no_speech_prob": 0.01,
        "temperature": 0.0
      }
    }
  ]
}
```

With `--timestamps word`, each segment also gets a `words` array of
`{ start, end, word, probability }`:

```json
"words": [
  { "start": 0.0, "end": 0.42, "word": "Ask", "probability": 0.98 }
]
```

The `whisper` block is diagnostic: `avg_logprob` (average token
log-probability — confidence), `compression_ratio` (zlib ratio; high means
repetitive), `no_speech_prob`, and the `temperature` the accepted result was
decoded at. `--text` instead prints one right-aligned `[start --> end] text`
line per segment (or just the transcript with `--timestamps none`), and errors
follow the usual `{ "error": { "code", "message" } }` contract.

While a long file decodes, a live progress line — audio position, percent,
elapsed, and a rough estimate of the time remaining — is written to **stderr**,
so stdout stays a clean JSON (or text) stream.

### Output files

By default the result goes to stdout as JSON (or text with `--text`).
`--output-format` **additionally** writes the transcript to files, without
changing what stdout prints:

```
--output-format txt|vtt|srt|tsv|json|all   # which file(s) to write
--output-dir <dir>                         # where (default: the current dir)
```

| Format | File | Contents |
|---|---|---|
| `txt` | `<audio>.txt` | one line per segment |
| `vtt` | `<audio>.vtt` | WebVTT subtitles |
| `srt` | `<audio>.srt` | SubRip (SRT) subtitles |
| `tsv` | `<audio>.tsv` | `start`⇥`end`⇥`text`, times in integer milliseconds |
| `json` | `<audio>.json` | the full JSON result (same as stdout) |
| `all` | — | every format above |

Each file is named after the audio (`talk.mp3` → `talk.srt`) and written into
`--output-dir` (created if it does not exist); the paths written are reported on
stderr. Subtitle cues are one per segment.

```sh
# Write SubRip subtitles (stdout still prints the JSON result)
trakktor asr whisper talk.mp3 --output-format srt

# Write every format into subs/
trakktor asr whisper talk.mp3 --output-format all --output-dir subs
```

### Decoding controls

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
- **Voice-activity detection** — `--vad` (with `--vad-mode` and the `--vad-*`
  tuning flags) detects speech and skips non-speech before transcribing; see
  the VAD section above.

### Examples

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

## `feed` — RSS / Atom / JSON Feed

### Discover feeds on a page

```sh
trakktor feed discover https://example.com
```

Returns the feeds declared on the page (`url`, `type`, `title`). An empty result
is success.

### Read a feed

```sh
trakktor feed read https://example.com/feed.xml             # JSON (default)
trakktor feed read https://example.com/feed.xml --all --fields all --text
```

Accepts a feed URL or a regular page (autodiscovery applies, reading the first
feed found). Each publication carries a stable `uid` and an `is_read` flag.

- By default only **unread** publications are returned; `--all` includes read
  ones.
- `uid` is each publication's **primary key** — the id you pass to `mark-read` —
  so it is **always** included, independent of `--fields`.
- `--fields <list>` selects the *additional* fields: a comma-separated list of
  `is_read,title,link,published,updated,summary,content,authors`, or the special
  values `minimal` (default, `title,link`) and `all`.
- With `--text`, `uid` is the first column and a `mark-read` hint is printed to
  stderr.

The `uid` is `hex(BLAKE3(feed_key ‖ 0x00 ‖ tag ‖ 0x00 ‖ item_key))` and is
stable across runs for the same feed + entry.

### Mark publications read

```sh
trakktor feed mark-read <uid> [<uid>...]
```

Idempotent. Read state is stored as plain files under `<work-dir>/feed/`, sharded
by uid; nothing else is needed (no database).

## Typical agent workflow

```sh
trakktor feed discover https://example.com             # find a feed
trakktor feed read https://example.com/feed.xml        # read unread items (JSON)
# … take each item's uid …
trakktor feed mark-read <uid1> <uid2>                  # mark them handled
```

On the next `read`, marked publications are no longer returned.

## `skill` — generate the agent skill

trakktor can describe itself to a coding agent as an Agent Skill. The content is
generated from the live `clap` definition, so it always matches the installed
binary — there is no hand-written reference to drift out of date.

### Show the skill

```sh
trakktor skill show          # narrative guide (JSON { "content": … } by default)
trakktor skill show --full   # plus the full command/flag/value reference
trakktor skill show --text   # the raw Markdown
```

By default the Markdown is wrapped as `{ "content": "…" }`; `--text` prints the
raw Markdown.

### Install the stub

```sh
trakktor skill install claude            # ./.claude/skills/trakktor/SKILL.md
trakktor skill install agents            # ./.agents/skills/trakktor/SKILL.md
trakktor skill install claude --global   # ~/.claude/skills/trakktor/SKILL.md
trakktor skill install claude --force    # overwrite an existing stub
```

Writes a thin discovery stub to `<dir>/skills/trakktor/SKILL.md`. The
destination is explicit: `claude` or `agents` selects the agent layout in the
current project, and `--global` (only valid with `claude`) targets `~/.claude`
in your home directory. A project install creates the whole path; a global
install requires `~/.claude` to already exist — it is never created, and the
command fails if it is missing. The stub only points back at `trakktor skill
show`, so it never goes stale between releases. An existing `SKILL.md` is left
untouched unless `--force` is given. By default the result is a single JSON
object `{ "path": "…", "status": "written" | "skipped" }`; `--text` prints it as
lines.
