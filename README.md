# trakktor

`trakktor` is a predictable, automation-friendly Rust CLI toolbox for coding
agents such as Claude Code and OpenCode (humans are welcome to use it too):
speech-to-text and text-to-speech, speech enhancement, voice-activity audio
editing, feeds, text structuring, and more — machine-readable output, stable
flags, and meaningful exit codes. No Python, no virtual environments, no
drawn-out setup: a single binary that takes care of everything itself,
downloading and caching the models it needs on first use.

## Commands

Each command is documented in full on its own page under
[`docs/features/`](docs/features/):

- [**`asr` — speech recognition**](docs/features/asr/README.md): transcribe
  (or translate) speech from an audio file, with three engines —
  [`whisper`](docs/features/asr/whisper.md) (many languages, autodetection,
  translation to English, plus the Podlodka fine-tunes for Russian),
  [`gigaam`](docs/features/asr/gigaam.md) (GigaAM Conformer models, mainly
  Russian), and [`vosk`](docs/features/asr/vosk.md) (Vosk's Zipformer2
  transducers, offline and streaming).
- [**`tts` — speech synthesis**](docs/features/tts/README.md): text in, an
  audio file out — plain text or Markdown, any length, fully local. Three
  engines — [`qwen3-tts`](docs/features/tts/qwen3-tts.md) (ten languages,
  nine preset voices), [`espeech`](docs/features/tts/espeech.md) (Russian, the
  voice cloned from a recording you supply, stress marked automatically), and
  [`silero`](docs/features/tts/silero.md) (sixty preset voices across **twenty
  languages** — Russian, Ukrainian, Belarusian, Kazakh, Tatar, Bashkir,
  Georgian, Armenian and a dozen more of the region, though no Latin-script
  language and so no English — at 48 kHz, and fast enough on a CPU to be the
  one to use without a GPU).
- [**`ocr` — read text off images**](docs/features/ocr/README.md): scans,
  photographs of pages and screenshots in, text out — page by page, so a
  document is read by passing its pages in order. Two engines.
  [`paddle`](docs/features/ocr/paddle.md) is a port of PaddleOCR's classic
  pipeline (detection, then recognition) reading its published models directly,
  both live generations of them: fourteen recognizers covering Cyrillic, Latin,
  Arabic, Devanagari, Korean, Thai, Greek, Tamil, Telugu and Chinese/Japanese.
  It reads with the best models it has for the language, which is not one
  generation for all of them — the newest carries no Cyrillic at all, so a
  Russian page is found by the new detector and read by the older recognizer.
  About 139 MB for an English page and some ten seconds; `--quality fast` trades
  that back for 13 MB and a quarter to a third off the time.
  [`vl`](docs/features/ocr/vl.md) is a port of the
  PaddleOCR-VL document model, which **writes out** what it sees instead of
  picking characters from a dictionary: it works out the writing system by
  itself, reads scripts the classic pipeline has no model for at all, and
  returns a table as markup or a formula as LaTeX — for about 2 GB of weights
  and tens of seconds a page. Output either way is JSON with every line's box
  and confidence, plain text, or Markdown with paragraphs and a reading order
  worked out from the geometry. A third model
  ([`layout`](docs/features/ocr/layout.md), 130 MB, about a second a page)
  labels the blocks of the page — title, heading, paragraph, footnote, running
  head, page number, table, formula, picture — so the structure is read rather
  than guessed; it runs by default, `--no-layout` skips it, and `ocr layout`
  runs it on its own and answers "what is on this page" without reading a word.
  A page **photographed** rather than scanned gets
  [three more steps](docs/features/ocr/photo.md), off by default and available
  to both engines: `--doc-orientation` finds which right angle the page is at
  and turns it upright, `--sheet` cuts the sheet out of the frame, and
  `--unwarp` straightens it — the perspective of a shot taken at an angle and
  the curve of a page that will not lie flat. On a set of twelve photographs of
  one page, that takes a sideways or curled shot from unreadable to reading
  exactly as well as a 300 dpi scan of the same page. Boxes still come back on
  the photograph you passed in.
- [**`enhance` — clean up a speech recording**](docs/features/enhance/README.md):
  a damaged recording in, a repaired one out — room noise and hiss removed,
  reverberation reduced, and, with the right engine, the holes a dropped packet
  leaves in a call filled in from the words on either side. Two engines.
  `gtcrn` is the one to reach for: a masking network of
  **forty-eight thousand** parameters, 580 KB of weights, about a sixtieth of
  real time on one CPU core. `unipase` is generative — five hundred and forty-six
  million parameters — and earns its keep on exactly one thing a mask cannot do,
  filling a hole. Either way this repairs damage and does **not** improve a
  recording that is already good: the page says where each helps and where it
  hurts, measured against two independent recognisers.
- [**`vad` — voice-activity audio editing**](docs/features/vad/README.md):
  find the speech in an audio file and report it, cut the silence out, or
  split the recording into clips.
- [**`text` — structure and transform text**](docs/features/text/README.md):
  split a wall of text into paragraphs (`structify`), restore punctuation and
  casing in raw transcripts (`punctuate`), and mark the stressed vowel in
  Russian — with the letter `ё` restored — for a speech synthesizer to read
  (`stress`), all fully offline.
- [**`feed` — RSS / Atom / JSON Feed**](docs/features/feed/README.md):
  discover feeds on a page and read them with per-item read state.
- [**`skill` — generate the agent skill**](docs/features/skill/README.md):
  trakktor describes itself to a coding agent, generated from the live CLI
  definition so it always matches the installed binary.

## Install

Requires a stable Rust toolchain **≥ 1.92** (edition 2024; nightly is only
needed for `cargo fmt`, not to build). The binary installs into `~/.cargo/bin`,
which must be on your `PATH`.

From git:

```sh
cargo install --locked --git https://github.com/lymar/trakktor.git trakktor
```

- `trakktor` (the trailing word) is the package to install — the workspace root
  is virtual, so it must be named.
- `--locked` builds with the exact dependency versions pinned in `Cargo.lock`
  (reproducible); omit it to pull the latest semver-compatible versions.

From a local clone (installs whatever is checked out):

```sh
git clone https://github.com/lymar/trakktor.git
cd trakktor
cargo install --locked --path trakktor
```

For GPU acceleration on macOS — speech recognition and synthesis, OCR, speech
enhancement, and the `text` commands — add `--features metal` to either
install command; the `burn` build feature adds the alternative burn runtime
(`--runtime burn`) the same way. See
[Device and precision](docs/features/asr/README.md#device-and-precision) in
the `asr` docs.

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
`~/.trakktor`, where model weights are cached — see
[Model storage](docs/features/asr/README.md#model-storage)), `--text`,
`--pretty`.

## Acknowledgments

trakktor ports and builds on several open-source projects, all MIT- or
Apache-licensed — Whisper, GigaAM, Vosk, Qwen3-TTS, F5-TTS and the ESpeech
checkpoints, Vocos, Silero-VAD, Silero TTS, Silero Stress, SaT / wtpsplit,
the 1-800-BAD-CODE punctuation model, PaddleOCR (with DB, the detection
algorithm it builds on), PaddleOCR-VL (with the ERNIE-4.5 decoder it is built
on) and UVDoc for straightening photographed pages, GTCRN and UniPASE (with
WavLM and PASE behind them) for speech enhancement, on the candle and burn
runtimes. The full credits — with licenses, upstream links, and papers — live
in [`docs/acknowledgments.md`](docs/acknowledgments.md); see
[`NOTICE`](NOTICE) for the complete third-party attributions and license
notices.
