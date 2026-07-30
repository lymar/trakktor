# `tts` — speech synthesis

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

`trakktor tts` is the counterpart to [`asr`](../asr/README.md): text in, an
audio file out. Like
`asr`, it is organized as a set of engines, each picked as a subcommand, and
runs entirely locally. Three engines, each documented on its own page:

- [**`silero`**](silero.md) — sixty **preset** voices across **twenty
  languages**: Russian (29 voices), Bashkir, Belarusian, Ukrainian, Kazakh,
  Tatar, Tajik, Khakas, Kalmyk, Armenian, Azerbaijani, Chuvash, Erzya,
  Georgian, Kabardian, Kyrgyz, Moksha, Udmurt, Uzbek, Yakut. No Latin script,
  so no English. 48 kHz, no randomness anywhere, and fast enough on a CPU that
  a GPU is an optimization rather than a requirement. Start here for Russian
  and for the languages of the region.
- [**`qwen3-tts`**](qwen3-tts.md) — ten languages, nine **preset** voices to
  pick from.
- [**`espeech`**](espeech.md) — Russian only, and the voice is **cloned** from
  a recording you
  supply: a few seconds of someone speaking, plus the text they said.

```bash
trakktor tts silero "Пример синтеза речи." -o hello.wav
trakktor tts silero "Д+обрий д+ень, як сьог+одні спр+ави." --voice ukr_igor -o hello.wav

trakktor tts qwen3-tts "Привет! Это синтез речи." --language russian -o hello.wav

# the same text in a voice taken from a recording
trakktor tts espeech "Привет! Это синтез речи." \
  --ref-audio voice.wav --ref-text "Что именно сказано в этой записи." -o hello.wav

# a whole article — plain text or Markdown, any length
trakktor tts qwen3-tts --text-file article.md --language russian -o article.mp3

# or from a pipe, saying how to read it
cat notes.md | trakktor tts qwen3-tts --text-file - --text-format md -o notes.wav
```

The text comes as the positional argument, from `--text-file`, or from standard
input (`--text-file -`) — exactly one of them.

**Long text is spoken whole.** One utterance is capped at two minutes of speech
(the checkpoints would allow more, but a single derailed generation costs
minutes and the model was trained on utterances, not chapters), so the text is
split into paragraphs, each is spoken separately, and the pieces are joined
into one file:

- `--text-format <auto|txt|md>` says where paragraphs end. `txt` takes one
  paragraph per line; `md` separates them with blank lines and strips the
  markup — headings, list and quote markers, emphasis, inline code, links (the
  link text is kept, the URL is not), front matter, HTML comments. Tables and
  code blocks keep their **text**: removing markup is not the same as deciding
  what you did not want to hear. `auto` (the default) reads the file extension,
  then the text itself, and falls back to `md`.
- A paragraph still too long for one utterance is split further with
  [`text structify`](../text/README.md#text-structify--split-text-into-paragraphs),
  which is
  already a boundary model — it picks the coarsest cut that fits, so the text
  breaks as few times as possible. That model is downloaded and loaded **only**
  if some paragraph actually needs it.
- `--pause-ms <ms>` (default 500) sets the gap between paragraphs. Each piece
  is trimmed of the silence the model leaves at its edges and faded at the
  join, so the pause is exactly what you asked for and the seams do not click;
  pieces of one split paragraph get half the gap.
- `--levels match` (the default) brings the paragraphs to a common loudness
  before joining. Each is spoken as its own utterance and the model picks a
  level for it anew — measured 3.4 dB of spread on `serena` and 7.1 dB on
  `uncle_fu` across one five-paragraph run, which is heard as the reading
  jumping in volume. Matching moves each piece to the median level, by at most
  6 dB and never into clipping. `--levels keep` leaves them as synthesized.
  (Pace and delivery drift too, and that part cannot be fixed after the fact —
  lowering the temperature does not help either.)
- Every piece is sampled from a seed derived from `--seed`, so the same run
  reproduces the same file.

The audio is always written to a file — raw samples on stdout would not survive
the machine-readable contract — and `stdout` carries the metadata:

```json
{
  "output": "hello.wav",
  "format": "wav",
  "sample_rate": 24000,
  "duration": 2.16,
  "chunks": 1,
  "language": "russian",
  "voice": { "kind": "preset", "name": "serena" },
  "engine": { "name": "qwen3-tts", "model": "0.6b-customvoice", "runtime": "candle" },
  "qwen3-tts": { "frames": 27, "sampling": "top_k", "seed": 0, "top_k": 50, "temperature": 0.9 }
}
```

`chunks` is how many pieces were spoken and stitched together, and `frames`
their total. Should a piece still run into the two-minute cap, it is cut down
and spoken again (twice at most); if even that does not help, the engine block
carries `"truncated": true` and the tail of that piece is missing.

The container follows the `--output` extension: `.wav` (32-bit float, exactly
as synthesized) and `.flac` (quantized to 24 bits) are written by the built-in
encoder, and anything else — mp3, m4a, opus, ogg — is handed to an installed
`ffmpeg` (`--audio-encoder` forces the choice, `--bitrate` sets `-b:a` for
lossy formats).
