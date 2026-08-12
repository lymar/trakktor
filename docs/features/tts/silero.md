# `tts silero` — Silero engine (preset voices, fast on the CPU)

> Part of [`tts`](README.md) and of [trakktor](../../../README.md); the global
> flags and the output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes), and the
> shared `tts` behavior — text input (`--text-file`, `--text-format`),
> paragraph splitting and joining, `-o/--output`, output containers and
> encoders (`--audio-encoder`, `--bitrate`) — on [the `tts` page](README.md).

A native port of Silero TTS v5: **sixty preset voices across twenty languages**,
synthesized at **48 kHz**. It is the engine to reach for on a machine without a
GPU: one pass, no sampling, no autoregression, and a page of prose spoken in a
few seconds on a laptop CPU.

```bash
trakktor tts silero "Пример синтеза речи." -o hello.wav

# another language: the voice decides which one
trakktor tts silero "Д+обрий д+ень, як сьог+одні спр+ави." --voice ukr_igor -o hello.wav

# a whole article, in a chosen voice
trakktor tts silero --text-file article.md --voice ru_eduard -o article.mp3

# what voices does this model have?
trakktor tts silero --voice list
```

Two things follow from the model having no randomness anywhere. There is no
`--seed`: the same command produces the same file, always. And the reading is
shaped by knobs rather than by re-rolling — `--rate` for pace, `--pitch` for
the voice's height.

```bash
trakktor tts silero --text-file article.md --rate 0.9 --pitch 1.05 -o slower.wav
```

| Flag | Default | Meaning |
|---|---|---|
| `<text>` / `--text-file <path\|->` | — | The text to speak, as the argument or from a file (`-` reads standard input); at most one. Only `--voice list` needs neither. |
| `--voice <name\|list>` | `ru_zhadyra` | The speaker — and with it the language, named by the prefix (see [Languages](#languages)). `list` prints the selected model's voices and exits. |
| `--model <name\|dir>` | `cis-base` | `cis-base`, `cis-base-nostress`, `cis-ext`, `ru-classic`, or a converted model directory (see [Voices and models](#voices-and-models)). |
| `--allow-non-commercial-models` | off | Required for the CC BY-NC-SA 4.0 models (`cis-ext`, `ru-classic`). Grants no license — it records that the choice was deliberate. |
| `--sample-rate <48000\|24000\|8000>` | `48000` | What the model produces: 48 kHz is synthesized, the lower two come from its own filterbank (see [Sample rate](#sample-rate)). |
| `--rate <float>` | `1.0` | Speech rate as a multiplier: below 1 slower, above 1 faster — it repaces the reading rather than replaying it at another speed. |
| `--pitch <float>` | `1.0` | Pitch as a multiplier, scaled by the speaker's own range, so the voice stays itself; useful values sit between 0.75 and 1.25. |
| `--stress <auto\|off>` | `auto` | Whether to mark the stress before speaking (see [Stress](#stress)). Marks you wrote yourself are never moved. |
| `--pause-ms <ms>` | `500` | Gap between paragraphs (half that inside a split paragraph). |
| `--levels <match\|keep>` | `match` | Bring the pieces to a common loudness before joining, or keep the levels the model gave them. |
| `-o, --output <path>` | `speech.wav` | Where to write the audio; the container follows the extension. |
| `--text-format <auto\|txt\|md>` | `auto` | Where paragraphs end and whether markup is stripped. |
| `--audio-encoder <auto\|builtin\|ffmpeg>` | `auto` | Who writes the file: the built-in wav/flac encoder, or ffmpeg for everything else. |
| `--bitrate <rate>` | — | `-b:a` for lossy ffmpeg formats, e.g. `192k`. |
| `--runtime <candle\|burn>` | `candle` | Inference runtime; burn needs the `burn` build feature (see [Speed](#speed)). |
| `--device <cpu\|metal>` | `cpu` | Compute device; this model is meant for the CPU rather than falling back to it. The `metal` feature covers the candle runtime; `--runtime burn` brings its own Metal backend. |

## Languages

**The language is not a flag — it is the voice.** Each speaker was trained for
one language and its name says which: the prefix is the ISO 639-3 code (`ru`
for Russian being the one exception). There is no `--language` to set and no
detection to get wrong.

The default `cis-base` model, with the voices it has for each:

| | | | |
|---|---|---|---|
| Russian `ru_` — 29 | Bashkir `bak_` — 5 | Belarusian `bel_` — 3 | Kazakh `kaz_` — 2 |
| Khakas `kjh_` — 2 | Kalmyk `xal_` — 2 | Tajik `tgk_` — 2 | Tatar `tat_` — 2 |
| Ukrainian `ukr_` — 2 | Armenian `hye_` — 1 | Azerbaijani `aze_` — 1 | Chuvash `chv_` — 1 |
| Erzya `erz_` — 1 | Georgian `kat_` — 1 | Kabardian `kbd_` — 1 | Kyrgyz `kir_` — 1 |
| Moksha `mdf_` — 1 | Udmurt `udm_` — 1 | Uzbek `uzb_` — 1 | Yakut `sah_` — 1 |

`cis-ext` (non-commercial) adds 35 more voices for six of those: Tatar (19),
Kazakh and Ukrainian (5 each), Chuvash, Kalmyk and Uzbek (2 each).

**Georgian, Armenian, and Azerbaijani or Uzbek written in Latin are
transliterated** into the model's own alphabet before it reads them, so they can
be written in their own script:

```bash
trakktor tts silero "გამარჯობა, როგორ ხარ" --model cis-base-nostress --voice kat_vika -o ka.wav
trakktor tts silero "Բարև ձեզ, ինչպես եք"  --model cis-base-nostress --voice hye_zara -o hy.wav
trakktor tts silero "Исәнмесез, хәерле көн" --model cis-base-nostress --voice tat_albina -o tt.wav
trakktor tts silero "Сәлеметсіз бе, қалыңыз қалай" --model cis-base-nostress --voice kaz_zhadyra -o kk.wav
```

Those four use `cis-base-nostress`, which needs stress marks only for the Slavic
languages; on the default `cis-base` every language wants them (see below).

> **English is not among the twenty, and neither is any other Latin-script
> language.** The model's alphabet is Cyrillic, and Latin letters are removed
> rather than read — an English sentence comes back as the error `text_empty`,
> and an English word inside a Russian one simply disappears. For English use
> [`tts qwen3-tts`](qwen3-tts.md).

## Stress

Russian does not write stress, and this model reads it as a **real input**: `+`
before the stressed vowel is a symbol of its alphabet, not a hint. `з+амок` is
a lock, `зам+ок` a castle.

By default the engine marks the text itself, with the same model
[`text stress`](../text/README.md#text-stress--mark-the-stress-in-russian-text)
uses (downloaded on first need). A `+` you wrote yourself is never moved, so
correcting the odd word by hand and leaving the rest to the model is the normal
way to work. The marker is Russian, and it runs only for a Russian voice: with
any other the text is spoken as written, and marks for those languages are
yours to place (the default model expects them in every language,
`cis-base-nostress` only in the Slavic ones):

```bash
# marked automatically (the default)
trakktor tts silero --text-file article.txt -o article.wav

# mark it yourself, look at it, fix it, then speak it
trakktor text stress article.txt --text > marked.txt
trakktor tts silero --text-file marked.txt --stress off -o article.wav
```

## Voices and models

`--voice list` prints the voices of the selected model, whose prefixes say what
each of them speaks (see [Languages](#languages)); the default voice is
`ru_zhadyra`.

| `--model` | License | Voices |
|---|---|---|
| `cis-base` (default) | MIT | 60 across twenty languages, 29 Russian; expects stress marks in every language |
| `cis-base-nostress` | MIT | the same 60, trained to need marks only for the Slavic languages |
| `cis-ext` | CC BY-NC-SA 4.0 | 35 more for Chuvash, Kalmyk, Kazakh, Tatar, Ukrainian, Uzbek |
| `ru-classic` | CC BY-NC-SA 4.0 | the five long-standing Russian voices — `aidar`, `baya`, `kseniya`, `eugene`, `xenia` — with question intonation |

**The license depends on the model, not on the engine.** The two `cis-base`
models are published under MIT and are available by default; the other two are
non-commercial and share-alike, and asking for one without
`--allow-non-commercial-models` is an error — before anything is downloaded:

```bash
$ trakktor tts silero "Пример." --model ru-classic
{"error":{"code":"model_license_restricted","message":"model `ru-classic` is
published under CC BY-NC-SA 4.0; pass --allow-non-commercial-models to use it,
or choose one of: cis-base, cis-base-nostress"}}
```

The flag grants no license — trakktor is not a party to it, and the file
travels from its publisher to you. It records that the choice was deliberate
and makes it reproducible in a script. Whichever model runs, its license is
reported in the output, so the constraint is visible where the decision is
made. Here is what the first example on this page reports:

```json
{
  "output": "hello.wav", "format": "wav", "sample_rate": 48000,
  "duration": 1.54, "chunks": 1,
  "voice": { "kind": "preset", "name": "ru_zhadyra" },
  "engine": { "name": "silero", "model": "cis-base", "runtime": "candle" },
  "silero": { "license": "MIT", "symbols": 25, "frames": 135,
              "rate": 1.0, "pitch": 1.0,
              "stressed_text": "Прим+ер с+интеза р+ечи." }
}
```

`stressed_text` is what the model actually read, present when the automatic
marker ran; `dropped_characters` and `skipped_paragraphs` appear when the
frontend removed anything (see [What the text loses](#what-the-text-loses));
`utterances` — the utterance types given to the intonation head (`statement`,
`question`, and so on) — only with `ru-classic`, the one model that has one.
Note there is no `language` field, unlike the other two engines: here the
language is the voice.

## Sample rate

The model synthesizes 48 kHz, and it derives 24 kHz and 8 kHz from that with a
filterbank of its own rather than by resampling — a different signal, not the
same one interpolated. `--sample-rate <48000|24000|8000>` picks which of the
three the model produces; the output file's container and codec are still
chosen by the extension of `--output`.

## What the text loses

The model's alphabet is Cyrillic, and its frontend removes what it cannot
spell rather than guessing:

- **Latin disappears.** An English word inside a Russian sentence is not
  transliterated, it is dropped; a text with nothing else in it is the error
  `text_empty`. So is anything in a script the model has no table for.
- **A long dash disappears** in the `cis-base` models — the frontend rewrites
  it as a short one, which their alphabet does not hold. `ru-classic` keeps it.
  Surrounding punctuation still gives the pause; the dash itself does not.
- **`!` disappears** in the `cis-base` models, for a different reason of the
  same kind. `ru-classic` reads it.
- **Digits disappear.** The model has no numbers and does not spell them out;
  `Глава 5` is read as `глава`. Write numbers out as words.

A paragraph with nothing readable in it at all — a code block, an English
quote, a bare number — is skipped whole rather than failing the run, and only
a text with no readable paragraph left is the error `text_empty`.

None of this is silent: the output reports `dropped_characters` when the
cleaning removed anything, and `skipped_paragraphs` when whole paragraphs
were passed over.

## Speed

One pass over a few tens of millions of parameters, so the CPU is the intended
place to run this rather than a fallback. On an Apple M-series laptop, reading
a page of Russian prose (2.4 minutes of speech), wall clock including loading
the model:

| Runtime | Device | Time | Times real time |
|---|---|---:|---:|
| candle | metal | 3.2 s | 44× |
| candle | cpu | 6.6 s | 22× |
| burn | cpu | 12.1 s | 12× |
| burn | metal | 9.9 s | 14× |

`--runtime burn` is the alternative runtime, available in builds with the
`burn` feature; for `--device metal`, the `metal` feature covers the candle
runtime, and `--runtime burn` brings its own Metal backend. The four
combinations agree with each other to within a cosine similarity of
0.99999998 — they are the same reading, not four different ones.

## Long text

Text of any length works. One pass is capped by the model's own position
tables at 62.5 seconds of speech, so the text is split into paragraphs, a
paragraph too long is split further (with a local sentence model, downloaded on
first need, and at punctuation where that is not enough), and the pieces are
joined into one file with `--pause-ms` between them. Slowing the reading
shortens the pieces in step — `--rate 0.5` stretches the same words over twice
the frames, and the cap is in frames. `--levels match` (the
default) brings the pieces to a common loudness first.

## Errors

Beyond the shared ones: `model_license_restricted` (a non-commercial model
without the flag), `unsupported_voice` (the message lists the voices this model
has), `text_too_long` (a piece that would not fit even after splitting), and
`text_empty` (nothing readable left after the cleaning — a line of Latin, say).
