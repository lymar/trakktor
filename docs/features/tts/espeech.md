# `tts espeech` — ESpeech engine (cloned voice, Russian)

> One of the engines of [trakktor's `tts` command](README.md); the shared
> behavior — text input, paragraph splitting and joining, loudness matching,
> output containers — is documented there.

A native port of the Russian **ESpeech-TTS-1** checkpoints, which are
**F5-TTS** models. Nothing here works frame by frame: the model starts from noise
the shape of the whole utterance and refines it over a fixed number of steps into
a mel spectrogram, and a small vocoder turns that into a 24 kHz waveform. Three
consequences you can see from the outside:

- **the length is decided before generation**, from the speech rate of your
  reference recording, so nothing can run away and nothing is ever truncated;
- **the voice comes only from the reference** — there are no preset speakers;
- **a seed pins the whole reading exactly**: all the randomness is that one
  starting point.

```bash
# a voice, and what is said in the recording
trakktor tts espeech "Вторая глава начинается с описания дороги." \
  --ref-audio narrator.wav --ref-text "Ветер стих только к утру, и стало слышно реку." \
  -o chapter.wav

# faster, at some cost in polish
trakktor tts espeech --text-file chapter.md --ref-audio narrator.wav \
  --ref-text-file narrator.txt --nfe-step 16 -o chapter.mp3
```

| Flag | Default | Meaning |
|---|---|---|
| `--ref-audio <path>` | — | **Required.** The recording whose voice to speak in. Silence at the edges is trimmed and anything past 12 s goes unused — that is what the model conditions on. |
| `--ref-text <text>` / `--ref-text-file <path>` | — | **Required** (exactly one). What is said in the recording, word for word: the model aligns the audio against this text, so a wrong transcript costs quality. |
| `--model <name\|dir>` | `rl-v2` | `rl-v2`, `rl-v1`, `sft-256k`, `sft-95k`, `podcaster` (trained on podcast delivery), or a checkpoint directory. All the same size — the choice is one of manner, not of quality against speed. |
| `--nfe-step <int>` | `32` | Solver steps per piece, and the time scales with it exactly: measured 33 / 62 / 91 s for 16 / 32 / 48. Intelligibility does not — all three transcribe back identically — so 16 is the honest way to halve the time, 32 keeps the reference pipeline's margin for naturalness, and 48 buys nothing measurable. |
| `--cfg-strength <float>` | `2.0` | How strongly the reading is pushed toward the text and the reference voice. `0` switches off the second, unguided pass — exactly twice as fast, and not worth it: measured, the words come out mangled and half-swallowed (transcribing the result back no longer matches what went in). |
| `--speed <float>` | `1.0` | Speech rate as a multiplier on the predicted duration: below 1 gives the words more room. |
| `--seed <int>` | `0` | The noise the reading starts from. Same seed, same file; another seed, another reading of the same text in the same voice. |
| `--stress <auto\|off>` | `auto` | Whether to mark the stress before speaking (see below). `auto` marks the text **and the reference transcript**; `off` speaks them exactly as given. |
| `--language <lang>` | `auto` | Exists for symmetry across the engines; this one speaks Russian — `auto` and `russian` are accepted, anything else is refused. |
| `-o/--output`, `--text-file`, `--text-format`, `--pause-ms`, `--levels`, `--audio-encoder`, `--bitrate` | — | The [shared `tts` options](README.md). |
| `--precision <f32\|f16>` | `f32` | `f16` is what the reference itself runs on a GPU and about 16 % faster here, but it is **not** the same run in fewer bits: over 32 steps the small differences compound into a slightly different — equally good — reading. The vocoder and the spectrogram always run in full precision. |
| `--runtime <candle\|burn>` | `candle` | burn needs the `burn` build feature, computes in f32 only, and measured ~1.7× slower here. |
| `--device <cpu\|metal>` | `cpu` | `metal` needs the `metal` build feature and is ~3.5× faster than the CPU. |

## Stress

Russian stress is written with `+` before the stressed vowel — `з+амок` is a
lock, `зам+ок` a castle — in the text to speak and in the reference transcript
alike. This model reads that mark as a real input, and it matters more often
than only for homographs. Measured by transcribing the result back with
[`asr gigaam`](../asr/gigaam.md), three seeds each: `На ф+орзаце была наклеена старая карта.` comes
back with "форзеце" every time, and the same sentence unmarked comes back with
"ф**а**рзаци" every time — with the stress elsewhere the first vowel reduces,
exactly as an unstressed Russian vowel does. Across a wider set of rare words the
mark never made one worse; it also cannot rescue a word the model simply does not
know how to say.

**So the engine marks the text for you, by default.** `--stress auto` runs
[`text stress`](../text/README.md#text-stress--mark-the-stress-in-russian-text)
over the text and over `--ref-text` before synthesis; its model is downloaded on
first use (54 MB) and loaded lazily, on the same `--runtime` and `--device` as
the synthesis. Marks **you** wrote are never moved, so marking part of the text
by hand and leaving the rest to the model is the normal way to work. What the
model ended up reading is reported as `espeech.stressed_text` and
`espeech.stressed_ref_text` — without that, a bad reading is indistinguishable
from a bad mark.

`--stress off` restores the old behavior: the text goes to the model exactly as
you wrote it, and nothing is downloaded.

```bash
# mark it yourself, look at it, fix it, then speak it
trakktor text stress chapter.txt --text > chapter.stressed.txt
trakktor tts espeech --text-file chapter.stressed.txt --stress off \
  --ref-audio narrator.wav --ref-text-file narrator.txt -o chapter.wav
```

## Preparing the reference recording

The recording decides the voice **and the pacing**, and the engine takes it as it
is: it trims silence at the edges, caps the length at 12 s, and leaves everything
in between alone. That last part is worth knowing, because a recording full of
pauses makes the model imitate them. Measured on one recording against a copy of
itself with three 0.6 s pauses inserted (same voice, same words, same transcript),
synthesizing the same sentence:

| Reference | Result | Speech | Silence | Longest pause |
|---|---|---|---|---|
| as recorded | 9.39 s | 8.27 s | 1.12 s | 0.11 s |
| with pauses added | 10.66 s | 7.61 s | **3.05 s** | **0.73 s** |
| pauses shortened first | 9.29 s | 7.83 s | 1.46 s | 0.34 s |

The pauses cost twice over: the model copies them into the reading, and they
inflate the seconds trakktor divides the transcript by to guess the speech rate —
so the same sentence is given 13 % more time, and long text breaks into more
pieces (the same paragraph: ten instead of eight). Intelligibility does not
suffer; the pacing does.

Shortening them is a job for [another trakktor command](../vad/README.md),
which is what the third row above measures:

```bash
trakktor vad cut narrator.wav --max-silence-ms 200 --output-dir prepared
trakktor tts espeech --text-file chapter.md \
  --ref-audio prepared/narrator.speech.wav --ref-text-file narrator.txt \
  -o chapter.wav
```

`--max-silence-ms` shortens the long gaps instead of removing them outright —
speech with no pauses at all sounds wrong as a reference too.

What makes a good reference, then:

- **5–12 seconds**, one speaker, no music or background noise. Anything past 12 s
  is not used.
- **No long pauses inside**, per the table above.
- **The shorter the reference, the more text fits in one piece**: the budget is
  roughly `bytes per second × (22 s − reference seconds)`, so a 6-second
  reference carries noticeably more text per utterance than an 11-second one.
- **A transcript that matches word for word**, with `+` where stress matters. It
  is not just for alignment — it is also the ruler trakktor measures the speech
  rate with, so a transcript missing half the words distorts the pacing.
- **Ending on a phrase boundary.** A recording cut mid-word invites the model to
  finish that word.

**Long text** works exactly as [the shared `tts` behavior](README.md)
describes (paragraphs, `--pause-ms`, `--levels`), with
one difference worth knowing: a single utterance has to fit a ~22-second window
that the reference recording eats into, so a long reference means short pieces.
The engine reports what it used:

```json
{
  "output": "chapter.wav", "format": "wav", "sample_rate": 24000,
  "duration": 9.42, "chunks": 1,
  "voice": { "kind": "clone", "mode": "icl", "ref_audio": "narrator.wav" },
  "engine": { "name": "espeech", "model": "rl-v2", "runtime": "candle" },
  "espeech": { "frames": 927, "nfe_step": 32, "cfg_strength": 2.0,
               "speed": 1.0, "seed": 0, "ref_seconds": 8.91,
               "stressed_text": "Втор+ая глав+а начин+ается с опис+ания дор+оги.",
               "stressed_ref_text": "В+етер ст+их т+олько к +утру, +и ст+ало сл+ышно р+еку." }
}
```

`ref_seconds` is how much of the reference was actually used; when the
recording ran past the 12-second cap and was trimmed, the block also carries
`"ref_clipped": true`.

The first use of a model downloads its checkpoint (2.7 GB as published) and
converts it in place to what inference actually needs — the moving average of the
weights, 1.35 GB — so that is what the model directory keeps.
