# `enhance` — clean up a speech recording

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

`trakktor enhance` takes a damaged speech recording and returns a repaired one.
It is the third shape of speech work in the toolbox: [`asr`](../asr/README.md)
turns a recording into text, [`tts`](../tts/README.md) turns text into a
recording, and this turns a recording into a better recording.

```sh
trakktor enhance gtcrn call.m4a                    # → call.enhanced.wav
trakktor enhance gtcrn call.m4a -o clean.wav       # pick the output
trakktor enhance unipase call.m4a --device metal   # the generative engine
trakktor enhance mpsenet talk.mp3 --runtime burn --device metal   # phase too
trakktor enhance resemble-denoise lecture.wav      # 44.1 kHz, band kept
trakktor enhance resemble-enhance lecture.wav      # rebuilt, not filtered
```

(The `--device metal` and `--runtime burn` lines need a build with the `metal`
feature — for the candle runtime — or the `burn` feature; a default build
accepts the flags and fails at run time.)

## Which engine

**`gtcrn`, unless you have a reason not to.**

| | `gtcrn` (the first choice) | `mpsenet` | `unipase` | `resemble-denoise` | `resemble-enhance` |
|---|---|---|---|---|---|
| what it is | a masking network on the spectrum | a transformer on the spectrum with separate magnitude and phase heads | a generative pipeline on a speech encoder's representation | a masking UNet on the spectrum | a generative pipeline: mel → flow → vocoder |
| parameters | **48 thousand** | 2.26 million | 546 million | 10.8 million | 346 million |
| download | **580 KB** | 9 MB | 2.17 GB | 713 MB (shared) | 713 MB (shared) |
| **works at** | 16 kHz | 16 kHz | 16 kHz | **44.1 kHz** | **44.1 kHz** |
| speed | **about a sixtieth of real time, one CPU core** | about four tenths of real time, on a GPU | about half of real time, on a GPU | **a fifth of real time** on a GPU with the burn runtime | **about three times real time** on a GPU, eight on four CPU cores |
| what it does to phase | moves it with the magnitude | **estimates it outright** | synthesises the wave | **rotates it, apart from the magnitude** | synthesises the wave |
| can put back what is not there | **no** | **no** | yes | **no** | yes — including words |
| measured against speech recognisers | yes | **not yet** | yes | **not yet** | **not yet** |

Read the questions in this order.

**Is the recording full-band, and do you want to keep it?** Then one of the
`resemble` pair, and there is no alternative: the other three work at 16 kHz and
return 16 kHz whatever rate you ask them to write. A lecture, an interview, a
podcast, anything a person will listen to end to end loses its top two octaves
otherwise.

**Did the recording lose packets?** Then `unipase`, always. A masking network
predicts what to attenuate and multiplies — a hard limit, not a tuning choice:
any mask times digital silence is silence, so no other engine here can fill the
hole a dropped packet left. Measured, that is the one degradation where
`unipase` is better.

**Is the result going to a speech recogniser, or do you just want it cheap?**
Then `gtcrn`. It costs about a twenty-fifth of the next one up, and of the two
engines the measurement below covers it is both the more helpful where
enhancement helps at all and the gentler on a recording that did not need it.

**Is a person going to listen, and has noise wrecked the recording?** Then three
engines are worth their cost, for three different reasons:

- **`mpsenet`** if what needs repair is the **phase**. It is the only one that
  estimates phase as its own output instead of carrying the input's over —
  exactly what a mask cannot do at any size, because multiplying a spectrum by
  one number moves magnitude and phase together, and noise damages phase first;
- **`resemble-denoise`** if the recording is **wide-band**. It also treats phase
  apart from magnitude, but by **rotating** the phase that came in rather than
  estimating one from nothing;
- **`resemble-enhance`** if the recording needs **rebuilding** rather than
  filtering. It does not filter at all: it reads a mel, walks a flow model from
  noise to a description of what clean speech would look like, and builds a
  waveform out of a second lot of noise. Nothing of the input waveform reaches
  the output — which is why it can widen a band, unpick reverberation and repair
  a clipped syllable, and why **it can put in a word that was never said**.

**Send all three to a person, not to a recogniser.** None has been through the
measurement below. For `mpsenet` the spot checks that exist carry a specific
warning: on a recording where hiss dominates the top of the band — an archival
transfer, say — it takes the hiss and the band with it, dropping everything
above 2.5 kHz by twenty decibels and more. The fricatives go with it, and a
recogniser lost a third of its words. On two ordinary room recordings the
transcript came through untouched. For a transcript, use `gtcrn`.

## Read this before you use it

**This repairs damage; it does not improve a recording that is already good.**
Measured against two independent speech recognisers on Russian material — a
Conformer and a Zipformer transducer, so that what follows is a property of the
enhancement and not of one recogniser:

| What is wrong with the recording | `gtcrn` | `unipase` |
|---|---|---|
| telephone band + noise + dropped packets | **clearly better** | **clearly better** |
| a live, reverberant room | **better** | slightly better |
| dropped packets alone | worse | **slightly better** |
| competing voices | worse | depends on the recogniser |
| a narrow telephone band alone | worse | worse |
| heavy broadband noise (around 0 dB SNR) | about the same | **much worse** — it starts inventing words |
| nothing — the recording is fine | **worse** | **worse**, by more |

**Three engines have no column here**, and that is a statement rather than an
omission: `mpsenet` and the two `resemble` ones have not been run on this bench.
For `mpsenet` the spot checks that exist say it leaves an ordinary room
recording's transcript alone but can cost a third of the words on a hissy
archival one, by removing the band the hiss lives in.

For the `resemble` pair there is a further problem with the bench itself: it
works at 16 kHz, and running a 44.1 kHz engine through it would measure the
resampling rather than the engine — everything the engine was taken for gets
thrown away before the recogniser sees it. And for `resemble-enhance` the
question is different in kind: it synthesises speech rather than cleaning it, so
a low error rate there might mean it made up something fluent. All three are for
listening.

The reason for the column that is there is what the model is: a *generative*
model that rebuilds speech from what a speech encoder understood of it, rather
than a filter that subtracts noise. When there is enough signal to understand,
that is exactly why it can put back a band or a lost packet. When there is not,
it confabulates — and a made-up word comes out of the recogniser looking as
confident as a real one. One more caution: the measurements above are on
Russian, which is not among the languages UniPASE's authors list for it
(English, Chinese, Spanish, French, German) — on a covered language the
confabulation line may sit elsewhere.

So: run it because you know the recording is damaged, not as a matter of course.
It is deliberately not wired into `asr` as an automatic preprocessing step.

## What `gtcrn` does

It predicts a complex ratio mask on the spectrum and multiplies. Forty-eight
thousand parameters buy that through four economies, each worth naming:

- the 192 bins above 2 kHz are folded into 64 equivalent-rectangular bands
  before the network sees them, and spread back out at the end;
- every band is concatenated with its two neighbours before a convolution one
  band wide, so a narrow kernel still sees a neighbourhood;
- half the channels skip each temporal block entirely and are woven back in
  afterwards, so only half the width is ever convolved;
- the recurrences are cut in two and run as independent halves.

The temporal blocks are causal — the only padding on the time axis is at the
front — but its recurrences remember the whole recording, so a long one is
processed in chunks that **continue** one another: the state carries across, and
where the boundaries fall is not visible in the result. A test asserts that, bit
for bit.

## What `mpsenet` does

It has **two output heads instead of one**, and that is the whole point. The
input is not a complex spectrum but a pair of planes — a magnitude raised to the
power 0.3, and a phase. A shared trunk reads both, and then one head predicts a
gain per frequency bin (a learned sigmoid, which can amplify a bin up to twice
as well as attenuate it) while the other predicts **two components** whose
arctangent is the output phase. The phase that comes out is not the phase that
went in.

A mask cannot do that at any size. Multiplying a spectrum by a complex number
moves magnitude and phase together, so attenuating a bin necessarily rotates it,
and a phase that noise has scrambled can never be put back. This is the one
operation neither of the other two engines has.

The trunk is a dense convolutional encoder and four two-stage blocks. Each block
runs one transformer along **frequency** and one along **time**, and each of
those replaces the usual feed-forward layer with a bidirectional recurrence.
Two and a quarter million (2.26 M) parameters, but they are applied to every
point of the time-by-frequency grid eight times over — which is where the cost
comes from.

Because the attention spans the whole window along both axes, this network
cannot stream and a long recording has to be cut. Upstream never had to answer
how: it runs one utterance in one pass. So the windows here were **chosen by
measurement** — a second of speech was run many times with the window placed
differently around it. The result turned out to depend on *which* window a
moment falls in (by 1 % to 25 %, because the network's normalizations take their
statistics from the whole window) and hardly at all on *where in the window* it
falls, past the first quarter-second. So the overlap is only as long as it needs
to be to hide the seam: eight-second windows sharing one second, cross-faded,
and the recording is processed 1.14 times rather than twice.

## What `unipase` does

Four networks, of which three run:

1. a fine-tuned **WavLM** encoder reads the recording and is tapped at two
   depths — an early layer that still carries what the room and the microphone
   did, and a deep one that carries what is being said;
2. an **adapter** works out what the early layer *should* have been if the
   recording had been clean;
3. a **vocoder** turns that back into sound.

Because the repair happens in the encoder's representation rather than on the
spectrum, it can put back what is missing and not only take away what is not
wanted.

**Packet loss** needs no fourth network — and it is the one thing neither
masking engine can do at all. The encoder was pre-trained with spans
of its input masked out, so a hole in a call — the digital silence a dropped
packet leaves — is handed to it as a mask and filled in from the words on either
side. A packet is 20 ms, which at 16 kHz is exactly one frame of the encoder.
This is on by default; `--no-plc` leaves the holes alone. The result says how
much was concealed, so a run that filled in half the recording cannot look like
one that changed nothing.

## What the `resemble` pair does

Two networks from one project, published in one checkpoint — which is why one
713 MB download covers both, whichever you asked for.

**`resemble-denoise`** is a UNet over the spectrum, treating time and frequency
alike as an image: four halvings down and four doublings back up, sixteen
channels at the top and two hundred and fifty-six at the bottom. Three planes in
(a magnitude and the cosine and sine of its phase) and three out: a gain, and
**two numbers that name a rotation** of that phase. So it sits exactly between
the other two masking engines — `gtcrn` moves magnitude and phase together
because a complex multiply cannot do otherwise, `mpsenet` estimates a phase from
nothing, and this one turns the phase that came in by an angle it decides
separately from the gain.

**`resemble-enhance`** has no waveform on its path at all:

1. the recording becomes a mel spectrogram — a mix of the denoiser's output
   and the recording as it is, in the proportion `--denoise` names (at its
   default of 1.0 the flow sees only the denoised one; at 0 the denoiser does
   not run at all, which is also the cheapest setting);
2. an autoencoder turns that into a 64-wide latent, and noise is mixed into it
   in the proportion `--temperature` names;
3. a **flow model** is integrated from that starting point to a description of
   what clean speech would look like — `--nfe` evaluations of it, spent by
   `--solver`, and the engine's cost is very nearly proportional to that count;
4. the autoencoder's decoder expands the result, and a vocoder builds a waveform
   out of **a second lot of noise**, steered by it.

That is why it can do what a filter cannot, and why it can invent. It is also
why it is not deterministic upstream: the reference draws both lots of noise
from a generator its command line never seeds, so two runs of it give two
different files. This port seeds its own, so a run repeats; `--seed` picks a
different one.

The knobs and their defaults:

| flag | default | range | what it sets |
|---|---|---|---|
| `--nfe` | 64 | 1 to 128 | flow evaluations per chunk; the cost is very nearly proportional |
| `--solver` | `midpoint` | `euler`, `midpoint`, `rk4` | evaluations spent per step: one, two, four |
| `--temperature` | 0.5 | 0 to 1 | how much of the walk's starting point is noise |
| `--denoise` | 1.0 | 0 to 1 | how much of the denoiser's output the flow is conditioned on |
| `--seed` | 0 | — | fixes the noise this engine draws, and with it the run |

## Sample rate

**`--sample-rate` means two different things, and which one depends on the
engine.** `gtcrn`, `mpsenet` and `unipase` work at 16 kHz, which is all the
bandwidth speech needs and exactly what a speech recogniser wants; for them
`--sample-rate` (default: the recording's own rate) **resamples** the result and
does not invent a wider band. A 44.1 kHz file comes back at 44.1 kHz, carrying
16 kHz worth of speech.

The `resemble` pair works at 44.1 kHz, so up to that rate it is real bandwidth.
For `resemble-enhance` it is more than that: the vocoder synthesises rather than
filters, so it will put content above what the recording carried — a 32 kHz
source comes back with something above 16 kHz. Whether that is welcome is a
question for the listener, not for the engine.

44.1 kHz is also the ceiling. A recording above it — 48 kHz, say — is resampled
down to 44.1 kHz before the networks run, and the result is resampled back up
to the recording's own rate (the `--sample-rate` default), so it comes back at
48 kHz carrying nothing a 44.1 kHz signal could not.

## Output

```sh
trakktor enhance unipase meeting.wav --device metal --pretty
```

```json
{
  "output": "meeting.enhanced.wav",
  "format": "wav",
  "sample_rate": 16000,
  "source_sample_rate": 16000,
  "duration": 20.0,
  "windows": 4,
  "packet_loss": { "enabled": true, "frames": 0, "seconds": 0.0 },
  "engine": {
    "name": "unipase",
    "model": "unipase",
    "runtime": "candle",
    "device": "metal"
  }
}
```

Every engine emits this same envelope, `packet_loss` and the
`engine.runtime`/`engine.device` fields included; for `gtcrn`, which has
neither flag, those two always read `candle` and `cpu`.

`--text` prints the path and a one-line summary instead. The container follows
the output extension: `wav` and `flac` are written directly, anything else goes
through an installed `ffmpeg` (`--bitrate` applies there).

**A note on FLAC.** The built-in FLAC encoder codes loud, noise-like 24-bit
material catastrophically badly — hundreds of bytes per sample where three is
the ceiling — and enhanced speech is noise-like in its low bits by
construction. A size guard catches that and refuses to write the file rather
than leave a valid one two orders of magnitude too big, so `-o out.flac` will
often fail here with `unsupported_encoding`. Write WAV, or go through `ffmpeg`
with a different extension.

## Cost

`gtcrn` downloads **580 KB** once and runs on one CPU core faster than the audio
plays. It has no `--device` or `--runtime`: forty-eight thousand parameters are
not work a GPU can help with, and a second tensor backend would be a second
implementation of the same thing. `--model` names the checkpoint (default
`dns3`) or takes a directory of converted weights.

`mpsenet` downloads **9 MB** once — `--model dns` (the default, trained on the
DNS Challenge data) or `--model vb` (VoiceBank+DEMAND). It is small but does a
great deal of arithmetic per second of audio, so it wants a GPU, and it is one
of the engines where **burn on a GPU is markedly faster than the default**:
`--runtime burn --device metal` runs at about four tenths of real time, roughly
three times `--runtime candle --device metal`. On the CPU the
order reverses and burn is much the slower, so the pairing matters. `f32` is the
default and the only verified mode; `--precision f16` works here and buys no
speed at all.

`unipase` downloads **2.17 GB** once, converted into a single file in the model
directory (`~/.trakktor` by default); its `--model` likewise names the
checkpoint (default `unipase`) or a local directory. Running it is half a
billion parameters over every eight seconds of audio, so `--device metal` is
worth having (the default is `cpu`; the build needs the `metal` feature). Both
runtimes are available there — `--runtime candle` (default) and `--runtime
burn`, behind the `burn` build feature — and they agree with the reference and
with each other to within arithmetic noise. Precision is `f32` by default and
is the only mode the port is verified in; `--precision f16` is a candle-only
speed option and is a different computation of the same model.

The `resemble` pair downloads **713 MB** once — one checkpoint holding both
networks, because that is how upstream publishes them — and converts it into two
files, so `resemble-denoise` loads ten million parameters rather than three
hundred and forty-six. `--model` names it (default `resemble`) or takes a
directory.

`resemble-denoise` runs at about half of real time on four CPU cores or on
Metal, and at **a fifth** of real time with `--runtime burn --device metal`,
which is the combination to use if you have it — the second engine in the
toolbox where burn on a GPU is the fastest thing available. (Measure it on the
*second* run: wgpu compiles and tunes a kernel for every new shape, and the
first run of this network paid twelve times the settled cost.)

`resemble-enhance` is the most expensive engine here by a wide margin — the flow
model alone runs `--nfe` times over every chunk. `--device metal` is about two
and a half times faster than the CPU and worth having; the default is
nevertheless `cpu`, because loading three hundred and forty-six million
parameters into GPU buffers takes about a gigabyte and a half out of memory the
rest of the machine shares. If the GPU has room, ask for it.

`resemble-enhance` takes `--runtime burn` as well, behind the `burn` build
feature and computing in f32 only. For both of the pair, `--precision` is `f32`
by default and the only mode the port is verified in; `f16` is a candle-only
speed option.

Every engine is checked against its reference implementation stage by stage:
`gtcrn`'s waveform lands within 4e-7 of it, `mpsenet`'s within 5e-7 (and its two
runtimes agree with each other to 6e-6), `unipase`'s within about 2e-5,
`resemble-denoise`'s within 3e-6. `resemble-enhance` is a special case, because
the reference cannot repeat its own output: fed the reference's own two draws of
noise, the port's waveform lands within 4e-3 — and its vocoder alone, given the
reference's own conditioning, within 2e-6. The gap between those two numbers is
the vocoder amplifying what it is conditioned on, which it does about twentyfold.

## Credits

Ported from [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) (MIT), from
[MP-SENet](https://github.com/yxlu-0102/MP-SENet) (MIT), from
[UniPASE](https://github.com/Xiaobin-Rong/unipase) (MIT), which
builds on [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) (MIT),
the Vocos backbone by way of
[WavTokenizer](https://github.com/jishengpeng/WavTokenizer) (MIT), and
[PASE](https://github.com/cisco-open/pase) (Apache-2.0); and from
[resemble-enhance](https://github.com/resemble-ai/resemble-enhance) (MIT), whose
vocoder takes its anti-aliased activation from
[BigVGAN](https://github.com/NVIDIA/BigVGAN) (MIT) and
[alias-free-torch](https://github.com/junjun3518/alias-free-torch) (Apache-2.0).
See [`docs/acknowledgments.md`](../../acknowledgments.md) and
[`NOTICE`](../../../NOTICE).
