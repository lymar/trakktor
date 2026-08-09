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
```

## Which engine

**`gtcrn`, unless you have a reason not to.**

| | `gtcrn` (the first choice) | `mpsenet` | `unipase` |
|---|---|---|---|
| what it is | a masking network on the spectrum | a transformer on the spectrum with separate magnitude and phase heads | a generative pipeline on a speech encoder's representation |
| parameters | **48 thousand** | 2.3 million | 546 million |
| weights | **580 KB** | 9 MB | 2.17 GB |
| speed | **about a sixtieth of real time, one CPU core** | about four tenths of real time, on a GPU | about half of real time, on a GPU |
| estimates phase | no | **yes, as its own output** | it does not estimate one — it synthesises the wave |
| can put back what is not there | **no** | **no** | yes |
| measured against speech recognisers | yes | **not yet** | yes |

Three clauses, in the order you should ask them.

**Did the recording lose packets?** Then `unipase`, always. A masking network
predicts what to attenuate and multiplies — a hard limit, not a tuning choice:
any mask times digital silence is silence, so neither of the other two can fill
the hole a dropped packet left. Measured, that is the one degradation where
`unipase` is better.

**Is the result going to a speech recogniser, or do you just want it cheap?**
Then `gtcrn`. It costs about a twenty-fifth of the next one up, and of the two
engines the measurement below covers it is both the more helpful where
enhancement helps at all and the gentler on a recording that did not need it.

**Is a person going to listen, and has noise wrecked the recording?** Then
`mpsenet` is worth the cost. It is the only one of the three that estimates
**phase** as a separate output instead of carrying the input's phase over —
which is exactly what a mask cannot do at any size, because multiplying a
spectrum by one number moves magnitude and phase together, and noise damages
phase first.

**Send its output to a person, though, not to a recogniser.** It has not been
through the measurement below, and the spot checks that do exist carry a
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

`mpsenet` has **no column here**, and that is a statement rather than an
omission: it has not been run on this bench. What spot checks there are say it
leaves an ordinary room recording's transcript alone but can cost a third of the
words on a hissy archival one, by removing the band the hiss lives in. Use it
for listening.

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
those replaces the usual feed-forward layer with a bidirectional recurrence. Two
million parameters, but they are applied to every point of the
time-by-frequency grid eight times over — which is where the cost comes from.

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

## Sample rate

All three engines work at 16 kHz, which is all the bandwidth speech needs and
exactly what a speech recogniser wants. `--sample-rate` (default: the
recording's own rate) **resamples** the result — it does not invent a wider
band. A 44.1 kHz file therefore comes back at 44.1 kHz, carrying 16 kHz worth of
speech.

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
great deal of arithmetic per second of audio, so it wants a GPU, and there it is
the one engine in trakktor where **the burn runtime is markedly faster than the
default**: `--runtime burn --device metal` runs at about four tenths of real
time, roughly three times `--runtime candle --device metal`. On the CPU the
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

All three engines are checked against their reference implementations stage by
stage: `gtcrn`'s waveform lands within 4e-7 of it, `mpsenet`'s within 5e-7 (and
its two runtimes agree with each other to 6e-6), `unipase`'s within about 2e-5.

## Credits

Ported from [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) (MIT), from
[MP-SENet](https://github.com/yxlu-0102/MP-SENet) (MIT), and from
[UniPASE](https://github.com/Xiaobin-Rong/unipase) (MIT), which
builds on [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) (MIT),
the Vocos backbone by way of
[WavTokenizer](https://github.com/jishengpeng/WavTokenizer) (MIT), and
[PASE](https://github.com/cisco-open/pase) (Apache-2.0). See
[`docs/acknowledgments.md`](../../acknowledgments.md) and [`NOTICE`](../../../NOTICE).
