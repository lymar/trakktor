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
```

## Which engine

**`gtcrn`, unless you know the recording lost packets.**

| | `gtcrn` (default) | `unipase` |
|---|---|---|
| what it is | a masking network on the spectrum | a generative pipeline on a speech encoder's representation |
| parameters | **48 thousand** | 546 million |
| weights | 580 KB | 2.17 GB |
| speed | **a hundredth of real time, one CPU core** | about half of real time, on a GPU |
| can put back what is not there | **no** | yes |

The rule has one clause, and it is about packet loss. A masking network predicts
what to attenuate and multiplies — which is a hard limit, not a tuning choice:
any mask times digital silence is silence, so it cannot fill the hole a dropped
packet left. Measured, that is the one degradation where `unipase` is better.
Everywhere else `gtcrn` is as good or better, at a hundred and fiftieth of the
cost, and it is the gentler of the two on a recording that did not need
enhancing.

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

The reason is what the model is: a *generative* model that rebuilds speech from
what a speech encoder understood of it, rather than a filter that subtracts
noise. When there is enough signal to understand, that is exactly why it can put
back a band or a lost packet. When there is not, it confabulates — and a made-up
word comes out of the recogniser looking as confident as a real one.

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

**Packet loss** needs no fourth network — and it is the one thing `gtcrn` cannot
do at all. The encoder was pre-trained with spans
of its input masked out, so a hole in a call — the digital silence a dropped
packet leaves — is handed to it as a mask and filled in from the words on either
side. A packet is 20 ms, which at 16 kHz is exactly one frame of the encoder.
This is on by default; `--no-plc` leaves the holes alone. The result says how
much was concealed, so a run that filled in half the recording cannot look like
one that changed nothing.

## Sample rate

The pipeline works at 16 kHz, which is all the bandwidth speech needs and
exactly what a speech recogniser wants. `--sample-rate` (default: the
recording's own rate) **resamples** the result — it does not invent a wider
band. A 44.1 kHz file therefore comes back at 44.1 kHz, carrying 16 kHz worth of
speech.

## Output

```sh
trakktor enhance unipase meeting.wav --pretty
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
the ceiling — and what a generative model produces is noise-like in its low bits
by construction. A size guard catches that and refuses to write the file rather
than leave a valid one two orders of magnitude too big, so `-o out.flac` will
often fail here with `unsupported_encoding`. Write WAV, or go through `ffmpeg`
with a different extension.

## Cost

`gtcrn` downloads **580 KB** once and runs on one CPU core faster than the audio
plays. It has no `--device` or `--runtime`: forty-eight thousand parameters are
not work a GPU can help with, and a second tensor backend would be a second
implementation of the same thing.

`unipase` downloads **2.17 GB** once, converted into a single file in the model
directory (`~/.trakktor` by default). Running it is half a billion parameters
over every eight seconds of audio, so `--device metal` is worth having; on a
CPU, expect a long recording to take a while. Both runtimes are available there
— `--runtime candle` (default) and `--runtime burn` — and they agree with the
reference and with each other to within arithmetic noise. Precision is `f32` by
default and is the only mode the port is verified in; `--precision f16` is a
candle-only speed option and is a different computation of the same model.

Both engines are checked against their reference implementations stage by stage:
`gtcrn`'s waveform lands within 4e-7 of it, `unipase`'s within 2e-5.

## Credits

Ported from [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) (MIT), and from
[UniPASE](https://github.com/Xiaobin-Rong/unipase) (MIT), which
builds on [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) (MIT),
the Vocos backbone by way of
[WavTokenizer](https://github.com/jishengpeng/WavTokenizer) (MIT), and
[PASE](https://github.com/cisco-open/pase) (Apache-2.0). See
[`docs/acknowledgments.md`](../../acknowledgments.md) and [`NOTICE`](../../../NOTICE).
