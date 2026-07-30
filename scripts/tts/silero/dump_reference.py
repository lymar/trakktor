#!/usr/bin/env python3
"""Dump Silero v5 reference intermediates as the golden trace for the port.

Runs the published `torch.package` — the acoustic model, the vocoder and the
inverse transform, CPU, fp32 — on one sentence and writes every stage the Rust
port is compared against. The pipeline has no sampling anywhere, so the whole
trace is reproducible and every stage can be checked, not just the last one.

The top-level forward is re-implemented here over the reference's own scripted
submodules, so the intermediates are observable; `--check` then runs the
reference module itself and reports the largest difference, which must be zero.
That check is what catches a mis-transcribed rule — the duration tail clamps in
particular are easy to get subtly wrong.

Writes to an output directory (raw little-endian f32 unless stated):

  ids.bin           symbol ids of the sentence, i32
  dur_log.bin       the duration head's output, one per symbol
  dur.bin           the durations actually used, in frames, i32
  pitch_raw.bin     the pitch head's output, one per symbol
  pitch.bin         the pitch after the reference's own thresholding
  encoded.bin       encoder + speaker + pitch, [symbols, 128]
  expanded.bin      after the length regulator, [frames, 128]
  mel.bin           the mel spectrogram, [frames, 192]
  backbone.bin      the vocoder backbone's output, [frames, 512]
  magnitude.bin     [frames, 1201]
  phase.bin         [frames, 1201]
  wave_48000.bin    the waveform at 48 kHz
  wave_24000.bin    the same run resampled by the model's own filterbank
  wave_8000.bin     likewise
  dims.json         shapes, the speaker id, and the text
  speech.wav        the 48 kHz waveform, to listen to

Run inside a venv with torch installed:
  python scripts/tts/silero/dump_reference.py \
      --package /path/to/v5_cis_base.pt --speaker ru_zhadyra \
      --out tmp/silero/golden --check
"""

import argparse
import json
import struct
import sys
import wave
from pathlib import Path

import torch

# Upstream's own published example sentence (models.yml), stressed the way the
# model expects.
DEFAULT_TEXT = "В н+едрах т+ундры в+ыдры в г+етрах т+ырят в в+ёдра +ядра к+едров."


def write(path, tensor, dtype="f"):
    values = tensor.detach().flatten().tolist()
    path.write_bytes(struct.pack(f"<{len(values)}{dtype}", *values))


def write_wav(path, wave_data, sample_rate):
    pcm = (wave_data.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def durations(jit, sequence, speaker_ids, mask, rate):
    """The duration head plus every rule the reference applies to its output."""
    log_dur = jit.dur_predictor(sequence, speaker_ids, mask, 1.0, None)
    dur = torch.exp(log_dur) - 1
    dur[dur < 0] = 0.0
    dur = torch.round(dur)
    if dur[0][0] > 5:
        dur[0][0] = 5
    dur = torch.round(dur / rate)
    # The reference clamps the head and the tail of the utterance: the leading
    # separator, the trailing end-of-speech symbol, and the punctuation before
    # it, whose predicted length it overrides outright.
    if dur[0][0] > 5:
        dur[0][0] = 5
    if dur[0][-1] > 7:
        dur[0][-1] = 7
    dur[0][-2] = 13
    if dur[0][-3] > 13:
        dur[0][-3] = 13
    return log_dur, dur


def pitch(jit, sequence, speaker_ids, mask, coefficients):
    """The pitch head and the reference's coefficient pass over it."""
    raw = jit.pitch_predictor(sequence, speaker_ids, mask, 1.0)
    shaped = jit.pitch_predictor.update_pitch_coef(
        raw.clone(), coefficients[0].unsqueeze(0), speaker_ids
    ) if hasattr(jit.pitch_predictor, "update_pitch_coef") else None
    if shaped is None:
        shaped = update_pitch_coef(jit, raw.clone(), coefficients[0].unsqueeze(0),
                                   speaker_ids)
    return raw, shaped


def update_pitch_coef(jit, norm_pitch, coefficient, speaker_ids):
    """`MultiTTSModel.update_pitch_coef`, which lives on the top-level module."""
    norm_pitch[abs(norm_pitch) < 0.001] = 0.0
    scaled = norm_pitch * coefficient
    coefficient = coefficient.clone()
    coefficient[:, (coefficient[0] == 0).nonzero()[:, 0]] = 1.0
    shift = (coefficient - 1.0) * jit.mean_std_coef[int(speaker_ids.item())]
    shift[scaled[0] == 0] = 0
    return scaled + shift


def encode(jit, sequence, speaker_ids, mask, pitch_hat, dur):
    """`JitMultiForward.forward`, stage by stage."""
    tacotron = jit.tacotron
    embedded = tacotron.embedding(sequence)
    speaker = tacotron.speaker_embedding(speaker_ids)
    speaker = speaker.unsqueeze(1).repeat(1, embedded.size(1), 1)
    encoded = tacotron.encoder(embedded, mask, None) + speaker
    projected = tacotron.pitch_proj(pitch_hat).transpose(1, 2)
    encoded = encoded + projected * tacotron.pitch_strength
    expanded = tacotron.len_reg(encoded, dur.clone())
    decoded = tacotron.decoder(expanded, None, None)
    mel = tacotron.lin(decoded).transpose(1, 2)
    return encoded, expanded, mel


def vocode(jit, mel):
    """`JitV4.forward` up to the point the inverse transform takes over."""
    backbone = jit.vocoder.backbone(mel)
    spectrum = jit.vocoder.head.out(backbone).transpose(1, 2)
    magnitude, phase = torch.chunk(spectrum, 2, 1)
    magnitude = torch.clamp(torch.exp(magnitude), max=100.0)
    return backbone, magnitude, phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--speaker", default="ru_zhadyra")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check", action="store_true",
                        help="also run the reference module and compare")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    text = args.text_file.read_text(encoding="utf-8").strip() if args.text_file \
        else args.text

    package = torch.package.PackageImporter(str(args.package))
    model = package.load_pickle("tts_models", "model")
    part = model.packages[model.speaker_to_package[args.speaker]]
    jit = part.models[part.speaker_to_model[args.speaker]]
    speaker_id = part.speaker_to_ids[part.speaker_to_model[args.speaker]][args.speaker]

    language = args.speaker.split("_")[0] if part.ext_alph else None
    sentences, _, breaks, rates, pitches, speaker_ids = part.prepare_tts_model_input(
        text, ssml=False, speaker_ids=[speaker_id], lang=language)
    sequence, symbol_durs, rate, pitch_coefficients = part.merge_batch_model(
        sentences, breaks, rates, pitches)
    assert not symbol_durs, "this dump does not cover explicit pauses"

    mask = (sequence == 0)
    log_dur, dur = durations(jit, sequence, speaker_ids, mask, rate)
    pitch_raw = jit.pitch_predictor(sequence, speaker_ids, mask, 1.0)
    pitch_hat = update_pitch_coef(jit, pitch_raw.clone(),
                                  pitch_coefficients[0].unsqueeze(0), speaker_ids)
    encoded, expanded, mel = encode(jit, sequence, speaker_ids, mask, pitch_hat, dur)
    backbone, magnitude, phase = vocode(jit, mel)
    waves = {rate_hz: jit.vocoder(mel, rate_hz, 0.0, True)[0]
             for rate_hz in (48000, 24000, 8000)}

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write(out / "ids.bin", sequence.to(torch.int32), "i")
    write(out / "dur_log.bin", log_dur)
    write(out / "dur.bin", dur.to(torch.int32), "i")
    write(out / "pitch_raw.bin", pitch_raw)
    write(out / "pitch.bin", pitch_hat)
    write(out / "encoded.bin", encoded)
    write(out / "expanded.bin", expanded)
    write(out / "mel.bin", mel.transpose(1, 2))
    write(out / "backbone.bin", backbone)
    write(out / "magnitude.bin", magnitude.transpose(1, 2))
    write(out / "phase.bin", phase.transpose(1, 2))
    for rate_hz, samples in waves.items():
        write(out / f"wave_{rate_hz}.bin", samples)
    write_wav(out / "speech.wav", waves[48000], 48000)
    (out / "text.txt").write_text(text + "\n", encoding="utf-8")
    (out / "dims.json").write_text(json.dumps({
        "text": text,
        "speaker": args.speaker,
        "speaker_id": speaker_id,
        "symbols": int(sequence.size(1)),
        "frames": int(mel.size(2)),
        "mel_channels": int(mel.size(1)),
        "bins": int(magnitude.size(1)),
        "samples": {str(k): int(v.numel()) for k, v in waves.items()},
        "package": args.package.name,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    if args.check:
        reference = jit(sequence, speaker_ids, 48000, None, rate,
                        pitch_coefficients, None, None, "cpu")[0][0]
        difference = (reference - waves[48000]).abs().max().item()
        print(f"re-implementation vs reference: max |diff| = {difference:g}")
        if difference != 0.0:
            sys.exit("the re-implementation does not reproduce the reference")
    print(f"wrote {out} — {sequence.size(1)} symbols, {mel.size(2)} frames, "
          f"{waves[48000].numel()} samples")


if __name__ == "__main__":
    main()
