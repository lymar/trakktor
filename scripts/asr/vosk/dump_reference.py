#!/usr/bin/env python3
"""Dump Vosk (Zipformer2 transducer) reference traces for the Rust port.

Runs the reference stack — kaldi-native-fbank features and onnxruntime over
the published ONNX export — so the port can be validated against exact
intermediates. Decoding (greedy and modified beam search) reimplements the
sherpa-onnx decoders in numpy and is cross-checked against the sherpa-onnx
package itself when it is installed.

Writes into an output directory:

  clip.pcm.bin        16 kHz mono f32 LE samples fed to everything below
  clip.fbank.bin      log-mel features [T, 80], raw f32 LE (dither 0)
  clip.encoder.bin    projected encoder output [T', 512], raw f32 LE
                      (offline: one pass; streaming: chunk outputs concat)
  clip.greedy.json    {text, tokens, frames} of greedy search
  clip.beam.json      {text, tokens, frames} of modified beam search (10)

Run inside the vosk venv:
  /Users/sergey/trakktor/reference/.venv-vosk/bin/python \
      scripts/asr/vosk/dump_reference.py --model tmp/vosk/models/ru \
      --audio tmp/sample.mp3 --offset 40 --duration 12 \
      --out tmp/vosk/golden/ru
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

import kaldi_native_fbank as knf


def load_pcm(path: str, offset: float, duration: float) -> np.ndarray:
    """Decodes audio to 16 kHz mono f32 via ffmpeg."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-ss", str(offset), "-t", str(duration),
        "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", "16000", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy()


def compute_fbank(samples: np.ndarray) -> np.ndarray:
    """The sherpa-onnx transducer fbank: 80 bins, dither 0."""
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, samples.tolist())
    fbank.input_finished()
    frames = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
    return np.asarray(frames, dtype=np.float32)


class Transducer:
    def __init__(self, model_dir: Path):
        so = ort.SessionOptions()
        self.encoder = ort.InferenceSession(
            str(model_dir / "encoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(
            str(model_dir / "decoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.joiner = ort.InferenceSession(
            str(model_dir / "joiner.onnx"), so, providers=["CPUExecutionProvider"])
        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.streaming = "decode_chunk_len" in meta
        self.meta = meta
        dmeta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(dmeta.get("context_size", 2))
        self.vocab_size = int(dmeta.get("vocab_size", 500))

    # ---- encoder ---------------------------------------------------------

    def encode_offline(self, feats: np.ndarray) -> np.ndarray:
        x = feats[None].astype(np.float32)
        names = [i.name for i in self.encoder.get_inputs()]
        lens_dtype = np.int64
        for i in self.encoder.get_inputs():
            if i.name == "x_lens" and "int32" in i.type:
                lens_dtype = np.int32
        out = self.encoder.run(
            None, {"x": x, "x_lens": np.array([feats.shape[0]], dtype=lens_dtype)}
        )
        assert "x" in names
        return out[0][0]  # [T', joiner_dim]

    def init_states(self) -> dict:
        states = {}
        for i in self.encoder.get_inputs():
            if i.name == "x":
                continue
            shape = [1 if isinstance(d, str) else d for d in i.shape]
            dtype = np.int64 if "int64" in i.type else np.float32
            states[i.name] = np.zeros(shape, dtype=dtype)
        return states

    def encode_streaming(self, feats: np.ndarray) -> np.ndarray:
        T = int(self.meta["T"])
        shift = int(self.meta["decode_chunk_len"])
        states = self.init_states()
        out_names = [o.name for o in self.encoder.get_outputs()]
        outs = []
        start = 0
        while start + T < feats.shape[0]:
            window = feats[start : start + T][None].astype(np.float32)
            result = self.encoder.run(None, {"x": window, **states})
            by_name = dict(zip(out_names, result))
            outs.append(by_name["encoder_out"][0])
            states = {
                name: by_name["new_" + name]
                for name in states
            }
            start += shift
        return np.concatenate(outs, axis=0)

    # ---- transducer head -------------------------------------------------

    def decoder_out(self, context: list[int]) -> np.ndarray:
        y_dtype = np.int64
        for i in self.decoder.get_inputs():
            if "int32" in i.type:
                y_dtype = np.int32
        y = np.array([context], dtype=y_dtype)
        return self.decoder.run(None, {"y": y})[0]  # [1, joiner_dim]

    def joint(self, enc: np.ndarray, dec: np.ndarray) -> np.ndarray:
        return self.joiner.run(
            None,
            {"encoder_out": enc.astype(np.float32), "decoder_out": dec},
        )[0]

    def greedy(self, encoder_out: np.ndarray, unk_id: int):
        ctx = [-1] * (self.context_size - 1) + [0]
        dec = self.decoder_out(ctx)
        tokens, frames = [], []
        for t in range(encoder_out.shape[0]):
            logits = self.joint(encoder_out[t : t + 1], dec)[0]
            y = int(np.argmax(logits))
            if y != 0 and y != unk_id:
                tokens.append(y)
                frames.append(t)
                ctx = (ctx + [y])[-self.context_size :]
                dec = self.decoder_out(ctx)
        return tokens, frames

    def modified_beam(self, encoder_out: np.ndarray, unk_id: int, beam: int):
        context = self.context_size
        hyps = [
            {
                "ys": [-1] * (context - 1) + [0],
                "frames": [],
                "log_prob": 0.0,
            }
        ]
        memo: dict[tuple, np.ndarray] = {}
        for t in range(encoder_out.shape[0]):
            scores = []
            for hyp in hyps:
                key = tuple(hyp["ys"][-context:])
                if key not in memo:
                    memo[key] = self.decoder_out(list(key))
                logits = self.joint(encoder_out[t : t + 1], memo[key])[0]
                logits = logits.astype(np.float32)
                m = logits.max()
                logsoft = logits - (m + np.log(np.exp(logits - m).sum()))
                scores.append(hyp["log_prob"] + logsoft.astype(np.float64))
            flat = np.concatenate(scores)
            k = min(beam, flat.shape[0])
            order = np.argsort(-flat, kind="stable")[:k]
            new_hyps: list[dict] = []
            index: dict[tuple, int] = {}
            for cand in order:
                h, token = divmod(int(cand), self.vocab_size)
                hyp = hyps[h]
                ys = list(hyp["ys"])
                frames = list(hyp["frames"])
                if token != 0 and token != unk_id:
                    ys.append(token)
                    frames.append(t)
                lp = float(flat[cand])
                key = tuple(ys)
                if key in index:
                    slot = new_hyps[index[key]]
                    hi, lo = max(slot["log_prob"], lp), min(slot["log_prob"], lp)
                    slot["log_prob"] = hi + np.log1p(np.exp(lo - hi))
                else:
                    index[key] = len(new_hyps)
                    new_hyps.append(
                        {"ys": ys, "frames": frames, "log_prob": lp}
                    )
            hyps = new_hyps
        best = max(hyps, key=lambda h: h["log_prob"] / len(h["ys"]))
        return [int(y) for y in best["ys"][context:]], best["frames"]


def load_tokens(path: Path) -> dict[int, str]:
    table = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        piece, idx = line.rsplit(" ", 1)
        table[int(idx)] = piece
    return table


def decode_text(tokens: list[int], table: dict[int, str]) -> str:
    text = "".join(table[t] for t in tokens)
    return text.replace("▁", " ").strip()


def sherpa_check(model_dir: Path, samples: np.ndarray, streaming: bool):
    """Returns sherpa-onnx's text for the same audio, if available."""
    try:
        import sherpa_onnx
    except ImportError:
        return None
    kw = dict(
        encoder=str(model_dir / "encoder.onnx"),
        decoder=str(model_dir / "decoder.onnx"),
        joiner=str(model_dir / "joiner.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=1,
        sample_rate=16000,
        dither=0,
        decoding_method="modified_beam_search",
        max_active_paths=10,
    )
    if streaming:
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(**kw)
        s = rec.create_stream()
        s.accept_waveform(16000, samples)
        s.accept_waveform(16000, np.zeros(int(16000 * 0.8), dtype=np.float32))
        s.input_finished()
        while rec.is_ready(s):
            rec.decode_stream(s)
        return rec.get_result(s)
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw)
    s = rec.create_stream()
    s.accept_waveform(16000, samples)
    rec.decode_stream(s)
    return s.result.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir with the bundle")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    samples = load_pcm(args.audio, args.offset, args.duration)
    samples.astype("<f4").tofile(out / "clip.pcm.bin")
    print(f"pcm: {samples.shape[0]} samples")

    feats = compute_fbank(samples)
    feats.astype("<f4").tofile(out / "clip.fbank.bin")
    print(f"fbank: {feats.shape}")

    model = Transducer(model_dir)
    if model.streaming:
        # The streaming pipeline pads the tail like the reference apps.
        padded = np.concatenate(
            [samples, np.zeros(int(16000 * 0.8), dtype=np.float32)]
        )
        feats_padded = compute_fbank(padded)
        encoder_out = model.encode_streaming(feats_padded)
    else:
        encoder_out = model.encode_offline(feats)
    encoder_out.astype("<f4").tofile(out / "clip.encoder.bin")
    print(f"encoder: {encoder_out.shape}")

    table = load_tokens(model_dir / "tokens.txt")
    unk_id = next((i for i, p in table.items() if p == "<unk>"), -1)

    tokens, frames = model.greedy(encoder_out, unk_id)
    greedy = {
        "text": decode_text(tokens, table),
        "tokens": tokens,
        "frames": frames,
    }
    (out / "clip.greedy.json").write_text(
        json.dumps(greedy, ensure_ascii=False, indent=1)
    )
    print(f"greedy: {greedy['text']!r}")

    tokens, frames = model.modified_beam(encoder_out, unk_id, beam=10)
    beam = {
        "text": decode_text(tokens, table),
        "tokens": tokens,
        "frames": frames,
    }
    (out / "clip.beam.json").write_text(
        json.dumps(beam, ensure_ascii=False, indent=1)
    )
    print(f"beam:   {beam['text']!r}")

    sherpa = sherpa_check(model_dir, samples, model.streaming)
    if sherpa is not None:
        print(f"sherpa: {sherpa!r}")
        if sherpa.strip() != beam["text"]:
            print("WARNING: sherpa-onnx text differs from the beam dump")
    else:
        print("sherpa-onnx not installed; cross-check skipped")


if __name__ == "__main__":
    main()
