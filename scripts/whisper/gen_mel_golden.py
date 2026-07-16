#!/usr/bin/env python3
"""Generate Whisper feature-extraction assets and test fixtures.

This is a developer tool, not part of the shipped binary. It produces two kinds
of files consumed by the Rust implementation:

1. Mel filterbanks (`assets/mel_80.bin`, `assets/mel_128.bin`) — the exact
   librosa/Slaney matrices Whisper ships, extracted from the reference
   `mel_filters.npz` and stored as raw little-endian f32, row-major
   (n_mels rows x 201 columns).

2. A hermetic golden fixture (`testdata/*.bin`) derived from a real audio
   sample: the first 2 seconds decoded to 16 kHz mono f32, plus the log-mel
   spectrogram for n_mels in {80, 128}. The Rust test loads the PCM and checks
   its own log-mel against these goldens, so the test needs neither ffmpeg nor
   Python.

The golden log-mel comes from the reference Whisper package when it is
importable (run this with an interpreter that has `openai-whisper` and its
`torch` dependency installed). When it is not, the script falls back to a
faithful NumPy reproduction of Whisper's audio pipeline (`torch.stft` with
center padding, reflect mode, a periodic Hann window, a power spectrum with the
last frame dropped, mel projection, log10, an 80 dB dynamic-range floor, and a
final rescale). When both are available it cross-checks them and reports the
difference, which validates the NumPy path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import whisper.audio as reference_audio

    HAVE_REFERENCE = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    HAVE_REFERENCE = False

# Whisper's fixed audio hyperparameters.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160

# Length of the fixture slice, in seconds.
FIXTURE_SECONDS = 2


def decode_audio(path: Path, ffmpeg: str) -> np.ndarray:
    """Decode `path` to 16 kHz mono f32 in roughly [-1, 1) via ffmpeg.

    Mirrors Whisper's own `load_audio`: downmix to mono, resample to 16 kHz,
    output signed 16-bit little-endian PCM, then scale by 1/32768.
    """
    cmd = [
        ffmpeg, "-nostdin", "-threads", "0",
        "-i", str(path),
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768.0


def hann_periodic(n: int) -> np.ndarray:
    """Periodic Hann window, matching torch.hann_window(periodic=True)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def log_mel_numpy(audio: np.ndarray, mel_filters: np.ndarray, padding: int = 0) -> np.ndarray:
    """Faithful NumPy reproduction of Whisper's log_mel_spectrogram."""
    x = audio.astype(np.float64)
    if padding > 0:
        x = np.concatenate([x, np.zeros(padding, dtype=np.float64)])

    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    window = hann_periodic(N_FFT)

    n_frames = 1 + (len(xp) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[:, None] + HOP_LENGTH * np.arange(n_frames)[None, :]
    frames = xp[idx] * window[:, None]              # (N_FFT, n_frames)

    spectrum = np.fft.rfft(frames, axis=0)          # (201, n_frames)
    spectrum = spectrum[:, :-1]                      # drop the last frame
    power = np.abs(spectrum) ** 2                    # power spectrum

    mel = mel_filters.astype(np.float64) @ power    # (n_mels, n_frames)
    logs = np.log10(np.clip(mel, 1e-10, None))
    logs = np.maximum(logs, logs.max() - 8.0)
    logs = (logs + 4.0) / 4.0
    return logs.astype(np.float32)


def log_mel_reference(audio: np.ndarray, n_mels: int, padding: int = 0) -> np.ndarray:
    """Whisper's own log_mel_spectrogram (requires the reference package)."""
    tensor = reference_audio.log_mel_spectrogram(
        audio.astype(np.float32), n_mels=n_mels, padding=padding
    )
    return tensor.cpu().numpy().astype(np.float32)


def write_f32(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype("<f4").tofile(path)
    print(f"wrote {path}  shape={array.shape}  bytes={array.astype('<f4').nbytes}")


def build_golden(pcm: np.ndarray, n_mels: int, mel_bank: np.ndarray) -> np.ndarray:
    """Compute the golden log-mel, preferring the reference and cross-checking."""
    numpy_mel = log_mel_numpy(pcm, mel_bank)
    if not HAVE_REFERENCE:
        print(f"  mel_{n_mels}: golden from NumPy reproduction (reference package not found)")
        return numpy_mel

    reference_mel = log_mel_reference(pcm, n_mels)
    diff = float(np.abs(reference_mel - numpy_mel).max())
    print(f"  mel_{n_mels}: numpy-vs-reference max abs diff = {diff:.3e}; golden from reference")
    return reference_mel


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    whisper_dir = repo_root / "trakktor_core" / "src" / "asr" / "whisper"

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="path to the reference mel_filters.npz")
    ap.add_argument("--sample", required=True, help="path to a sample audio file")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--assets-dir", default=str(whisper_dir / "assets"))
    ap.add_argument("--testdata-dir", default=str(whisper_dir / "testdata"))
    args = ap.parse_args()

    print(f"reference whisper package: {'available' if HAVE_REFERENCE else 'NOT available'}")

    filters = np.load(args.npz)
    mel_80, mel_128 = filters["mel_80"], filters["mel_128"]
    assert mel_80.shape == (80, 201) and mel_128.shape == (128, 201)

    assets = Path(args.assets_dir)
    write_f32(assets / "mel_80.bin", mel_80)
    write_f32(assets / "mel_128.bin", mel_128)

    pcm = decode_audio(Path(args.sample), args.ffmpeg)[: FIXTURE_SECONDS * SAMPLE_RATE]
    print(f"decoded sample: {len(pcm)} samples ({len(pcm) / SAMPLE_RATE:.2f}s)")

    testdata = Path(args.testdata_dir)
    write_f32(testdata / "sample_2s.pcm.bin", pcm)
    write_f32(testdata / "sample_2s.mel80.bin", build_golden(pcm, 80, mel_80))
    write_f32(testdata / "sample_2s.mel128.bin", build_golden(pcm, 128, mel_128))
    return 0


if __name__ == "__main__":
    sys.exit(main())
