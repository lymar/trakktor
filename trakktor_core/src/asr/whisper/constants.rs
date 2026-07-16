//! Whisper's fixed audio and framing hyperparameters.
//!
//! These values are intrinsic to the Whisper model family: the network was
//! trained on 16 kHz mono audio turned into 80- or 128-band log-mel
//! spectrograms with a 25 ms analysis window and a 10 ms hop. They never vary
//! between models.

/// Sample rate the model expects, in Hz.
pub const SAMPLE_RATE: usize = 16_000;

/// Short-time Fourier transform window length, in samples (25 ms).
pub const N_FFT: usize = 400;

/// Hop between consecutive STFT frames, in samples (10 ms).
pub const HOP_LENGTH: usize = 160;

/// Length of one processing window, in seconds.
pub const CHUNK_LENGTH: usize = 30;

/// Audio samples in one processing window (`CHUNK_LENGTH * SAMPLE_RATE`).
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480_000

/// Mel frames in one processing window (`N_SAMPLES / HOP_LENGTH`).
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3_000

/// Audio samples per decoded token: `HOP_LENGTH * 2`, reflecting the encoder's
/// stride-2 convolution.
pub const N_SAMPLES_PER_TOKEN: usize = HOP_LENGTH * 2; // 320

/// Mel frames per second of audio (`SAMPLE_RATE / HOP_LENGTH`).
pub const FRAMES_PER_SECOND: usize = SAMPLE_RATE / HOP_LENGTH; // 100

/// Decoded tokens per second of audio (`SAMPLE_RATE / N_SAMPLES_PER_TOKEN`).
pub const TOKENS_PER_SECOND: usize = SAMPLE_RATE / N_SAMPLES_PER_TOKEN; // 50

/// One-sided STFT frequency bins kept per frame (`N_FFT / 2 + 1`).
pub const N_FREQS: usize = N_FFT / 2 + 1; // 201
