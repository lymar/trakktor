//! Fixed constants of the GigaAM pipeline.
//!
//! Only values common to every checkpoint live here. Feature-extraction
//! geometry (`n_fft`, window, centering, mel bands) varies per checkpoint and
//! is carried by [`MelConfig`](super::feature::MelConfig); model geometry
//! (channel width, layer count, vocabulary) comes from the checkpoint config.

/// Audio sample rate the models expect (Hz).
pub const SAMPLE_RATE: usize = 16000;

/// STFT hop, i.e. samples between frames (`sample_rate / 100`). Every published
/// checkpoint uses this hop.
pub const HOP_LENGTH: usize = 160;

/// Lower clamp of the log spectrogram (`SpecScaler`): `log(clamp(x, 1e-9,
/// 1e9))`.
pub const SPEC_CLAMP_MIN: f32 = 1e-9;

/// Upper clamp of the log spectrogram.
pub const SPEC_CLAMP_MAX: f32 = 1e9;

/// The encoder subsampling factor: the encoder emits one frame per this many
/// feature frames (two conv stages of stride 2).
pub const SUBSAMPLING_FACTOR: usize = 4;

/// Seconds of audio per encoder output frame: `HOP_LENGTH * SUBSAMPLING_FACTOR
/// / SAMPLE_RATE = 0.04` s (25 frames per second).
pub const FRAME_SHIFT_S: f64 =
    (HOP_LENGTH * SUBSAMPLING_FACTOR) as f64 / SAMPLE_RATE as f64;

/// Short-form audio limit in seconds. The reference `.transcribe` refuses audio
/// longer than this; longer input is segmented and transcribed chunk by chunk.
pub const LONGFORM_THRESHOLD_S: f64 = 25.0;
