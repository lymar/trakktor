//! Engine-wide constants shared by feature extraction, decoding, and
//! transcription.

/// Input sample rate the models are trained for.
pub const SAMPLE_RATE: usize = 16_000;

/// Seconds of audio per encoder output frame: the 10 ms fbank hop times the
/// fixed 4× subsampling of the reference runtime, used to map emission frames
/// to timestamps.
pub const ENCODER_FRAME_S: f64 = 0.04;

/// Audio at most this long is transcribed as one chunk by an offline model;
/// anything longer goes through speech segmentation.
pub const LONGFORM_THRESHOLD_S: f64 = 25.0;
