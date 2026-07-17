//! Audio decoding for the Whisper engine.
//!
//! The engine consumes 16 kHz mono f32 PCM. Turning an arbitrary media file
//! into that form is kept behind the [`AudioDecoder`] trait so the backend
//! can be swapped (tests use scripted decoders) without touching feature
//! extraction or decoding.

use std::path::Path;

use super::{constants::SAMPLE_RATE, error::WhisperError};

/// Decodes a media file into 16 kHz mono f32 PCM in roughly `[-1, 1)`.
pub trait AudioDecoder {
    /// Decodes `path` into mono 16 kHz f32 samples.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::AudioDecode`] if the file cannot be decoded.
    fn decode(&self, path: &Path) -> Result<Vec<f32>, WhisperError>;
}

/// The built-in [`AudioDecoder`].
///
/// Decodes with the crate's audio subsystem (pure-Rust decoding plus a
/// conversion pipeline numerically faithful to the classic
/// `ffmpeg -f s16le -ac 1 -ar 16000` chain): downmix to mono, resample to
/// 16 kHz, quantize to s16, then scale to f32. The resulting signal matches
/// that reference bit for bit on lossless inputs and within ±1 least
/// significant bit on a small fraction of samples for lossy codecs.
#[cfg(feature = "audio")]
#[derive(Debug, Clone, Default)]
pub struct BuiltinDecoder;

#[cfg(feature = "audio")]
impl AudioDecoder for BuiltinDecoder {
    fn decode(&self, path: &Path) -> Result<Vec<f32>, WhisperError> {
        let samples =
            crate::audio::decode_to_mono_s16(path, SAMPLE_RATE as u32)
                .map_err(|e| WhisperError::AudioDecode(e.to_string()))?;
        Ok(samples.iter().map(|&s| f32::from(s) / 32768.0).collect())
    }
}

/// Right-pads with zeros or truncates `samples` to exactly `length`.
///
/// Whisper works on fixed 30 s windows; shorter tails are zero-filled and
/// longer inputs are cut.
pub fn pad_or_trim(samples: &[f32], length: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(length);
    let take = samples.len().min(length);
    out.extend_from_slice(&samples[..take]);
    out.resize(length, 0.0);
    out
}

#[cfg(test)]
mod tests;
