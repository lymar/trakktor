//! Audio decoding for the Whisper engine.
//!
//! The engine consumes 16 kHz mono f32 PCM. Turning an arbitrary media file
//! into that form is kept behind the [`AudioDecoder`] trait so the backend can
//! change later — today an external ffmpeg process, in future an in-process
//! decoder — without touching feature extraction or decoding.

use std::{path::Path, process::Command};

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

/// An [`AudioDecoder`] backed by an external `ffmpeg` process.
///
/// Runs `ffmpeg` to downmix to mono, resample to 16 kHz, and emit signed 16-bit
/// little-endian PCM, then scales the samples to f32. This mirrors the
/// reference decoding path, so the resulting signal is numerically comparable.
#[derive(Debug, Clone)]
pub struct FfmpegDecoder {
    binary: String,
}

impl FfmpegDecoder {
    /// Uses the given `ffmpeg` executable (a name resolved on `PATH`, or an
    /// absolute path).
    pub fn new(binary: impl Into<String>) -> Self {
        Self {
            binary: binary.into(),
        }
    }
}

impl Default for FfmpegDecoder {
    fn default() -> Self { Self::new("ffmpeg") }
}

impl AudioDecoder for FfmpegDecoder {
    fn decode(&self, path: &Path) -> Result<Vec<f32>, WhisperError> {
        let output = Command::new(&self.binary)
            .args(["-nostdin", "-threads", "0", "-i"])
            .arg(path)
            .args(["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar"])
            .arg(SAMPLE_RATE.to_string())
            .arg("-")
            .output()
            .map_err(|e| {
                WhisperError::AudioDecode(format!(
                    "could not run '{}': {e}",
                    self.binary
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("unknown error");
            return Err(WhisperError::AudioDecode(format!(
                "ffmpeg exited with {}: {detail}",
                output.status
            )));
        }

        Ok(pcm_s16le_to_f32(&output.stdout))
    }
}

/// Converts signed 16-bit little-endian PCM bytes to f32 in `[-1, 1)`.
///
/// A trailing odd byte (an incomplete sample) is ignored.
fn pcm_s16le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
        .collect()
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
