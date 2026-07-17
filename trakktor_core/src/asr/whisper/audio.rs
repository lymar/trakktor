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

/// An [`AudioDecoder`] backed by an external `ffmpeg` process.
///
/// Runs the classic Whisper loading chain
/// (`ffmpeg -nostdin -threads 0 -i <file> -f s16le -ac 1 -acodec pcm_s16le -ar
/// 16000 -`) and reads the 16 kHz mono PCM it writes to stdout. It is opt-in:
/// it requires `ffmpeg` on `PATH`, and in return decodes input formats the
/// [`BuiltinDecoder`] does not (opus, wma, amr, and others). The built-in
/// decoder remains the default and needs no external tools.
#[derive(Debug, Clone, Default)]
pub struct FfmpegDecoder;

impl AudioDecoder for FfmpegDecoder {
    fn decode(&self, path: &Path) -> Result<Vec<f32>, WhisperError> {
        use std::process::{Command, Stdio};

        let output = Command::new("ffmpeg")
            .args(["-nostdin", "-threads", "0", "-i"])
            .arg(path)
            .args(["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar"])
            .arg(SAMPLE_RATE.to_string())
            .arg("-")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| {
                WhisperError::AudioDecode(format!(
                    "could not run ffmpeg ({e}); is it installed and on PATH?"
                ))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = stderr
                .lines()
                .rev()
                .find(|line| !line.trim().is_empty())
                .unwrap_or("")
                .trim();
            return Err(WhisperError::AudioDecode(format!(
                "ffmpeg could not decode {}: {detail}",
                path.display()
            )));
        }

        // s16le little-endian PCM → f32 in `[-1, 1)`, matching the built-in
        // decoder and the reference `load_audio`.
        let bytes = output.stdout;
        let samples = bytes
            .chunks_exact(2)
            .map(|pair| {
                f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32768.0
            })
            .collect();
        Ok(samples)
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
