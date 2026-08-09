//! The seam between the driver and a runtime.
//!
//! The driver owns everything that is not a tensor operation — decoding the
//! file, cutting it into windows, deciding which frames are holes, stitching
//! the windows back together, matching the original loudness, writing the
//! result. A runtime owns exactly one operation, and it takes and returns
//! plain host values:
//!
//! - [`enhance_window`](EnhanceModel::enhance_window): 16 kHz samples in, 16
//!   kHz samples out.
//!
//! Keeping the seam this coarse is deliberate. Everything inside one window is
//! a single chain of matrix multiplications with no host decision in the
//! middle, so there is nothing to gain from a finer split and one synchronous
//! round trip per stage to lose.

use std::path::Path;

use crate::{
    audio::{
        AudioError, DecodedAudio,
        encode::{self, Format},
        pipeline::NativeBuf,
    },
    enhance::error::EnhanceError,
};

/// One loaded pipeline, on one runtime and one device.
pub trait EnhanceModel: Send {
    /// Enhances one window of 16 kHz mono audio.
    ///
    /// `lost` marks the frames the packet-loss detector found, one flag per 320
    /// samples; an empty slice disables concealment for this window. The result
    /// is `frames × 320` samples long, where `frames` is what the window aligns
    /// to — that is, as long as the aligned window, not necessarily as long as
    /// what was passed in.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Compute`] when the tensor backend fails.
    fn enhance_window(
        &mut self,
        samples: &[f32],
        lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError>;

    /// Which runtime this is, for the run's report.
    fn runtime(&self) -> Runtime;
}

/// The tensor backend a run computes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Runtime {
    /// The candle runtime.
    #[default]
    Candle,
    /// The burn runtime.
    Burn,
}

impl Runtime {
    /// The name this runtime reports under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Burn => "burn",
        }
    }
}

/// Compute precision of a run.
///
/// The published checkpoints are `f32` throughout and the reference offers no
/// half-precision path, so [`F32`](Precision::F32) is both the default and the
/// only mode parity is claimed in. [`F16`](Precision::F16) is a candle-only
/// speed option, and is a different computation of the same model rather than
/// the same one faster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Full precision (`f32`), the default.
    #[default]
    F32,
    /// Half precision (`f16`).
    F16,
}

impl Precision {
    /// The name this precision reports under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

/// Everything one enhancement run needs beyond the recording itself.
#[derive(Debug, Clone)]
pub struct EnhanceOptions {
    /// The rate to write at. `None` keeps the recording's own rate.
    ///
    /// The pipeline works at 16 kHz whatever this is, and the result is
    /// resampled to what is asked for. A rate above 16 kHz therefore buys a
    /// container that matches the source, not bandwidth that is not there.
    pub sample_rate: Option<u32>,
    /// Whether to conceal lost packets.
    pub plc: bool,
}

impl Default for EnhanceOptions {
    fn default() -> Self {
        Self {
            sample_rate: None,
            plc: true,
        }
    }
}

/// How far along a run is, reported as it goes.
#[derive(Debug, Clone, Copy)]
pub struct EnhanceProgress {
    /// 1-based index of the window just finished.
    pub window: usize,
    /// Windows in the recording.
    pub windows: usize,
    /// Seconds of audio the finished windows cover.
    pub done_seconds: f64,
    /// Seconds of audio in the recording.
    pub total_seconds: f64,
}

/// A callback invoked after each window.
pub type Progress<'a> = &'a mut dyn FnMut(EnhanceProgress);

/// The enhanced recording.
#[derive(Debug, Clone)]
pub struct Enhanced {
    /// Mono samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Their rate.
    pub sample_rate: u32,
    /// The recording's own rate, before anything was done to it.
    pub source_sample_rate: u32,
    /// Windows the pipeline ran.
    pub windows: usize,
    /// Frames the packet-loss detector concealed. Zero for an engine that has
    /// no concealment to do — a masking network cannot fill a hole, it can
    /// only attenuate what is in one.
    pub concealed_frames: usize,
    /// The same, in seconds of audio.
    pub concealed_seconds: f64,
}

impl Enhanced {
    /// Length of the result, in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// Writes the audio to `path`.
    ///
    /// WAV keeps the samples exactly as enhanced — 32-bit float. FLAC is
    /// integer-only, so the samples are quantized to 24 bits first.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError`] when the encoder or the file write fails.
    pub fn write(&self, path: &Path, format: Format) -> Result<(), AudioError> {
        let audio = match format {
            Format::Wav => DecodedAudio::from_parts(
                self.sample_rate,
                None,
                None,
                NativeBuf::F32(vec![self.samples.clone()]),
            ),
            Format::Flac => {
                let peak = f64::from((1u32 << 23) - 1);
                let quantized: Vec<i32> = self
                    .samples
                    .iter()
                    .map(|&sample| {
                        let scaled = (f64::from(sample) * peak).round();
                        (scaled.clamp(-peak, peak) as i32) << 8
                    })
                    .collect();
                DecodedAudio::from_parts(
                    self.sample_rate,
                    None,
                    Some(24),
                    NativeBuf::S32(vec![quantized]),
                )
            },
        };
        encode::write(path, &audio, format)
    }

    /// Writes the audio through an external `ffmpeg`, which picks the container
    /// and codec from `path`'s extension.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError`] when ffmpeg is missing, fails, or the file cannot
    /// be written.
    pub fn write_ffmpeg(
        &self,
        path: &Path,
        bitrate: Option<&str>,
    ) -> Result<(), AudioError> {
        let audio = DecodedAudio::from_parts(
            self.sample_rate,
            None,
            None,
            NativeBuf::F32(vec![self.samples.clone()]),
        );
        encode::write_ffmpeg(path, &audio, bitrate)
    }
}
