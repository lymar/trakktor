//! Speech synthesis (`trakktor tts`): text in, an audio file out.
//!
//! The counterpart to [`asr`](crate::asr) and, like it, a domain with several
//! interchangeable engines rather than one universal call: engines differ in
//! their voices, conditioning modes, and knobs, and those do not reduce to a
//! common denominator. Each engine lives in its own submodule and is selected
//! as a subcommand.
//!
//! This module holds only what generalizes across engines — how a voice is
//! described and what one synthesis run returns. Everything engine-specific
//! (model variants, sampling, runtime plumbing) belongs to the engine.

pub mod qwen3_tts;

use crate::audio::{
    AudioError, DecodedAudio,
    encode::{self, Format},
    pipeline::NativeBuf,
};

/// How the voice for one run was chosen.
///
/// The three kinds are mutually exclusive within a run; which ones an engine
/// offers depends on the engine and the selected model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Voice {
    /// A ready-made named timbre shipped with the model.
    Preset {
        /// The speaker name, as the engine knows it.
        name: String,
    },
    /// A voice described in natural language.
    Design,
    /// A voice cloned from reference audio.
    Clone {
        /// Which cloning mode produced it.
        mode: CloneMode,
        /// The reference audio the voice was taken from.
        ref_audio: String,
    },
}

impl Voice {
    /// The discriminator reported in the output contract.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Voice::Preset { .. } => "preset",
            Voice::Design => "design",
            Voice::Clone { .. } => "clone",
        }
    }
}

/// Which conditioning a cloned voice uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloneMode {
    /// Reference audio plus its transcript: the model conditions on both the
    /// reference codes and the speaker embedding. Higher quality.
    InContext,
    /// Reference audio alone, through the speaker embedding. No transcript
    /// needed, at some cost in quality.
    XVectorOnly,
}

impl CloneMode {
    /// The value reported in the output contract.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            CloneMode::InContext => "icl",
            CloneMode::XVectorOnly => "x_vector_only",
        }
    }
}

/// Synthesized audio: interleaved-free mono samples plus their rate.
///
/// Engines return the waveform; writing it to a file (and choosing the
/// container) is the caller's job, so the same result can be encoded as WAV or
/// FLAC without the engine knowing about either.
#[derive(Debug, Clone)]
pub struct Speech {
    /// Mono samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

/// Bit depth the samples are quantized to for FLAC, which cannot hold floats.
/// Deep enough that the step is inaudible.
const FLAC_BITS: u32 = 24;

impl Speech {
    /// Duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// Writes the audio to `path` in the requested container.
    ///
    /// WAV keeps the samples exactly as synthesized — 32-bit float. FLAC is
    /// integer-only, so the samples are quantized to 24 bits first.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError`] when the encoder or the file write fails.
    pub fn write(
        &self,
        path: &std::path::Path,
        format: Format,
    ) -> Result<(), AudioError> {
        let audio = match format {
            Format::Wav => DecodedAudio::from_parts(
                self.sample_rate,
                None,
                None,
                NativeBuf::F32(vec![self.samples.clone()]),
            ),
            Format::Flac => {
                let peak = f64::from((1u32 << (FLAC_BITS - 1)) - 1);
                let quantized: Vec<i32> = self
                    .samples
                    .iter()
                    .map(|&sample| {
                        let scaled = (f64::from(sample) * peak).round();
                        // 24-bit samples ride in the high bits of an i32.
                        (scaled.clamp(-peak, peak) as i32) << 8
                    })
                    .collect();
                DecodedAudio::from_parts(
                    self.sample_rate,
                    None,
                    Some(FLAC_BITS),
                    NativeBuf::S32(vec![quantized]),
                )
            },
        };
        encode::write(path, &audio, format)
    }
}
