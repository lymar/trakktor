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
pub mod text;

#[cfg(test)]
mod tests;

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

/// Amplitude below which a sample counts as silence when a piece's edges are
/// trimmed: about −60 dBFS. The models' own noise floor sits some 25 dB below
/// that and speech some 30 dB above it, so nothing quiet is mistaken for
/// silence.
const SILENCE_LEVEL: f32 = 1e-3;

/// How pieces of a long run are joined into one waveform.
#[derive(Debug, Clone, Copy)]
pub struct Join {
    /// Linear fade at each join, in seconds. Short enough to be inaudible,
    /// long enough that a splice does not click.
    pub fade: f64,
    /// Silence left at a piece's edge after trimming, in seconds. A piece is
    /// not cut back to the first sample of speech: a breath of room before a
    /// word sounds natural, an abrupt start does not.
    pub edge: f64,
}

impl Default for Join {
    fn default() -> Self {
        Self {
            fade: 0.01,
            edge: 0.03,
        }
    }
}

/// Joins `pieces` in order, putting `pauses[i]` seconds of silence between
/// piece `i` and piece `i + 1`.
///
/// Each piece is first trimmed to its speech (plus [`Join::edge`]) and faded at
/// both ends. Trimming is what makes the pause mean anything: a piece arrives
/// with two to three tenths of a second of silence at each edge, and left
/// alone that would add itself to every gap, at a length the model — not the
/// caller — decided.
///
/// Pieces are assumed to share a sample rate; the first one's is used.
#[must_use]
pub fn stitch(pieces: &[Speech], pauses: &[f64], join: Join) -> Speech {
    let Some(first) = pieces.first() else {
        return Speech {
            samples: Vec::new(),
            sample_rate: 0,
        };
    };
    let sample_rate = first.sample_rate;
    let seconds = |value: f64| (value * f64::from(sample_rate)) as usize;

    let mut samples: Vec<f32> = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        if index > 0 {
            let pause = pauses.get(index - 1).copied().unwrap_or(0.0);
            samples.extend(std::iter::repeat_n(0.0, seconds(pause)));
        }
        let trimmed = trim_to_speech(&piece.samples, seconds(join.edge));
        let start = samples.len();
        samples.extend_from_slice(trimmed);
        fade_edges(&mut samples[start..], seconds(join.fade));
    }
    Speech {
        samples,
        sample_rate,
    }
}

/// The span of `samples` holding speech, widened by `edge` samples on each
/// side. Empty when the piece is silence throughout.
fn trim_to_speech(samples: &[f32], edge: usize) -> &[f32] {
    let speech = |sample: &f32| sample.abs() > SILENCE_LEVEL;
    let Some(first) = samples.iter().position(speech) else {
        return &[];
    };
    let last = samples
        .iter()
        .rposition(speech)
        .expect("a first speech sample implies a last one");
    let start = first.saturating_sub(edge);
    let end = (last + 1 + edge).min(samples.len());
    &samples[start..end]
}

/// Fades a piece in and out linearly over `fade` samples, so its ends meet the
/// neighbouring silence at zero. The fade is clamped to half the piece.
fn fade_edges(samples: &mut [f32], fade: usize) {
    let fade = fade.min(samples.len() / 2);
    if fade == 0 {
        return;
    }
    let len = samples.len();
    for offset in 0..fade {
        let gain = offset as f32 / fade as f32;
        samples[offset] *= gain;
        samples[len - 1 - offset] *= gain;
    }
}

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

    /// Writes the audio through an external `ffmpeg` process, which picks the
    /// container and codec from `path`'s extension — the way out to the
    /// formats the built-in encoders do not write (mp3, m4a, opus, ogg, …).
    ///
    /// `bitrate` (for example `"192k"`) applies to lossy formats.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError`] when ffmpeg is missing, fails, or the file
    /// cannot be written.
    pub fn write_ffmpeg(
        &self,
        path: &std::path::Path,
        bitrate: Option<&str>,
    ) -> Result<(), AudioError> {
        // ffmpeg reads the samples as they were synthesized and converts to
        // whatever the container needs.
        let audio = DecodedAudio::from_parts(
            self.sample_rate,
            None,
            None,
            NativeBuf::F32(vec![self.samples.clone()]),
        );
        encode::write_ffmpeg(path, &audio, bitrate)
    }
}
