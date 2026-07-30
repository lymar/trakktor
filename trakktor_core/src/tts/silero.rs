//! The Silero engine: a native port of Silero TTS v5.
//!
//! The third synthesis engine in the tree, and the one that behaves least like
//! a generative model. There is no autoregression, no solver and — this is the
//! part that shapes everything else — **no sampling anywhere**. Symbol ids go
//! in, a waveform comes out, in one pass:
//!
//! 1. a duration head says how many frames each symbol lasts and a pitch head
//!    says how it is intoned;
//! 2. an encoder embeds the symbols, the speaker and the pitch, and the length
//!    regulator repeats each symbol for as many frames as it was given;
//! 3. an hourglass decoder — full resolution, a third of it, full again — turns
//!    that into a mel spectrogram;
//! 4. a Vocos vocoder predicts a complex spectrum, and the inverse transform
//!    makes it a waveform at 48 kHz.
//!
//! Three consequences follow, and they are why this engine is worth having next
//! to the other two:
//!
//! - **a run is repeatable by construction**, so there is no `--seed` and the
//!   whole pipeline can be checked against the reference stage by stage rather
//!   than "in spirit";
//! - **stress is an input, not a hope**: `+` before a stressed vowel is symbol
//!   id 5, an ordinary member of the alphabet;
//! - **the CPU is the intended device**, not the fallback. Twenty-three million
//!   parameters and one pass make a machine without a GPU a first-class place
//!   to run this.
//!
//! # Credits
//!
//! Ported from **Silero TTS v5** by the Silero Team — the FastPitch-family
//! acoustic model, its text frontend, and the vocoder's filterbanks — and from
//! **Vocos** by Charactr / gemelo.ai (MIT), the vocoder itself. The base pair
//! of models is MIT; the rest of the published line is not, which is why the
//! model registry carries a licence field.

pub mod config;
pub mod download;
mod error;
mod frontend;
mod intonation;
pub mod istft;
pub mod model;
pub mod runtime;
#[cfg(feature = "tts-burn")]
pub mod runtime_burn;
mod synthesize;
pub mod tables;

pub use config::{Config, FRAME_SECONDS, SAMPLE_RATE};
pub use download::{
    DEFAULT_MODEL, KNOWN_MODELS, KnownModel, License, ResolvedModel,
    permissive_model_names, resolve_model,
};
pub use error::SileroError;
pub use model::SpeechModel;
pub use synthesize::{SAMPLE_RATES, SplitParagraph, Synthesis, Synthesizer};
pub use tables::Tables;

/// Everything one synthesis run needs beyond the text itself.
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// The speaker to read in.
    pub voice: String,
    /// Which of the model's own output rates to produce.
    pub sample_rate: u32,
    /// Speech-rate multiplier: above 1 speaks faster. It divides the predicted
    /// durations, so it changes how long the words are given rather than
    /// replaying them at a different speed.
    pub rate: f32,
    /// Pitch multiplier: it scales the predicted contour and shifts it by the
    /// speaker's own range, so a voice raised this way still sounds like
    /// itself.
    pub pitch: f32,
    /// Silence between paragraphs, in seconds.
    pub pause: f64,
    /// Whether the pieces are brought to a common loudness before they are
    /// joined ([`crate::tts::Join::match_levels`]).
    pub match_levels: bool,
}

/// What a piece is busy with.
///
/// The whole pipeline is one call per piece, so the two stages are what a
/// caller can honestly be told: the piece has started, and its spectrum is
/// being turned back into sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The networks are running.
    Synthesizing,
    /// The spectrum is being inverted.
    Vocoding,
}

/// How far along a run is, reported as it goes.
///
/// The costs are in the unit the chunk budget counts (bytes of UTF-8 text), so
/// a caller can turn them into a percentage.
#[derive(Debug, Clone, Copy)]
pub struct SpeechProgress {
    /// What the piece is doing right now.
    pub stage: Stage,
    /// 1-based index of the piece being spoken.
    pub chunk: usize,
    /// Pieces planned.
    pub chunks: usize,
    /// Seconds of audio from the pieces already finished.
    pub finished_audio: f64,
    /// Seconds of audio including what the current piece will contribute.
    pub audio: f64,
    /// Cost of the pieces already finished.
    pub done_cost: usize,
    /// Cost of the whole text.
    pub total_cost: usize,
}
