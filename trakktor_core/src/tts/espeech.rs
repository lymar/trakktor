//! The ESpeech engine: a native port of the Russian ESpeech-TTS-1 checkpoints,
//! which are F5-TTS models.
//!
//! Unlike the other synthesis engine in the tree, this one does not generate
//! frame by frame. It is a **flow-matching** model: it starts from noise the
//! shape of the whole utterance and refines it over a fixed number of steps
//! into a mel spectrogram, which a small vocoder then turns into a waveform.
//! Three things follow from that, and they shape the whole port:
//!
//! - the duration is decided **before** generation, from the speech rate of the
//!   reference, so there is no end-of-speech code to wait for and nothing can
//!   run away;
//! - all the randomness is one tensor of initial noise, so a run with a fixed
//!   seed is reproducible exactly — including against the reference
//!   implementation, which is what the parity tests use;
//! - the voice comes only from a reference recording. There are no preset
//!   speakers: the model is zero-shot, conditioned on a reference mel and the
//!   transcript that goes with it.
//!
//! # Credits
//!
//! Ported from **F5-TTS** by Yushen Chen and co-authors (MIT) — the DiT
//! backbone, the flow-matching sampler, and the inference pipeline — with the
//! Russian **ESpeech-TTS-1** checkpoints (Apache-2.0) and the **Vocos**
//! vocoder by Charactr / gemelo.ai (MIT).

pub mod config;
pub mod download;
mod error;
pub mod mel;
mod model;
pub mod reference;
pub mod runtime;
#[cfg(feature = "tts-burn")]
pub mod runtime_burn;
mod schedule;
mod synthesize;
pub mod tokenizer;

pub use config::{DitConfig, VocoderConfig};
pub use download::{
    KNOWN_MODELS, KnownModel, ResolvedModel, VOCODER_DIR, resolve_model,
};
pub use error::EspeechError;
pub use model::SpeechModel;
pub use synthesize::{SplitParagraph, Synthesis, Synthesizer};
pub use tokenizer::CharTokenizer;

/// Compute precision of the DiT.
///
/// The mel frontend and the vocoder always run in full precision: the frontend
/// is host-side arithmetic over a few seconds of audio, and the vocoder is the
/// stage whose output is compared byte for byte.
///
/// Half precision here is `f16`, not `bf16` — the opposite of the other
/// synthesis engine, and for a plain reason: this checkpoint is stored in `f32`
/// and the reference itself runs it in `f16` whenever it has a CUDA device, so
/// `f16` is the tested half-precision path upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Half precision: less memory traffic, and what the reference runs on GPU.
    F16,
    /// Full precision: the parity baseline.
    F32,
}

/// Everything one synthesis run needs beyond the text itself.
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Steps the ODE solver takes. Quality against time, near linearly.
    pub nfe_step: usize,
    /// Strength of classifier-free guidance. Zero drops the unconditional
    /// branch and halves the work.
    pub cfg_strength: f32,
    /// Speech-rate multiplier: it scales the predicted duration, so a smaller
    /// value gives the same words more room and a slower reading.
    pub speed: f32,
    /// Seed of the initial noise, making a run repeatable.
    pub seed: u64,
    /// Silence between paragraphs, in seconds, when a text is spoken in
    /// several pieces.
    pub pause: f64,
    /// Whether the pieces are brought to a common loudness before they are
    /// joined ([`crate::tts::Join::match_levels`]).
    pub match_levels: bool,
}

/// What a piece is busy with.
///
/// Solving is reported per step; the vocoder is one long call that reports only
/// its start, so a caller can say what the silence is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The solver is refining this piece's mel spectrogram.
    Solving,
    /// The vocoder is turning the finished mel into audio.
    Vocoding,
}

/// How far along a run is, reported as it goes.
///
/// The costs are in the unit the chunk budget counts (bytes of UTF-8 text), so
/// a caller can turn them into a percentage; the audio seconds are what will
/// have been produced once the current piece is vocoded — unlike a frame-by-
/// frame engine, this one knows a piece's length before it starts.
#[derive(Debug, Clone, Copy)]
pub struct SpeechProgress {
    /// What the piece is doing right now.
    pub stage: Stage,
    /// 1-based index of the piece being spoken.
    pub chunk: usize,
    /// Pieces planned.
    pub chunks: usize,
    /// Solver steps finished for this piece.
    pub step: usize,
    /// Solver steps this piece takes in total.
    pub steps: usize,
    /// Seconds of audio from the pieces already finished.
    pub finished_audio: f64,
    /// Seconds of audio including what the current piece will contribute.
    pub audio: f64,
    /// Cost of the pieces already finished.
    pub done_cost: usize,
    /// Cost of the whole text.
    pub total_cost: usize,
}
