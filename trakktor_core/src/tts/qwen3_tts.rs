//! The Qwen3-TTS engine: a native port of the open 12 Hz family.
//!
//! The pipeline has three stages. A **talker** — a Qwen3 decoder fed two tracks
//! at once, text and codec tokens summed channel-wise — predicts codebook 0 of
//! every 12.5 Hz frame. A small **code predictor** then fills the remaining
//! residual codebooks of that frame, one pass each. Finally the **codec
//! decoder**, a causal convolutional network with a windowed-attention stack,
//! turns the finished frames into a 24 kHz waveform.
//!
//! Only the decoding half of the codec is implemented: synthesis never needs to
//! turn audio back into codes.
//!
//! Unlike the recognition engines, generation samples, so results are
//! reproducible per run through a seeded generator rather than identical across
//! runs. The codec decoder is the exception — with the codes fixed it is purely
//! feed-forward, and its output is reproducible exactly.
//!
//! # Credits
//!
//! Ported from **Qwen3-TTS** by the Alibaba Qwen team (Apache-2.0): the
//! `qwen-tts` package and the published 12 Hz checkpoints, including their
//! bundled speech tokenizer. The speaker-verification network used for voice
//! cloning follows **ECAPA-TDNN** (Desplanques, Thienpondt, and Demuynck).

mod chunking;
pub mod config;
pub mod download;
mod error;
mod model;
pub mod prompt;
pub mod runtime;
#[cfg(feature = "tts-burn")]
pub mod runtime_burn;
mod sampler;
mod synthesize;
mod tokenizer;

pub use config::{
    CodePredictorConfig, CodecConfig, GenerationDefaults, ModelConfig,
    ModelType, TalkerConfig,
};
pub use download::{
    KNOWN_MODELS, KnownModel, REQUIRED_FILES, ResolvedModel, resolve_model,
};
pub use error::Qwen3TtsError;
pub use model::SpeechModel;
pub use synthesize::{SplitParagraph, Synthesis, Synthesizer};
pub use tokenizer::TextTokenizer;

/// Compute precision of the talker and the code predictor.
///
/// The codec decoder always runs in full precision: it is the stage whose
/// output is required to reproduce exactly.
///
/// The half-precision option is `bf16`, not `f16`. The checkpoints are stored
/// in `bf16`, and the two are not interchangeable here: `bf16` keeps the
/// exponent range of `f32` and only sheds mantissa bits, while `f16` has a far
/// narrower range. In `f16` the attention scores and the residual stream
/// overflow partway through generation, and the run degenerates into minutes
/// of babble instead of one sentence — measured, not assumed. The reference
/// likewise runs `bf16` throughout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// Half precision, as the weights are stored: less memory, and what the
    /// reference itself runs.
    Bf16,
    /// Full precision: reproducible, at twice the memory.
    F32,
}

/// How the next token is picked, at both sampling levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Always take the most likely token. Deterministic, and the mode the
    /// runtimes are compared in.
    Greedy,
    /// Sample from the top `top_k` candidates after temperature scaling.
    TopK {
        /// Candidates kept before sampling.
        top_k: usize,
        /// Temperature applied to the logits; higher is more random.
        temperature: f32,
        /// Penalty applied to codes already generated.
        repetition_penalty: f32,
        /// Seed making a run reproducible.
        seed: u64,
    },
}

impl Sampling {
    /// The value reported in the output contract.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Sampling::Greedy => "greedy",
            Sampling::TopK { .. } => "top_k",
        }
    }
}

/// Everything one synthesis run needs beyond the text itself.
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// The preset speaker to voice the text with.
    pub voice: String,
    /// The target language; `None` leaves the choice to the model.
    pub language: Option<String>,
    /// How tokens are picked at both levels.
    pub sampling: Sampling,
    /// Silence between paragraphs, in seconds, when a text is spoken in
    /// several pieces.
    pub pause: f64,
}

/// What a piece is busy with. Generation reports every frame; turning the
/// finished frames into a waveform is one long call that reports only its
/// start, so a caller can say what the silence is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// The talker and code predictor are producing frames.
    Generating,
    /// The codec decoder is turning this piece's frames into audio.
    Decoding,
}

/// How far along a multi-piece run is, reported as it goes.
///
/// The costs are in the same unit the chunk budget counts (text tokens), so a
/// caller can turn them into a percentage or a time estimate; the audio
/// seconds are what has actually been generated.
#[derive(Debug, Clone, Copy)]
pub struct SpeechProgress {
    /// What the piece is doing right now.
    pub stage: Stage,
    /// 1-based index of the piece being spoken.
    pub chunk: usize,
    /// Pieces planned so far — it can grow if a piece has to be cut down.
    pub chunks: usize,
    /// Seconds of audio from the pieces already finished.
    pub finished_audio: f64,
    /// Seconds of audio generated in total, the current piece included.
    pub audio: f64,
    /// Cost of the pieces already finished.
    pub done_cost: usize,
    /// Cost of the whole text.
    pub total_cost: usize,
}
