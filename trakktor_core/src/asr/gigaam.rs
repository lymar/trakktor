//! The GigaAM speech-recognition engine.
//!
//! A faithful port of the reference GigaAM pipeline (Sber's Conformer-based
//! acoustic models): log-mel feature extraction, the Conformer encoder, and
//! CTC / RNN-T greedy decoding. The primary runtime is
//! [candle](candle_core), with Metal acceleration behind a cargo feature; an
//! alternative burn runtime is available behind the `gigaam-burn` feature and
//! selected at run time through the [`CtcModel`] seam.
//!
//! Like the [`whisper`](super::whisper) engine, this module depends only on
//! external crates and standard-library facilities, with no ties to the rest of
//! the crate, so it can later move into its own library unchanged.

#[cfg(feature = "gigaam-runtime")]
pub mod config;
pub mod constants;
pub mod decode;
#[cfg(feature = "gigaam-runtime")]
pub mod download;
pub mod error;
pub mod feature;
#[cfg(feature = "gigaam-runtime")]
pub mod runtime;
#[cfg(feature = "gigaam-burn")]
pub mod runtime_burn;
pub mod segment;
#[cfg(all(feature = "gigaam-runtime", feature = "vad"))]
pub mod stream;
pub mod tokenizer;
#[cfg(feature = "gigaam-runtime")]
pub mod transcribe;

#[cfg(feature = "gigaam-runtime")]
pub use config::{
    KNOWN_MODELS, ModelClass, ModelConfig, TokenizerConfig, config_for,
};
pub use decode::{Decoded, Word, decode_chunk, frames_to_words};
#[cfg(feature = "gigaam-runtime")]
pub use download::{ResolvedModel, resolve_model};
pub use error::GigaamError;
pub use feature::{FeatureExtractor, Mel, MelConfig};
#[cfg(feature = "gigaam-runtime")]
pub use runtime::{CtcModel, GigaamModel, Precision};
#[cfg(feature = "gigaam-burn")]
pub use runtime_burn::GigaamBurnModel;
pub use segment::Interval;
#[cfg(all(feature = "gigaam-runtime", feature = "vad"))]
pub use stream::{StreamTranscriber, transcribe, transcribe_with_progress};
pub use tokenizer::Tokenizer;
#[cfg(feature = "gigaam-runtime")]
pub use transcribe::{
    Segment, TranscribeOptions, TranscribeProgress, Transcription,
};
