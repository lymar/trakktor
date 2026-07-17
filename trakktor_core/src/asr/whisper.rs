//! The Whisper speech-recognition engine.
//!
//! A faithful port of the reference Whisper pipeline. This layer provides audio
//! decoding and log-mel feature extraction; the tokenizer, the runtime-backed
//! forward pass, the window decoder, the transcription loop, and word-level
//! timing are added incrementally on top.
//!
//! The module depends only on external crates and standard library facilities,
//! with no ties to the rest of the crate, so it can later move into its own
//! library unchanged.

mod assets;
pub mod audio;
pub mod constants;
pub mod decoding;
pub mod download;
pub mod error;
pub mod feature;
pub mod model;
#[cfg(feature = "whisper-runtime")]
pub mod runtime;
#[cfg(test)]
mod testing;
pub mod timing;
pub mod tokenizer;
mod trace;
pub mod transcribe;

#[cfg(feature = "audio")]
pub use audio::BuiltinDecoder;
pub use audio::{AudioDecoder, pad_or_trim};
pub use decoding::{
    DecodeResult, DecodingOptions, PromptInput, decode, detect_language,
};
pub use download::{KNOWN_MODELS, ResolvedModel, resolve_model};
pub use error::WhisperError;
pub use feature::{Mel, MelBands, MelWindow, log_mel_spectrogram};
pub use model::{CrossQk, ForwardProvider, Logits, ModelDims, alignment_heads};
#[cfg(feature = "whisper-runtime")]
pub use runtime::{CandleRuntime, Precision};
pub use timing::Word;
pub use tokenizer::{Task, TokenId, Tokenizer};
pub use transcribe::{
    Segment, TranscribeOptions, TranscribeProgress, Transcription, transcribe,
    transcribe_with_progress,
};
