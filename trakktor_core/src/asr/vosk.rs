//! The Vosk speech-recognition engine (Zipformer2 transducer line).
//!
//! A native port of the current Vosk model line (Alpha Cephei): end-to-end
//! RNN-T transducers with a Zipformer2 encoder, trained with `k2-fsa/icefall`
//! and published as ONNX `encoder`/`decoder`/`joiner` exports plus a
//! SentencePiece token table. The inference pipeline — Kaldi-compatible fbank
//! features, the encoder, and greedy / modified-beam-search transducer
//! decoding — reproduces the reference `k2-fsa/sherpa-onnx` runtime. Both
//! offline (full-context) and streaming (causal, chunked with cached state)
//! models are supported. The primary runtime is [candle](candle_core); an
//! alternative burn runtime is available behind the `vosk-burn` feature and
//! selected at run time through the [`EncoderSeam`] seam.
//!
//! Like the sibling engines, this module depends only on external crates and
//! standard-library facilities (the long-form path also uses the shared
//! [`crate::vad`] detector), so it can later move into its own library
//! unchanged.

#[cfg(feature = "vosk-runtime")]
pub mod catalog;
pub mod constants;
#[cfg(feature = "vosk-runtime")]
pub mod decode;
#[cfg(feature = "vosk-runtime")]
pub mod download;
pub mod error;
pub mod feature;
#[cfg(feature = "vosk-runtime")]
pub mod onnx;
#[cfg(feature = "vosk-runtime")]
pub mod runtime;
#[cfg(feature = "vosk-burn")]
pub mod runtime_burn;
pub mod segment;
#[cfg(all(feature = "vosk-runtime", feature = "vad"))]
pub mod stream;
pub mod tokenizer;
#[cfg(feature = "vosk-runtime")]
pub mod transcribe;
#[cfg(feature = "vosk-runtime")]
pub mod weights;

#[cfg(feature = "vosk-runtime")]
pub use catalog::{KNOWN_MODELS, ModelKind, ModelSpec};
pub use constants::SAMPLE_RATE;
#[cfg(feature = "vosk-runtime")]
pub use decode::{Decoding, TransducerHead};
#[cfg(feature = "vosk-runtime")]
pub use download::{ResolvedModel, resolve_model};
pub use error::VoskError;
pub use feature::{FbankExtractor, Features};
#[cfg(feature = "vosk-runtime")]
pub use runtime::{EncoderSeam, Precision, StreamingSession, VoskModel};
#[cfg(feature = "vosk-burn")]
pub use runtime_burn::VoskBurnModel;
#[cfg(all(feature = "vosk-runtime", feature = "vad"))]
pub use stream::{StreamTranscriber, transcribe, transcribe_with_progress};
pub use tokenizer::Tokenizer;
#[cfg(feature = "vosk-runtime")]
pub use transcribe::{
    Segment, TranscribeOptions, TranscribeProgress, Transcription, Word,
};
