//! `trakktor_core` — reusable implementation of trakktor's features,
//! independent of the CLI.
//!
//! The CLI crate (`trakktor`) is a thin layer that parses arguments, resolves
//! configuration, and formats the values returned here.

pub mod asr;
#[cfg(feature = "audio")]
pub mod audio;
pub mod download;
#[cfg(feature = "enhance-runtime")]
pub mod enhance;
pub mod feed;
pub mod http;
#[cfg(feature = "ocr-runtime")]
pub mod ocr;
#[cfg(feature = "punctuate-runtime")]
pub mod punctuate;
pub mod skill;
#[cfg(feature = "stress-runtime")]
pub mod stress;
#[cfg(feature = "structify-runtime")]
pub mod structify;
#[cfg(any(feature = "stress-runtime", feature = "tts-runtime"))]
pub mod torch_package;
#[cfg(feature = "tts-runtime")]
pub mod tts;
#[cfg(feature = "vad")]
pub mod vad;
