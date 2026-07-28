//! `trakktor_core` — reusable implementation of trakktor's features,
//! independent of the CLI.
//!
//! The CLI crate (`trakktor`) is a thin layer that parses arguments, resolves
//! configuration, and formats the values returned here.

pub mod asr;
#[cfg(feature = "audio")]
pub mod audio;
pub mod download;
pub mod feed;
pub mod http;
#[cfg(feature = "punctuate-runtime")]
pub mod punctuate;
pub mod skill;
#[cfg(feature = "structify-runtime")]
pub mod structify;
#[cfg(feature = "tts-runtime")]
pub mod tts;
#[cfg(feature = "vad")]
pub mod vad;
