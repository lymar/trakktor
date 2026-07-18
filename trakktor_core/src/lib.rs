//! `trakktor_core` — reusable implementation of trakktor's features,
//! independent of the CLI.
//!
//! The CLI crate (`trakktor`) is a thin layer that parses arguments, resolves
//! configuration, and formats the values returned here.

pub mod asr;
#[cfg(feature = "audio")]
pub mod audio;
pub mod feed;
pub mod http;
pub mod skill;
#[cfg(feature = "structify-runtime")]
pub mod structify;
#[cfg(feature = "vad")]
pub mod vad;
