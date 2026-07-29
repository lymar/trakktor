//! `trakktor` — CLI entry point.
//!
//! A thin layer over `trakktor_core`: parse arguments, resolve configuration,
//! delegate to the library, and format the result.

mod asr;
#[cfg(feature = "burn")]
mod burn_notice;
mod cli;
mod error;
mod output;
mod punctuate;
mod skill;
mod stress;
mod structify;
mod tts;
mod vad;

fn main() { std::process::exit(cli::run()); }
