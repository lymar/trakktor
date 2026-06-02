//! `trakktor` — CLI entry point.
//!
//! A thin layer over `trakktor_core`: parse arguments, resolve configuration,
//! delegate to the library, and format the result.

mod cli;
mod error;
mod output;
mod skill;

fn main() { std::process::exit(cli::run()); }
