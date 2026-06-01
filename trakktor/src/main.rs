//! `trakktor` — CLI entry point.
//!
//! A thin layer over `trakktor_core`: parse arguments, resolve configuration,
//! delegate to the library, and format the result. See architecture.md.

mod cli;
mod output;

fn main() { std::process::exit(cli::run()); }
