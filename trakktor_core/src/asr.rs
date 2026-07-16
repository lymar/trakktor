//! Speech recognition (ASR).
//!
//! ASR is modeled as a domain of interchangeable engines rather than a single
//! universal call: different models expose different capabilities and settings
//! that do not reduce to a common denominator. Each engine lives in its own
//! submodule and owns its inputs, options, and result shape. The first engine
//! is [`whisper`].

pub mod whisper;
