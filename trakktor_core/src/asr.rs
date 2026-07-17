//! Speech recognition (ASR).
//!
//! ASR is modeled as a domain of interchangeable engines rather than a single
//! universal call: different models expose different capabilities and settings
//! that do not reduce to a common denominator. Each engine lives in its own
//! submodule and owns its inputs, options, and result shape. The first engine
//! is [`whisper`].
//!
//! Alongside the engines, the domain has engine-independent preprocessing that
//! works on any engine's 16 kHz mono PCM input; the first such stage is
//! voice-activity detection ([`vad`]), which finds speech and drops non-speech
//! before transcription.

#[cfg(feature = "vad")]
pub mod vad;
pub mod whisper;
