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
//! voice-activity detection ([`crate::vad`]), which finds speech and drops
//! non-speech before transcription. The detector lives in the shared
//! [`crate::vad`] module (it is not ASR-specific), and this domain uses it as a
//! preprocessing stage.

pub mod whisper;
