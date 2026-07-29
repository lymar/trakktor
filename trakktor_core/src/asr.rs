//! Speech recognition (ASR).
//!
//! ASR is modeled as a domain of interchangeable engines rather than a single
//! universal call: different models expose different capabilities and settings
//! that do not reduce to a common denominator. Each engine lives in its own
//! submodule and owns its inputs, options, and result shape. The engines are
//! [`whisper`], [`gigaam`], and [`vosk`].
//!
//! Alongside the engines, the domain has engine-independent preprocessing that
//! works on any engine's 16 kHz mono PCM input; the first such stage is
//! voice-activity detection ([`crate::vad`]), which finds speech and drops
//! non-speech before transcription. The detector lives in the shared
//! [`crate::vad`] module (it is not ASR-specific), and this domain uses it as a
//! preprocessing stage. The second is [`segment`]: turning the detected speech
//! into the chunks a whole-utterance model can take, shared by every engine
//! that chunks long audio.

pub mod gigaam;
pub mod segment;
pub mod vosk;
pub mod whisper;
