//! Typed error for the VAD stage.
//!
//! The single variant maps to the stable `vad_failed` code at the CLI
//! boundary; the `#[error]` message is human-facing and may change without
//! affecting the contract.

/// Error returned by voice-activity detection (`vad_failed`).
///
/// Covers loading the embedded model and running the network — both backend
/// failures the caller cannot act on beyond disabling VAD.
#[derive(Debug, thiserror::Error)]
#[error("voice-activity detection failed: {0}")]
pub struct VadError(pub String);
