//! Speech enhancement: a recording in, a cleaner recording out.
//!
//! The youngest of the speech domains, and the only one whose input and output
//! are both audio. Where [`asr`](crate::asr) turns a recording into text and
//! [`tts`](crate::tts) text into a recording, this one repairs the recording
//! itself: room noise, reverberation, a telephone band, the holes a dropped
//! packet leaves in a call.
//!
//! It has two consumers, and they want different things from it. A person wants
//! a file back — the same recording, cleaner. A recogniser wants a
//! preprocessing stage, and cares only about the 16 kHz mono signal the
//! acoustic model will see.
//!
//! Engines live one per module, the way they do under [`asr`](crate::asr) and
//! [`tts`](crate::tts):
//!
//! - [`gtcrn`] — an ultra-light masking network: 48 K parameters, a hundredth
//!   of real time on one CPU core;
//! - [`mpsenet`] — a transformer over the spectrum that decodes magnitude and
//!   phase apart: 2.3 M parameters, and the only masking engine that can repair
//!   a phase rather than only attenuate a magnitude;
//! - [`unipase`] — a four-network generative pipeline around a fine-tuned WavLM
//!   encoder: 546 M parameters, and the only engine that fills the holes a
//!   dropped packet leaves;
//! - [`resemble`] — two more, published together: a masking denoiser and a
//!   generative restorer, and the only two that run at **44.1 kHz** rather than
//!   16 kHz.
//!
//! What is shared sits here rather than in any of them: the failures
//! ([`error`]), and the seam a runtime implements plus the result it produces
//! ([`model`]). Neither depends on which network is behind the command.

pub mod error;
pub mod gtcrn;
#[cfg(feature = "enhance-runtime")]
pub mod model;
pub mod mpsenet;
pub mod resemble;
pub mod unipase;

pub use error::EnhanceError;
#[cfg(feature = "enhance-runtime")]
pub use model::{
    EnhanceModel, EnhanceOptions, EnhanceProgress, Enhanced, Precision,
    Progress, Runtime,
};
