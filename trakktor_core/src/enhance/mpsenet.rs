//! The MP-SENet engine: a native port of a magnitude-and-phase enhancer.
//!
//! The third engine of the domain, and the one that sits between the other
//! two: two and a quarter million parameters against
//! [`gtcrn`](super::gtcrn)'s forty-eight thousand and
//! [`unipase`](super::unipase)'s five hundred and forty-six million.
//!
//! # What makes it different from the other masking engine
//!
//! [`gtcrn`](super::gtcrn) predicts a complex mask and multiplies the spectrum
//! by it, which moves magnitude and phase together — a complex multiply cannot
//! change one without changing the other. This network **decodes them
//! separately**: one head predicts a gain per bin, the other predicts a phase
//! outright, as the two components of an arctangent. Nothing ties the phase it
//! produces to the phase that came in.
//!
//! That is the whole of the difference, and it is what it can do that a
//! multiply cannot: a mask can only attenuate, and it can never repair a phase
//! that noise has scrambled. It still cannot fill a hole — a bin whose
//! magnitude is zero stays zero however good the phase is — so packet loss
//! remains [`unipase`](super::unipase)'s.
//!
//! # Shape of the network
//!
//! A dense encoder over the 400-point spectrum, four two-stage blocks that
//! attend along frequency and then along time, and the two decoders. The
//! blocks are where the parameters and most of the arithmetic are: each holds
//! two transformers, and each transformer replaces the usual feed-forward
//! layer with a bidirectional recurrence.
//!
//! Two consequences follow from the attention spanning the whole window, and
//! both show up in the driver ([`enhance`]): the network cannot stream, and a
//! long recording has to be cut into windows whose results are cross-faded
//! together.
//!
//! # Credits
//!
//! Ported from **MP-SENet** by Ye-Xin Lu, Yang Ai and Zhen-Hua Ling (MIT).

pub mod config;
#[cfg(feature = "enhance-runtime")]
pub mod download;
#[cfg(feature = "enhance-runtime")]
pub mod enhance;
#[cfg(feature = "enhance-runtime")]
pub mod runtime;
#[cfg(feature = "enhance-burn")]
pub mod runtime_burn;
#[cfg(feature = "enhance-runtime")]
pub mod stft;

#[cfg(feature = "enhance-runtime")]
pub use download::{
    DEFAULT_MODEL, KNOWN_MODELS, ResolvedModel, download_size, model_dir,
    resolve_model,
};
#[cfg(feature = "enhance-runtime")]
pub use enhance::{enhance_file, enhance_samples};
#[cfg(feature = "enhance-runtime")]
pub use runtime::CandleModel;
