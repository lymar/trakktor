//! The GTCRN engine: a native port of an ultra-light masking enhancer.
//!
//! Forty-eight thousand parameters, and it runs at a hundredth of real time on
//! one CPU core. Next to [`unipase`](super::unipase) — five hundred and
//! forty-three *million* parameters and a GPU — that ratio is the whole reason
//! this engine exists, and the measurements say it does not pay for the
//! difference in quality.
//!
//! # What it does, and what it therefore cannot do
//!
//! This is a **masking** network, not a generative one. It works on the
//! spectrum, predicts a complex ratio mask, and multiplies. That means it can
//! only ever *attenuate* what is there: it cannot invent a band the telephone
//! never carried, and it cannot fill the hole a dropped packet left — a hole is
//! digital silence, and any mask times silence is silence. That is the one
//! place [`unipase`](super::unipase) still earns its keep.
//!
//! Within those limits the architecture is unusually careful about where it
//! spends its parameters:
//!
//! - the 192 bins above 2 kHz are folded into 64 equivalent-rectangular bands
//!   before the network sees them, and spread back out at the end;
//! - every convolution one band wide is preceded by concatenating each band
//!   with its two neighbours, so a narrow kernel still sees a neighbourhood;
//! - half the channels skip each temporal block entirely and are interleaved
//!   back in afterwards, so only half the width is ever convolved;
//! - the recurrences are split in two and run as separate halves.
//!
//! # Credits
//!
//! Ported from **GTCRN** by Xiaobin Rong et al. (MIT).

pub mod config;
#[cfg(feature = "enhance-runtime")]
pub mod download;
#[cfg(feature = "enhance-runtime")]
pub mod enhance;
#[cfg(feature = "enhance-runtime")]
pub mod runtime;
#[cfg(feature = "enhance-runtime")]
pub mod stft;

#[cfg(feature = "enhance-runtime")]
pub use download::{
    DEFAULT_MODEL, KNOWN_MODELS, ResolvedModel, download_size, resolve_model,
};
#[cfg(feature = "enhance-runtime")]
pub use enhance::{enhance_file, enhance_samples};
#[cfg(feature = "enhance-runtime")]
pub use runtime::CandleModel;
