//! The resemble-enhance engines: a native port of a denoiser and a generative
//! restorer, published together.
//!
//! Two networks, one download, and they are as different from each other as any
//! two engines in the domain:
//!
//! - **the denoiser** is a masking network on the spectrum — ten million
//!   parameters, one pass, deterministic. It predicts a gain and a **rotation**
//!   per point, which puts it between [`gtcrn`](super::gtcrn), whose complex
//!   mask moves magnitude and phase together, and [`mpsenet`](super::mpsenet),
//!   whose second head predicts a phase outright;
//! - **the enhancer** does not filter the recording at all. It reads a mel,
//!   walks a flow model from noise to a latent, and hands that to a vocoder
//!   which builds a waveform **out of noise**. Nothing of the input waveform
//!   reaches the output.
//!
//! # Both run at 44.1 kHz
//!
//! Every other engine in this domain works at 16 kHz, and `--sample-rate` there
//! resamples rather than widens ([`unipase`](super::unipase) has a bandwidth
//! extender that is deliberately not ported). These two are trained at the full
//! rate: what comes out carries the whole band, and for the enhancer that
//! includes band the recording did not have — it is a bandwidth extender by
//! construction, because it synthesises rather than filters.
//!
//! # The enhancer is not deterministic upstream
//!
//! It draws Gaussian noise twice per chunk and upstream draws both from torch's
//! global generator, which its command line never seeds — two runs give two
//! files. This port draws from its own stream ([`noise`]) fixed by a seed, so a
//! run repeats. That means the two cannot be compared sample by sample unless
//! the reference's own draws are fed in, which is exactly what the parity tests
//! do.
//!
//! # Credits
//!
//! Ported from **resemble-enhance** by Resemble AI (MIT), whose vocoder follows
//! UnivNet and LVCNet, with the anti-aliased activation from BigVGAN.

pub mod config;
#[cfg(feature = "enhance-runtime")]
pub mod download;
#[cfg(feature = "enhance-runtime")]
pub mod enhance;
#[cfg(feature = "enhance-runtime")]
pub mod mel;
#[cfg(feature = "enhance-runtime")]
pub mod noise;
#[cfg(feature = "enhance-runtime")]
pub mod runtime;
#[cfg(feature = "enhance-burn")]
pub mod runtime_burn;
#[cfg(feature = "enhance-runtime")]
pub mod solver;
#[cfg(feature = "enhance-runtime")]
pub mod stft;

#[cfg(feature = "enhance-runtime")]
pub use download::{ResolvedModel, download_size, model_dir, resolve_model};
#[cfg(feature = "enhance-runtime")]
pub use enhance::{enhance_file, enhance_samples};
#[cfg(feature = "enhance-runtime")]
pub use runtime::{DenoiserModel, EnhancerModel};
#[cfg(feature = "enhance-runtime")]
pub use solver::Method;

/// Everything the generative engine needs beyond the recording.
///
/// The defaults are the reference command line's, not its library's: they
/// differ, and what `resemble-enhance in_dir out_dir` does is the behaviour
/// worth matching.
#[derive(Debug, Clone, Copy)]
pub struct EnhancerSettings {
    /// Evaluations of the flow model per chunk. The solver spends them at its
    /// own rate — two per step under the midpoint rule.
    pub nfe: usize,
    /// How the flow is integrated.
    pub method: Method,
    /// How much of the **denoised** recording the flow model is conditioned
    /// on, against the recording as it is. At 0 the denoiser does not run at
    /// all.
    pub lambda: f32,
    /// How much of the walk's starting point is noise rather than the
    /// recording's own encoding. At 0 the walk starts from the recording; at 1
    /// it starts from noise and keeps only what the conditioning imposes.
    pub temperature: f32,
    /// Fixes the two Gaussian draws, and with them the whole run. Upstream has
    /// no equivalent — it draws from a generator it never seeds.
    pub seed: u64,
}

impl Default for EnhancerSettings {
    fn default() -> Self {
        Self {
            nfe: 64,
            method: Method::Midpoint,
            lambda: 1.0,
            temperature: 0.5,
            seed: 0,
        }
    }
}
