//! The network's geometry, written down once and checked against the tensor
//! shapes when the model is loaded.
//!
//! Everything here is fixed by the published checkpoints. Both of them — the
//! one trained on VoiceBank+DEMAND and the one trained on the DNS Challenge
//! data — are the same network with the same shapes, so this is a description
//! of one architecture and not of a family.

/// The rate the network works at.
pub const SAMPLE_RATE: u32 = 16_000;

/// Transform length of the analysis, and the window length with it.
pub const N_FFT: usize = 400;
/// Hop between analysis frames.
pub const HOP: usize = 100;
/// Bins of the analysis spectrum.
pub const BINS: usize = N_FFT / 2 + 1;

/// Exponent the magnitude spectrum is raised to before the network sees it.
///
/// Power-law compression: it pulls the dynamic range of a spectrum into
/// something a convolution can work on, and the synthesis raises the result
/// back by the reciprocal.
pub const COMPRESS: f64 = 0.3;

/// Guard added inside the square root of the magnitude.
pub const MAG_EPS: f64 = 1e-9;
/// Guard added to the imaginary part before the phase's `atan2`.
pub const PHASE_EPS_IMAG: f64 = 1e-10;
/// Guard added to the real part before the phase's `atan2`.
///
/// **Five** orders of magnitude larger than the imaginary one, and not
/// symmetric — which is a quirk of the reference rather than a considered
/// choice, but it rotates every quiet bin it touches, so it is reproduced
/// exactly.
pub const PHASE_EPS_REAL: f64 = 1e-5;

/// Channels every convolution inside the network works in.
pub const CHANNELS: usize = 64;

/// Convolutions in each dense block.
pub const DENSE_DEPTH: usize = 4;
/// Kernel of a dense block's convolution, over time and frequency.
pub const DENSE_KERNEL: (usize, usize) = (2, 3);

/// Width of the grid the transformers see, after the encoder's stride-2
/// convolution over frequency: `⌊(BINS + 2 − 3)/2⌋ + 1`.
pub const CORE_WIDTH: usize = (BINS + 2 - 3) / 2 + 1;

/// Two-stage transformer blocks between the encoder and the decoders.
pub const TS_BLOCKS: usize = 4;
/// Attention heads in each of them.
pub const ATTENTION_HEADS: usize = 4;
/// Hidden width of each direction of the block's recurrence.
pub const GRU_HIDDEN: usize = CHANNELS * 2;

/// Epsilon of every layer norm in the transformers.
pub const NORM_EPS: f64 = 1e-5;
/// Epsilon of the instance norms that follow every convolution.
pub const INSTANCE_NORM_EPS: f64 = 1e-5;
/// Slope of the leaky rectifier between the recurrence and its projection.
pub const LEAKY_SLOPE: f64 = 0.01;

/// Ceiling of the mask the magnitude decoder predicts. The reference calls it
/// `beta`: the mask is `β · σ(slope · x)`, so it can amplify a bin as well as
/// attenuate it — up to twice.
pub const MASK_BETA: f64 = 2.0;

/// Seconds of audio one window covers.
///
/// The reference has no long-form inference at all: it runs a whole utterance
/// in one pass, and the utterances it was written for are seconds long. A
/// recording of any length has to be cut, and this is where.
///
/// Eight seconds is what the measurement asks for. The network's per-window
/// normalizations take their statistics over the whole window and its
/// attention spans it, so the result depends on *which* window a second of
/// speech lands in — measured, by anything from 1 % to 25 % in relative terms
/// between two placements. Its distance from the window's edge barely enters
/// into it: past the first quarter-second the difference stops shrinking.
/// Longer windows steady the statistics; the cost is quadratic in the
/// attention, and beyond eight seconds the model is past the lengths its
/// authors ran it at.
pub const WINDOW_SECONDS: usize = 8;

/// Seconds two consecutive windows share.
///
/// Since the edges are barely the weak part, this is not a margin to be thrown
/// away — it is long enough to fade one window into the next without a step in
/// level, and to give the quarter-second that *is* weak a weight below a half
/// while it is. Consecutive windows therefore start [`WINDOW_SECONDS`] −
/// [`OVERLAP_SECONDS`] apart, and a recording is processed 1.14 times rather
/// than twice.
pub const OVERLAP_SECONDS: usize = 1;

/// Frames a sample count produces, the way the analysis pads it.
#[must_use]
pub const fn frames(samples: usize) -> usize { samples / HOP + 1 }

/// Samples the synthesis returns for a frame count — what `torch.istft`
/// returns for a centred analysis, which is not in general the length that went
/// in.
#[must_use]
pub const fn samples(frames: usize) -> usize {
    if frames == 0 { 0 } else { (frames - 1) * HOP }
}

#[cfg(test)]
mod tests;
