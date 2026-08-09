//! The network's geometry, written down once and checked against the tensor
//! shapes when the model is loaded.
//!
//! Everything here is fixed by the published checkpoint. GTCRN is one network,
//! not a family: there is no width to derive and no depth to infer.

/// The rate the network works at.
pub const SAMPLE_RATE: u32 = 16_000;

/// Transform length of the analysis.
pub const N_FFT: usize = 512;
/// Hop between analysis frames.
pub const HOP: usize = 256;
/// Bins of the analysis spectrum.
pub const BINS: usize = N_FFT / 2 + 1;

/// Bins below this are passed to the network as they are; above it they are
/// folded into equivalent-rectangular bands.
///
/// The split is what makes the network small: speech detail lives low, and the
/// 192 bins above 2 kHz carry so little independent information that 64 bands
/// hold all of it.
pub const ERB_LOW_BINS: usize = 65;
/// Bands the bins above the split are folded into.
pub const ERB_BANDS: usize = 64;
/// Width of the banded spectrum the network sees: the low bins plus the bands.
pub const BANDED: usize = ERB_LOW_BINS + ERB_BANDS;

/// Neighbouring bands each subband-feature extraction step concatenates.
pub const SFE_KERNEL: usize = 3;

/// Channels the encoder works in.
pub const CHANNELS: usize = 16;
/// Width of the grid the recurrent core sees, after two stride-2 convolutions.
pub const CORE_WIDTH: usize = 33;
/// Grouped temporal convolutions in the encoder, and in the decoder.
pub const GT_BLOCKS: usize = 3;
/// Dilations of those blocks, in encoder order; the decoder mirrors them.
pub const GT_DILATIONS: [usize; GT_BLOCKS] = [1, 2, 5];
/// Kernel of the grouped temporal convolution, over time and frequency.
pub const GT_KERNEL: (usize, usize) = (3, 3);

/// Dual-path recurrent blocks between encoder and decoder.
pub const DP_BLOCKS: usize = 2;
/// Hidden width of the dual-path blocks.
pub const DP_HIDDEN: usize = 16;

/// Epsilon of the dual-path layer norms.
pub const DP_NORM_EPS: f64 = 1e-8;
/// Epsilon the reference's batch norms were trained with, needed to fold them.
pub const BN_EPS: f64 = 1e-5;
/// Guard inside the magnitude the front end computes.
pub const MAG_EPS: f64 = 1e-12;

/// Seconds of audio one pass covers.
///
/// The network is causal and every piece of state it carries is threaded from
/// one chunk to the next, so this is a memory budget and nothing else: the
/// dual-path block holds the whole time-by-frequency grid at once, and an hour
/// of audio is 225 000 frames of it. Changing this changes peak memory and
/// nothing about the result.
pub const CHUNK_SECONDS: usize = 30;

/// Frames a sample count produces, the way the analysis pads it.
#[must_use]
pub const fn frames(samples: usize) -> usize { samples / HOP + 1 }

#[cfg(test)]
mod tests;
