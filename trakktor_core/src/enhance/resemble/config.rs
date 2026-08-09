//! The geometry both networks are built to, and the one place it is written
//! down.
//!
//! Two things here are unlike every other engine in the domain. The first is
//! the rate: this pipeline runs at [`SAMPLE_RATE`] — **44.1 kHz**, the full
//! band, not the 16 kHz the others work at. The second follows from it: the
//! transforms are large (a 1680-point analysis every 420 samples for the
//! denoiser, a 2048-point one for the mel), because a wide band needs a long
//! window to resolve the bottom of it.

/// The rate both networks were trained at, and the rate everything here runs
/// at.
pub const SAMPLE_RATE: u32 = 44_100;

// --- The denoiser's transform ------------------------------------------------

/// Samples between analysis frames. Upstream calls it `hop_size` and derives
/// everything else from it; 420 samples is 9.52 ms.
pub const HOP: usize = 420;

/// Points in the denoiser's transform — four hops, as upstream builds it.
pub const N_FFT: usize = HOP * 4;

/// Frequency bins a real transform of that size has.
pub const BINS: usize = N_FFT / 2 + 1;

/// The guard under the magnitude before a phase is divided out of it.
pub const MAGPHASE_EPS: f32 = 1e-7;

/// The guard under a waveform's peak before it is divided by.
pub const PEAK_EPS: f32 = 1e-7;

// --- The mel the enhancer conditions on --------------------------------------

/// Points in the mel transform. Not the denoiser's: the mel is a separate
/// analysis of the same signal, at the same hop but through a longer window.
pub const MEL_N_FFT: usize = 2048;

/// Mel bands.
pub const MELS: usize = 128;

/// The first-difference filter applied before the mel transform.
pub const PREEMPHASIS: f32 = 0.97;

/// The floor a mel magnitude is clamped to before its logarithm is taken.
pub const MEL_MAGNITUDE_MIN: f32 = 1e-4;

/// Decibels of room left above 0 dB when the mel is normalized.
pub const MEL_HEADROOM_DB: f32 = 15.0;

// --- The dense UNet of the denoiser ------------------------------------------

/// Channels the UNet lifts its three input planes to.
pub const UNET_HIDDEN: usize = 16;

/// Encoder blocks, and decoder blocks; each halves or doubles both axes.
pub const UNET_BLOCKS: usize = 4;

/// Blocks between the encoder and the decoder, at the narrowest width.
pub const UNET_MIDDLE: usize = 2;

/// Channels in one normalization group. The group *count* varies with the
/// width, this does not.
pub const GROUP_CHANNELS: usize = 16;

/// What both axes must be a multiple of before the encoder can halve them four
/// times over.
pub const UNET_ALIGN: usize = 1 << UNET_BLOCKS;

// --- The latent autoencoder --------------------------------------------------

/// Width of the latent the flow model works in.
pub const LATENT: usize = 64;

/// Width inside the autoencoder's residual stacks.
pub const IRMAE_HIDDEN: usize = 1024;

/// Groups in the autoencoder's normalizations. A count, not a width — unlike
/// [`GROUP_CHANNELS`].
pub const IRMAE_GROUPS: usize = 32;

/// Residual blocks in the encoder, and in the decoder.
pub const IRMAE_BLOCKS: usize = 4;

/// Dilations of the four convolutions inside one residual block.
pub const IRMAE_DILATIONS: [usize; 4] = [1, 2, 4, 8];

/// Rank-minimizing projections at the end of the encoder.
pub const IRMAE_PROJECTIONS: usize = 4;

/// Channels the vocoder takes beyond the mel it reconstructs.
pub const VOCODER_EXTRA: usize = 32;

/// Width the decoder produces, and the vocoder consumes.
pub const VOCODER_INPUT: usize = MELS + VOCODER_EXTRA;

/// What the latent is multiplied by before the flow model sees it.
pub const Z_SCALE: f32 = 6.0;

// --- The flow model ----------------------------------------------------------

/// Layers in the velocity field.
pub const WN_LAYERS: usize = 30;

/// Width inside it.
pub const WN_HIDDEN: usize = 512;

/// Width of its convolutions.
pub const WN_KERNEL: usize = 3;

/// After this many layers the dilation returns to one.
pub const WN_DILATION_CYCLE: usize = 5;

/// Width of the sinusoidal embedding of the solver's time.
pub const TIME_EMB: usize = 128;

/// The largest exponent in that embedding: the frequencies are `10^p` for `p`
/// evenly spaced from zero to here.
pub const TIME_EMB_MAX_EXPONENT: f32 = 4.0;

/// Divisor of the solver's time mapping. Upstream solves for the base that
/// puts half the trajectory in the first `1/n` of the time, and this is `n`.
pub const TIME_MAPPING_DIVISOR: f64 = 4.0;

// --- The vocoder -------------------------------------------------------------

/// Channels through the vocoder's trunk.
pub const UNIVNET_CHANNELS: usize = 96;

/// Width of the noise the vocoder starts from.
pub const NOISE_CHANNELS: usize = 128;

/// Upsampling factors, in order. Their product is [`HOP`], which is what makes
/// one conditioning frame into one hop of waveform.
pub const STRIDES: [usize; 4] = [7, 5, 4, 3];

/// Dilations of the four location-variable convolutions in each block.
pub const LVC_DILATIONS: [usize; 4] = [1, 3, 9, 27];

/// Width of the kernel predictor's hidden layers.
pub const KPNET_HIDDEN: usize = 64;

/// Width of its convolutions, and of the kernels it predicts.
pub const KPNET_KERNEL: usize = 3;

/// Slope of the vocoder's leaky rectifiers.
pub const LEAKY_SLOPE: f64 = 0.2;

/// Slope of the kernel predictor's.
pub const KPNET_LEAKY_SLOPE: f64 = 0.1;

/// Dilations of the three anti-aliased layers in one AMP block.
pub const AMP_DILATIONS: [usize; 3] = [1, 3, 5];

/// Bounds the Snake activation's two learned parameters are clamped to.
pub const SNAKE_CLAMP: (f32, f32) = (1e-2, 50.0);

/// Ratio the anti-aliased activation resamples by, up and back down.
pub const RESAMPLE_RATIO: usize = 2;

/// Taps in its Kaiser-windowed sinc filter.
pub const RESAMPLE_TAPS: usize = 12;

/// Conditioning frames the vocoder pads with before it runs, and trims after.
pub const VOCODER_PAD: usize = 10;

/// Conditioning frames of context the vocoder needs on each side of a slice for
/// that slice to come out the same as it would in one pass.
///
/// Its receptive field is about fifteen frames: the kernel predictor reaches
/// twelve, and the four blocks together about three more once their dilations
/// are converted through the upsampling ratios. This is three times that, which
/// costs a little recomputation and removes any need to be exactly right.
pub const VOCODER_HALO: usize = 48;

/// Elements in one full-rate vocoder activation, which is what its slice length
/// is chosen to keep under.
///
/// The vocoder turns one conditioning frame into [`HOP`] samples across
/// [`UNIVNET_CHANNELS`] channels, so a thirty-second chunk in one pass is half
/// a gigabyte per activation and several gigabytes live. On a machine whose GPU
/// shares the system's memory that is not merely slow — it starves everything
/// else — so the run is cut into slices under this budget and the peak stops
/// depending on how long the recording is.
pub const VOCODER_BUDGET: usize = 16 << 20;

/// Conditioning frames one vocoder slice covers.
#[must_use]
pub fn vocoder_slice() -> usize {
    (VOCODER_BUDGET / (HOP * UNIVNET_CHANNELS)).max(1)
}

// --- Long form ---------------------------------------------------------------

/// Seconds in one chunk, as upstream's long-form path cuts them.
pub const CHUNK_SECONDS: f64 = 30.0;

/// Seconds two consecutive chunks share.
pub const OVERLAP_SECONDS: f64 = 1.0;

/// Zeros appended to a chunk before it is run, and dropped after. Upstream's
/// `npad`: a tenth of a second, enough that the transform's own tail does not
/// eat the end of the chunk.
pub const TAIL_PAD: usize = 441;

/// Samples in one chunk.
#[must_use]
pub fn chunk_samples() -> usize {
    (f64::from(SAMPLE_RATE) * CHUNK_SECONDS) as usize
}

/// Samples two consecutive chunks share.
#[must_use]
pub fn overlap_samples() -> usize {
    (f64::from(SAMPLE_RATE) * OVERLAP_SECONDS) as usize
}

/// Analysis frames the denoiser's transform yields for `samples` samples —
/// what `torch.stft` returns for a centred analysis, less the last frame,
/// which upstream drops.
#[must_use]
pub fn frames(samples: usize) -> usize { samples / HOP }

#[cfg(test)]
mod tests;
