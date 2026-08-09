//! The pipeline's geometry, and the constants the reference keeps in prose.
//!
//! Every number here is read off the published checkpoints (each carries its
//! own `cfg`), written down once, and checked against the tensor shapes when a
//! model is loaded. Nothing is inferred at run time: the pipeline is one
//! published set of four networks, not a family.

/// The rate the whole pipeline works at. Input at any other rate is resampled
/// to this before the encoder and back afterwards.
pub const SAMPLE_RATE: u32 = 16_000;

/// Samples per frame of the encoder and of the vocoder alike — the product of
/// the convolutional extractor's strides, and the vocoder's hop.
pub const HOP: usize = 320;

/// Samples the extractor's receptive field overhangs a whole number of hops
/// by. The reference pads every input up to `k · 320 + 80` so the extractor and
/// the vocoder agree on the frame count.
pub const PAD_REMAINDER: usize = 80;

/// The convolutional feature extractor: `(channels, kernel, stride)` per block.
/// In `layer_norm` mode every block is convolution → layer norm → GELU, and no
/// convolution carries a bias.
pub const CONV_LAYERS: [(usize, usize, usize); 7] = [
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
    (512, 2, 2),
];

/// Width of the transformer.
pub const ENCODER_DIM: usize = 1024;
/// Depth of the transformer.
pub const ENCODER_LAYERS: usize = 24;
/// Attention heads per layer.
pub const ATTENTION_HEADS: usize = 16;
/// Width of the feed-forward block.
pub const FFN_DIM: usize = 4096;

/// Kernel of the convolutional positional embedding.
pub const CONV_POS: usize = 128;
/// Groups of the convolutional positional embedding.
pub const CONV_POS_GROUPS: usize = 16;

/// Buckets of the relative position bias.
pub const NUM_BUCKETS: usize = 320;
/// Distance beyond which the bias stops growing.
pub const MAX_DISTANCE: usize = 800;

/// Width of the gate that scales the relative position bias per query.
pub const GREP_DIM: usize = 8;

/// The two encoder layers the pipeline taps: the acoustic one, which carries
/// what the recording sounds like, and the deep one, which carries what is
/// being said. `1` and `24` are 1-based over the transformer's layers.
pub const TAP_ACOUSTIC: usize = 1;
/// The deep tap. See [`TAP_ACOUSTIC`].
pub const TAP_PHONETIC: usize = 24;

/// Epsilon of every layer norm in the encoder.
pub const ENCODER_NORM_EPS: f64 = 1e-5;
/// Epsilon of the normalization applied to each tapped layer's output.
pub const TAP_NORM_EPS: f64 = 1e-6;

/// Width of the adapter and of the vocoder backbone alike.
pub const BACKBONE_DIM: usize = 1024;
/// Width of the ConvNeXt mixer in both backbones.
pub const BACKBONE_FF: usize = 3072;
/// ConvNeXt blocks in both backbones.
pub const BACKBONE_LAYERS: usize = 12;
/// Residual blocks around the attention in the backbone's positional network.
/// The reference builds `num_res / 2` before it and as many after.
pub const POS_NET_RES: usize = 4;
/// Attention blocks in the middle of the positional network.
pub const POS_NET_ATTN: usize = 1;
/// Groups of the positional network's group norms.
pub const POS_NET_GROUPS: usize = 32;
/// Epsilon of every normalization in the backbones.
pub const BACKBONE_NORM_EPS: f64 = 1e-6;

/// Transform length of the vocoder's inverse-STFT head.
pub const N_FFT: usize = 1280;

/// Bounds the vocoder clamps its predicted log-magnitudes to before the
/// exponential — the reference's own guard against a spectrum that would
/// explode into a click.
pub const LOG_MAGNITUDE_RANGE: (f64, f64) = (-20.0, 5.0);

/// Packet length the loss detector works in, in milliseconds.
pub const PACKET_MS: usize = 20;
/// Below this a sample counts as silent.
pub const PACKET_SILENCE: f32 = 1e-7;
/// A packet this silent, or more, counts as lost.
pub const PACKET_LOST_RATIO: f32 = 0.99;

/// Seconds of audio the reference runs the networks over at a time.
pub const WINDOW_SECONDS: usize = 8;
/// Seconds between the starts of two consecutive windows.
pub const HOP_SECONDS: usize = 4;

/// The rate the bandwidth extender works at.
pub const POSTNET_SAMPLE_RATE: u32 = 48_000;
/// Transform length of the bandwidth extender (32 ms at 48 kHz).
pub const POSTNET_N_FFT: usize = 1536;
/// Hop of the bandwidth extender's transform.
pub const POSTNET_HOP: usize = POSTNET_N_FFT / 2;
/// Frequency bands the extender splits its spectrum into.
pub const POSTNET_BANDS: usize = 3;
/// Blocks of the extender.
pub const POSTNET_LAYERS: usize = 5;
/// Hidden units of each direction of the extender's BLSTMs.
pub const POSTNET_LSTM_UNITS: usize = 100;
/// Attention heads in each extender block.
pub const POSTNET_HEADS: usize = 4;
/// Channels the extender's attention projects a query and a key to.
pub const POSTNET_QK_CHANNELS: usize = 2;
/// Embedding width of the extender.
pub const POSTNET_EMB_DIM: usize = 48;
/// Kernel the extender unfolds and deconvolves with.
pub const POSTNET_EMB_KS: usize = 4;
/// Epsilon of the extender's normalizations.
pub const POSTNET_EPS: f64 = 1e-5;
/// The band the low half of the extender's output is faded in over: everything
/// up to `cutoff - transition` is the input's own spectrum, everything above
/// `cutoff` is the network's.
pub const POSTNET_CUTOFF_HZ: f64 = 8000.0;
/// Width of that fade, in hertz.
pub const POSTNET_TRANSITION_HZ: f64 = 800.0;

/// Frames the extender's spectrum has per second — used only for sizing.
#[must_use]
pub fn postnet_bins() -> usize { POSTNET_N_FFT / 2 + 1 }

/// Bins of the vocoder's spectrum.
#[must_use]
pub const fn bins() -> usize { N_FFT / 2 + 1 }

/// Head width of the encoder's attention.
#[must_use]
pub const fn head_dim() -> usize { ENCODER_DIM / ATTENTION_HEADS }

/// The length the extractor wants a window brought to: `⌊n/320⌋ · 320 + 80`.
///
/// Note that this **shortens** a window whose remainder is already past 80 —
/// the reference's own arithmetic, which pads by a negative number there and so
/// trims. Reproduced rather than corrected: the frame count either way is
/// `⌊n/320⌋`, and a run that disagreed with the reference about the last 200
/// samples of a window would disagree about every sample after it.
#[must_use]
pub const fn aligned_len(samples: usize) -> usize {
    if samples % HOP == PAD_REMAINDER {
        return samples;
    }
    samples / HOP * HOP + PAD_REMAINDER
}

#[cfg(test)]
mod tests;
