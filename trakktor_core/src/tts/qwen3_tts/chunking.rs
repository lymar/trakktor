//! How a decode is laid out in time: the attention window of the codec's
//! transformer stack, and the way a long run is split into chunks.
//!
//! Both come from the reference and both change the audio if they drift, so
//! they live once here and are shared by every runtime rather than copied into
//! each.

/// Frames decoded per chunk, and the frames of left context each chunk carries.
pub(super) const CHUNK_FRAMES: usize = 300;
pub(super) const LEFT_CONTEXT_FRAMES: usize = 25;

/// Step a chunk's length is rounded up to before it reaches a backend that
/// specializes its kernels per shape.
///
/// A run's last chunk holds whatever frames are left over, so its length is
/// effectively arbitrary — and on burn every new length means compiling and
/// autotuning the decoder's convolutions again, which costs far more than
/// decoding a few extra frames. Rounding to a step leaves at most five
/// shapes for a whole run. The decoder is causal, so the frames added on the
/// right cannot change the samples before them; they are decoded and dropped.
/// Measured: the waveform moves by less than 1e-6 (the reordering of a
/// reduction), while the stalls between pieces fall from tens of seconds to
/// none.
///
/// candle compiles nothing per shape, so its path decodes the exact span.
#[cfg(feature = "tts-burn")]
pub(super) const DECODE_ALIGN: usize = 64;

/// The length a chunk of `span` frames is decoded at.
#[cfg(feature = "tts-burn")]
pub(super) fn aligned_span(span: usize) -> usize {
    span.div_ceil(DECODE_ALIGN) * DECODE_ALIGN
}

/// Whether a frame may attend to another: itself and the `window - 1` frames
/// before it, and nothing after it.
pub(super) fn window_visible(query: usize, key: usize, window: usize) -> bool {
    key <= query && key + window > query
}

/// One chunk of a long decode: the frames to emit and the context before them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Chunk {
    /// First frame whose samples are kept.
    pub(super) start: usize,
    /// One past the last frame whose samples are kept.
    pub(super) end: usize,
    /// Frames of context decoded before `start` and then discarded.
    pub(super) context: usize,
}

impl Chunk {
    /// First frame fed to the network, context included.
    pub(super) fn context_start(self) -> usize { self.start - self.context }

    /// Frames fed to the network, context included.
    pub(super) fn span(self) -> usize { self.end - self.context_start() }
}

/// Splits `total` frames the way the reference splits them: fixed-size chunks,
/// each (after the first) primed with a fixed left context.
pub(super) fn chunk_plan(total: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total {
        let end = (start + CHUNK_FRAMES).min(total);
        // The first chunk has nothing to its left; later ones carry the fixed
        // context, bounded by how many frames actually precede them.
        let context = if start > LEFT_CONTEXT_FRAMES {
            LEFT_CONTEXT_FRAMES
        } else {
            start
        };
        chunks.push(Chunk {
            start,
            end,
            context,
        });
        start = end;
    }
    chunks
}

#[cfg(test)]
mod tests;
