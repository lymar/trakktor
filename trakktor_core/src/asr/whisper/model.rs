//! The model contract: what a runtime backend must provide.
//!
//! The decoding policy is runtime-agnostic. It drives a [`ForwardProvider`],
//! which owns the network weights and executes the encoder and decoder
//! forward passes on some backend. Everything the transcription pipeline
//! needs from the network is expressed here — logits in full precision, an
//! incremental self-attention cache, cache reordering for beam search, and
//! raw cross-attention scores for word-level timing — and nothing else.

#[cfg(test)]
mod tests;

use super::{
    error::WhisperError,
    feature::{MelBands, MelWindow},
    tokenizer::TokenId,
};

/// Model geometry, read from the checkpoint.
///
/// All official models share `n_audio_ctx = 1500` and `n_text_ctx = 448`;
/// the rest varies by model size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDims {
    /// Mel bands the encoder expects (80, or 128 for large-v3 and turbo).
    pub n_mels: usize,
    /// Encoder output positions per window (frames after the stride-2 conv).
    pub n_audio_ctx: usize,
    /// Encoder embedding width.
    pub n_audio_state: usize,
    /// Encoder attention heads.
    pub n_audio_head: usize,
    /// Encoder layers.
    pub n_audio_layer: usize,
    /// Vocabulary size (base tokens plus specials).
    pub n_vocab: usize,
    /// Maximum decoder sequence length.
    pub n_text_ctx: usize,
    /// Decoder embedding width.
    pub n_text_state: usize,
    /// Decoder attention heads.
    pub n_text_head: usize,
    /// Decoder layers.
    pub n_text_layer: usize,
}

impl ModelDims {
    /// Whether the model is multilingual, judged by the vocabulary size.
    pub fn is_multilingual(&self) -> bool { self.n_vocab >= 51865 }

    /// Number of languages in the model's vocabulary (99, or 100 for the
    /// vocabularies that add Cantonese).
    pub fn num_languages(&self) -> usize {
        self.n_vocab - 51765 - usize::from(self.is_multilingual())
    }

    /// The mel filterbank matching the encoder input.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidModel`] for a band count the pipeline
    /// has no filterbank for.
    pub fn mel_bands(&self) -> Result<MelBands, WhisperError> {
        match self.n_mels {
            80 => Ok(MelBands::Mel80),
            128 => Ok(MelBands::Mel128),
            other => Err(WhisperError::InvalidModel(format!(
                "unsupported mel band count: {other} (expected 80 or 128)"
            ))),
        }
    }
}

/// Decoder logits for the token positions fed in one forward call, in full
/// precision: `n_batch` sequences by `n_positions` fed positions by `n_vocab`.
#[derive(Debug, Clone)]
pub struct Logits {
    n_batch: usize,
    n_positions: usize,
    n_vocab: usize,
    data: Vec<f32>,
}

impl Logits {
    /// Wraps a flat row-major buffer of `n_batch * n_positions * n_vocab`
    /// values.
    pub fn new(
        n_batch: usize,
        n_positions: usize,
        n_vocab: usize,
        data: Vec<f32>,
    ) -> Self {
        assert_eq!(
            data.len(),
            n_batch * n_positions * n_vocab,
            "logits buffer must be n_batch * n_positions * n_vocab"
        );
        Self {
            n_batch,
            n_positions,
            n_vocab,
            data,
        }
    }

    /// Number of sequences.
    pub fn n_batch(&self) -> usize { self.n_batch }

    /// Number of token positions covered by this forward call.
    pub fn n_positions(&self) -> usize { self.n_positions }

    /// Vocabulary size.
    pub fn n_vocab(&self) -> usize { self.n_vocab }

    /// The distribution over the vocabulary at (`batch`, `position`).
    pub fn row(&self, batch: usize, position: usize) -> &[f32] {
        let start = (batch * self.n_positions + position) * self.n_vocab;
        &self.data[start..start + self.n_vocab]
    }

    /// The distribution at the last fed position of `batch` — what sampling
    /// looks at each step.
    pub fn last_position(&self, batch: usize) -> &[f32] {
        self.row(batch, self.n_positions - 1)
    }
}

/// Raw pre-softmax cross-attention scores of one full decoder forward:
/// `n_layers * n_heads` matrices of `n_tokens` rows by `n_frames` columns
/// (the encoder positions). Word-level timing aligns text to time with these.
#[derive(Debug, Clone)]
pub struct CrossQk {
    n_layers: usize,
    n_heads: usize,
    n_tokens: usize,
    n_frames: usize,
    data: Vec<f32>,
}

impl CrossQk {
    /// Wraps a flat buffer laid out `[layer][head][token][frame]`.
    pub fn new(
        n_layers: usize,
        n_heads: usize,
        n_tokens: usize,
        n_frames: usize,
        data: Vec<f32>,
    ) -> Self {
        assert_eq!(
            data.len(),
            n_layers * n_heads * n_tokens * n_frames,
            "cross-attention buffer must be layers * heads * tokens * frames"
        );
        Self {
            n_layers,
            n_heads,
            n_tokens,
            n_frames,
            data,
        }
    }

    /// Decoder layers covered.
    pub fn n_layers(&self) -> usize { self.n_layers }

    /// Heads per layer.
    pub fn n_heads(&self) -> usize { self.n_heads }

    /// Token positions covered.
    pub fn n_tokens(&self) -> usize { self.n_tokens }

    /// Encoder positions covered.
    pub fn n_frames(&self) -> usize { self.n_frames }

    /// The `n_tokens * n_frames` score matrix of (`layer`, `head`),
    /// row-major.
    pub fn head(&self, layer: usize, head: usize) -> &[f32] {
        let size = self.n_tokens * self.n_frames;
        let start = (layer * self.n_heads + head) * size;
        &self.data[start..start + size]
    }
}

/// A runtime backend executing the Whisper network.
///
/// The decoding policy calls this trait and nothing deeper. A session runs
/// one 30 s window: [`encode`](Self::encode) the window, then
/// [`begin_decode`](Self::begin_decode) → repeated
/// [`decode_step`](Self::decode_step) (with
/// [`rearrange_kv_cache`](Self::rearrange_kv_cache) between steps under beam
/// search) → [`end_decode`](Self::end_decode).
///
/// The caller mirrors the reference stepping: the first `decode_step` feeds
/// each sequence's whole initial token sequence, every later call feeds
/// exactly one token per sequence; the backend accumulates self-attention
/// state across calls and computes cross-attention state once per session.
pub trait ForwardProvider {
    /// Opaque encoded audio: the encoder output for one window, typically
    /// living on the backend's device.
    type AudioFeatures;

    /// The model geometry.
    fn dims(&self) -> &ModelDims;

    /// Encoder forward over one window of log-mel input.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidModel`] when the window does not match
    /// the model or the backend fails.
    fn encode(
        &mut self,
        mel_window: &MelWindow,
    ) -> Result<Self::AudioFeatures, WhisperError>;

    /// Starts a decoding session: `n_batch` parallel sequences, an empty
    /// self-attention cache, and cross-attention state computed from
    /// `features` once for the whole session.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidModel`] when the backend fails.
    fn begin_decode(
        &mut self,
        n_batch: usize,
        features: &Self::AudioFeatures,
    ) -> Result<(), WhisperError>;

    /// Decoder forward appending `step_tokens` — `n_batch` rows of `n_step`
    /// tokens each, flattened row-major — after the cached positions.
    /// Returns logits for exactly the fed positions.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidModel`] when the session state is
    /// missing or the backend fails.
    fn decode_step(
        &mut self,
        step_tokens: &[TokenId],
        n_batch: usize,
    ) -> Result<Logits, WhisperError>;

    /// Reorders the cached sequences so that row `i` continues from previous
    /// row `source_indices[i]` (beam-search bookkeeping). The identity
    /// permutation is a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidModel`] when the session state is
    /// missing or the backend fails.
    fn rearrange_kv_cache(
        &mut self,
        source_indices: &[usize],
    ) -> Result<(), WhisperError>;

    /// Ends the decoding session and releases its cache.
    fn end_decode(&mut self);

    /// One full decoder forward over a single complete sequence, outside any
    /// session, additionally returning the raw pre-softmax cross-attention
    /// scores of every decoder layer and head. Word-level timing depends on
    /// this call.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidModel`] when the backend fails.
    fn forward_with_cross_qk(
        &mut self,
        tokens: &[TokenId],
        features: &Self::AudioFeatures,
    ) -> Result<(Logits, CrossQk), WhisperError>;
}
