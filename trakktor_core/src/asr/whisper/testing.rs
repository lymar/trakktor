//! Shared test support for the Whisper engine (compiled only for tests).

use super::{
    error::WhisperError,
    feature::MelWindow,
    model::{CrossQk, ForwardProvider, Logits, ModelDims},
    tokenizer::TokenId,
};

/// A provider that replays scripted logits: call `n` fills every fed
/// position of row `r` with `script[n][r]`.
pub(crate) struct FakeProvider {
    pub(crate) dims: ModelDims,
    pub(crate) script: Vec<Vec<Vec<f32>>>,
    pub(crate) calls: usize,
    pub(crate) rearranges: Vec<Vec<usize>>,
    pub(crate) sessions: usize,
    /// When set to `(n_frames, offset)`, `forward_with_cross_qk` succeeds:
    /// uniform logits, and one head whose attention row `t` peaks at frame
    /// `offset + 2 * t` — a synthetic diagonal alignment.
    pub(crate) cross_qk_diagonal: Option<(usize, usize)>,
}

impl FakeProvider {
    pub(crate) fn new(n_vocab: usize, script: Vec<Vec<Vec<f32>>>) -> Self {
        Self {
            dims: ModelDims {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 8,
                n_audio_head: 2,
                n_audio_layer: 1,
                n_vocab,
                n_text_ctx: 448,
                n_text_state: 8,
                n_text_head: 2,
                n_text_layer: 1,
            },
            script,
            calls: 0,
            rearranges: Vec::new(),
            sessions: 0,
            cross_qk_diagonal: None,
        }
    }
}

impl ForwardProvider for FakeProvider {
    type AudioFeatures = ();

    fn dims(&self) -> &ModelDims { &self.dims }

    fn encode(&mut self, _mel_window: &MelWindow) -> Result<(), WhisperError> {
        Ok(())
    }

    fn begin_decode(
        &mut self,
        _n_batch: usize,
        _features: &(),
    ) -> Result<(), WhisperError> {
        self.sessions += 1;
        Ok(())
    }

    fn decode_step(
        &mut self,
        step_tokens: &[TokenId],
        n_batch: usize,
    ) -> Result<Logits, WhisperError> {
        let rows = &self.script[self.calls];
        self.calls += 1;
        let n_positions = step_tokens.len() / n_batch;
        let mut data =
            Vec::with_capacity(n_batch * n_positions * self.dims.n_vocab);
        for row in rows.iter().take(n_batch) {
            for _ in 0..n_positions {
                data.extend_from_slice(row);
            }
        }
        Ok(Logits::new(n_batch, n_positions, self.dims.n_vocab, data))
    }

    fn rearrange_kv_cache(
        &mut self,
        source_indices: &[usize],
    ) -> Result<(), WhisperError> {
        self.rearranges.push(source_indices.to_vec());
        Ok(())
    }

    fn end_decode(&mut self) {}

    fn forward_with_cross_qk(
        &mut self,
        tokens: &[TokenId],
        _features: &(),
    ) -> Result<(Logits, CrossQk), WhisperError> {
        let Some((n_frames, offset)) = self.cross_qk_diagonal else {
            return Err(WhisperError::InvalidModel("not scripted".into()));
        };
        let n_tokens = tokens.len();
        let logits = Logits::new(
            1,
            n_tokens,
            self.dims.n_vocab,
            vec![0.0; n_tokens * self.dims.n_vocab],
        );
        // Smooth rows peaking on the diagonal: every column carries distinct
        // values across tokens, so the standardization never divides by a
        // zero deviation.
        let mut qk = vec![0.0f32; n_tokens * n_frames];
        for t in 0..n_tokens {
            let peak = (offset + 2 * t).min(n_frames - 1) as f32;
            for f in 0..n_frames {
                qk[t * n_frames + f] = -0.01 * (f as f32 - peak).abs();
            }
        }
        Ok((logits, CrossQk::new(1, 1, n_tokens, n_frames, qk)))
    }
}

/// A logits row with one dominant token.
pub(crate) fn peak(n_vocab: usize, index: TokenId, height: f32) -> Vec<f32> {
    let mut row = vec![0.0f32; n_vocab];
    row[index as usize] = height;
    row
}
