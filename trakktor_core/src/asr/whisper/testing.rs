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
        _tokens: &[TokenId],
        _features: &(),
    ) -> Result<(Logits, CrossQk), WhisperError> {
        Err(WhisperError::InvalidModel("not scripted".into()))
    }
}

/// A logits row with one dominant token.
pub(crate) fn peak(n_vocab: usize, index: TokenId, height: f32) -> Vec<f32> {
    let mut row = vec![0.0f32; n_vocab];
    row[index as usize] = height;
    row
}
