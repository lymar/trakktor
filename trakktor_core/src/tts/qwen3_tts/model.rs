//! The contract a runtime fulfils.
//!
//! Everything above this line — the prompt layout, the sampler, the generation
//! loop — is arithmetic-free and shared; everything below it is a backend. The
//! seam is drawn so that no tensor type crosses it: a runtime takes token ids
//! and finished frames, and hands back logits and audio as plain host values.
//!
//! That costs one device round trip per sampling decision, which is inherent
//! anyway — the sampler is host-side and seeded, and every code it picks
//! decides what the next pass reads.

use super::{config::ModelConfig, error::Qwen3TtsError, prompt::Position};

/// The networks of one loaded checkpoint, driving one synthesis at a time.
///
/// A run is a sequence of calls: [`prime`](Self::prime) once, then
/// [`predict_residuals`](Self::predict_residuals) and
/// [`advance`](Self::advance) per frame, and finally
/// [`decode`](Self::decode). The talker's key/value cache and the state the
/// code predictor conditions on live inside the implementation; priming again
/// starts a fresh run.
pub trait SpeechModel {
    /// The checkpoint's configuration — its voices, languages, and geometry.
    fn config(&self) -> &ModelConfig;

    /// The sample rate of the waveform the codec decoder produces.
    fn sample_rate(&self) -> u32;

    /// Starts a run: forgets any cached state, feeds the laid-out prompt, and
    /// returns the codebook-0 logits of the first frame.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] on a backend failure.
    fn prime(
        &mut self,
        positions: &[Position],
    ) -> Result<Vec<f32>, Qwen3TtsError>;

    /// Fills codebooks 1.. of the frame the talker last scored, given the
    /// codebook-0 code that was picked from those logits.
    ///
    /// `pick` is called once per residual codebook with that codebook's logits
    /// and its index among the residuals.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] on a backend failure.
    fn predict_residuals(
        &mut self,
        first_code: u32,
        pick: &mut dyn FnMut(&[f32], usize) -> u32,
    ) -> Result<Vec<u32>, Qwen3TtsError>;

    /// Folds a finished frame back into the talker and takes one step,
    /// returning the next frame's codebook-0 logits.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] on a backend failure.
    fn advance(&mut self, frame: &[u32]) -> Result<Vec<f32>, Qwen3TtsError>;

    /// Turns finished frames — one row of codes per frame — into a mono
    /// waveform at [`sample_rate`](Self::sample_rate).
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] on a backend failure.
    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>, Qwen3TtsError>;
}
