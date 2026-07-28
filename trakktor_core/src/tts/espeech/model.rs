//! The contract a runtime fulfils.
//!
//! Everything above this line — preparing the reference, tokenizing, estimating
//! the duration, choosing the timesteps, drawing the noise, joining the
//! pieces — is arithmetic-free and shared; everything below it is a backend.
//! As in the other synthesis engine, no tensor type crosses the seam.
//!
//! Where the seam sits is a performance decision, not a logical one. The state
//! the solver refines stays **inside** the runtime between steps: it is the
//! size of the whole utterance, and moving it to the host and back thirty-two
//! times would be thirty-two device synchronizations for no gain. What stays
//! outside is everything that decides *what* to compute.

use super::{config::DitConfig, error::EspeechError, mel::MelBasis};

/// The networks of one loaded checkpoint, driving one utterance at a time.
///
/// A piece is spoken as: [`prepare`](Self::prepare) once, then
/// [`step`](Self::step) per solver step, then [`finish`](Self::finish). The
/// conditioning, the cached text encoding, and the solver state live inside the
/// implementation; preparing again starts a fresh utterance.
pub trait SpeechModel {
    /// The geometry the checkpoint was loaded with.
    fn config(&self) -> &DitConfig;

    /// The analysis basis for the mel spectrogram, as the vocoder ships it.
    fn mel_basis(&self) -> &MelBasis;

    /// The sample rate of the waveform the vocoder produces.
    fn sample_rate(&self) -> u32;

    /// Starts an utterance.
    ///
    /// `cond` is the reference's log-mel as `[cond_frames, mel]` in row-major
    /// order, `text` the character ids of the reference transcript followed by
    /// the text to speak, `frames` the length to generate (the reference
    /// included), and `noise` the starting point as `[frames, mel]`.
    ///
    /// This is where the work that does not change between steps belongs: the
    /// text encoder runs once for each guidance branch, and the part of the
    /// input projection that only sees the conditioning is folded into a
    /// constant.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::Checkpoint`] on a backend failure.
    fn prepare(
        &mut self,
        cond: &[f32],
        cond_frames: usize,
        text: &[u32],
        frames: usize,
        noise: &[f32],
    ) -> Result<(), EspeechError>;

    /// Takes one Euler step of size `dt` from time `t`, guided by
    /// `cfg_strength` (zero runs the conditional branch alone).
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::Checkpoint`] on a backend failure.
    fn step(
        &mut self,
        t: f32,
        dt: f32,
        cfg_strength: f32,
    ) -> Result<(), EspeechError>;

    /// Finishes the utterance: splices the conditioning back over its own
    /// frames, drops the first `cut_frames`, and vocodes the rest.
    ///
    /// The cut is one frame short of the conditioning on purpose — see
    /// [`Reference::cut_frames`](super::reference::Reference::cut_frames).
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::Checkpoint`] on a backend failure.
    fn finish(&mut self, cut_frames: usize) -> Result<Vec<f32>, EspeechError>;
}
