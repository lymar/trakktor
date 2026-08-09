//! The candle runtime: loading the converted network and running one chunk.
//!
//! Full precision only, and the CPU is the intended device rather than the
//! fallback. Forty-eight thousand parameters over a hundredth of real time is
//! not work a GPU can help with: the kernel launches would cost more than the
//! arithmetic, which is the same reason the reference ships this as the model
//! you run on a phone.

pub mod net;
#[cfg(test)]
mod tests;

use std::path::Path;

use candle_core::DType;
pub use candle_core::Device;
use candle_nn::VarBuilder;

use super::{download::WEIGHTS_FILE, stft};
use crate::enhance::{EnhanceError, EnhanceModel, Runtime};

/// Wraps a candle failure as a checkpoint error, naming what was being read.
pub(super) fn model_err(what: &str, error: candle_core::Error) -> EnhanceError {
    EnhanceError::Checkpoint(format!("{what}: {error}"))
}

/// The network on candle, plus where the recording it is part-way through left
/// off.
pub struct CandleModel {
    net: net::Gtcrn,
    state: net::State,
    synth: stft::Synthesizer,
    /// Frames already consumed of the recording in progress.
    frames: usize,
}

impl CandleModel {
    /// Loads the converted checkpoint from `dir`.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when the weights are missing or do
    /// not match the geometry the engine expects.
    pub fn load(dir: &Path) -> Result<Self, EnhanceError> {
        let path = dir.join(WEIGHTS_FILE);
        if !path.is_file() {
            return Err(EnhanceError::Checkpoint(format!(
                "no {}",
                path.display()
            )));
        }
        // SAFETY: the checkpoint is memory-mapped read-only; candle requires
        // the file not to be mutated while mapped, which nothing here does.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[&path],
                DType::F32,
                &Device::Cpu,
            )
        }
        .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self {
            net: net::Gtcrn::load(vb)
                .map_err(|e| model_err("the network", e))?,
            state: net::State::default(),
            synth: stft::Synthesizer::new(),
            frames: 0,
        })
    }

    /// The network, for the parity tests.
    #[cfg(test)]
    pub(crate) fn net(&self) -> &net::Gtcrn { &self.net }
}

impl CandleModel {
    /// Starts a new recording: the recurrences forget, and so does the
    /// overlap-add.
    pub fn reset(&mut self) {
        self.state = net::State::default();
        self.synth = stft::Synthesizer::new();
        self.frames = 0;
    }

    /// Enhances `count` frames of `samples` starting where the last call left
    /// off, and returns the samples that are complete.
    ///
    /// The whole recording is passed every time: the analysis takes each
    /// frame's context from it rather than mirroring at a chunk edge, which is
    /// what keeps a piecewise run identical to one long pass.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Compute`] when the network fails.
    pub fn enhance_frames(
        &mut self,
        samples: &[f32],
        count: usize,
    ) -> Result<Vec<f32>, EnhanceError> {
        let spectrum = stft::analyze_range(samples, self.frames, count);
        let (real, imag) = self
            .net
            .enhance(
                &spectrum.real,
                &spectrum.imag,
                spectrum.frames,
                &mut self.state,
            )
            .map_err(|e| EnhanceError::Compute(e.to_string()))?;
        self.frames += count;
        Ok(self.synth.push(&stft::Spectrum {
            real,
            imag,
            frames: count,
        }))
    }

    /// The samples the overlap-add is still holding at the end.
    pub fn finish(&mut self) -> Vec<f32> { self.synth.finish() }
}

impl EnhanceModel for CandleModel {
    /// Enhances a whole short recording in one call — the shape the shared seam
    /// speaks. A long one goes through
    /// [`enhance_frames`](CandleModel::enhance_frames) instead, so that its
    /// pieces continue one another.
    fn enhance_window(
        &mut self,
        samples: &[f32],
        _lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        self.reset();
        let count = super::config::frames(samples.len());
        let mut wave = self.enhance_frames(samples, count)?;
        wave.extend(self.finish());
        wave.resize(samples.len(), 0.0);
        Ok(wave)
    }

    fn runtime(&self) -> Runtime { Runtime::Candle }
}
