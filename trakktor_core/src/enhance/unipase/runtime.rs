//! The candle runtime: loading the converted pipeline and running one window.
//!
//! Full precision is the default and the only mode parity is claimed in — the
//! published checkpoints are `f32` throughout and the reference offers no half
//! precision to be compared against. `f16` is available on this runtime as a
//! speed option; it is a different computation of the same model.

pub mod net;
#[cfg(test)]
mod tests;

use std::path::Path;

pub use candle_core::Device;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use super::{
    config::{HOP, PAD_REMAINDER},
    download::WEIGHTS_FILE,
    istft,
};
use crate::enhance::{EnhanceError, EnhanceModel, Precision, Runtime};

/// Creates the compute device, reporting a build without Metal as a validation
/// error rather than falling back silently.
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidOptions`] when Metal is asked for and this
/// build has no Metal support, and [`EnhanceError::Checkpoint`] when the device
/// cannot be created.
pub fn device(metal: bool) -> Result<Device, EnhanceError> {
    if !metal {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "enhance-metal")]
    {
        Device::new_metal(0).map_err(|e| {
            EnhanceError::Checkpoint(format!("opening the Metal device: {e}"))
        })
    }
    #[cfg(not(feature = "enhance-metal"))]
    Err(EnhanceError::InvalidOptions(
        "this build has no Metal support; rebuild with the `metal` feature, \
         or use `--device cpu`"
            .into(),
    ))
}

/// Wraps a candle failure as a checkpoint error, naming what was being read.
pub(super) fn model_err(what: &str, error: candle_core::Error) -> EnhanceError {
    EnhanceError::Checkpoint(format!("{what}: {error}"))
}

/// The pipeline on candle.
pub struct CandleModel {
    encoder: net::Encoder,
    adapter: net::Adapter,
    vocoder: net::Vocoder,
    device: Device,
    dtype: DType,
}

impl CandleModel {
    /// Loads the converted checkpoint from `dir` onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when the weights are missing or do
    /// not match the geometry the engine expects.
    pub fn load(
        dir: &Path,
        device: Device,
        precision: Precision,
    ) -> Result<Self, EnhanceError> {
        let path = dir.join(WEIGHTS_FILE);
        if !path.is_file() {
            return Err(EnhanceError::Checkpoint(format!(
                "no {}",
                path.display()
            )));
        }
        let dtype = match precision {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
        };
        // SAFETY: the checkpoint is memory-mapped read-only; candle requires
        // the file not to be mutated while mapped, which nothing here does.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&path], dtype, &device)
        }
        .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self {
            encoder: net::Encoder::load(vb.pp("wavlm"))
                .map_err(|e| model_err("the encoder", e))?,
            adapter: net::Adapter::load(vb.pp("adapter"))
                .map_err(|e| model_err("the adapter", e))?,
            vocoder: net::Vocoder::load(vb.pp("vocoder"))
                .map_err(|e| model_err("the vocoder", e))?,
            device,
            dtype,
        })
    }

    /// The device the pipeline is on.
    #[must_use]
    pub fn device(&self) -> &Device { &self.device }
}

impl EnhanceModel for CandleModel {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        debug_assert_eq!(samples.len() % HOP, PAD_REMAINDER);
        let frames = samples.len() / HOP;
        let run = || -> candle_core::Result<Vec<f32>> {
            let input =
                Tensor::from_slice(samples, (1, samples.len()), &self.device)?
                    .to_dtype(self.dtype)?;
            let (acoustic, phonetic) = self.encoder.features(&input, lost)?;
            let adapted = self.adapter.forward(&acoustic, &phonetic)?;
            self.vocoder.spectrum(&adapted)
        };
        let head = run().map_err(|e| EnhanceError::Compute(e.to_string()))?;
        Ok(istft::spectrum_to_wave(&head, frames))
    }

    fn runtime(&self) -> Runtime { Runtime::Candle }
}
