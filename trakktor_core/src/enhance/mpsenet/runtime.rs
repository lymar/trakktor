//! The candle runtime: loading the converted network and running one window.
//!
//! Full precision is the default and the only mode parity is claimed in — the
//! published checkpoints are `f32` throughout and the reference offers no half
//! precision to be compared against. `f16` is available here as a speed
//! option; it is a different computation of the same model.

pub mod net;
#[cfg(test)]
mod tests;

use std::path::Path;

use candle_core::DType;
pub use candle_core::Device;
use candle_nn::VarBuilder;

use super::{download::WEIGHTS_FILE, enhance, stft};
use crate::enhance::{
    EnhanceError, EnhanceModel, Precision, Runtime,
    mpsenet::stft::{Prediction, Spectrum},
};

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

/// The network on candle.
pub struct CandleModel {
    net: net::Mpsenet,
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
            net: net::Mpsenet::load(vb)
                .map_err(|e| model_err("the network", e))?,
        })
    }

    /// The device the network is on.
    #[must_use]
    pub fn device(&self) -> &Device { self.net.device() }

    /// The network, for the parity tests.
    #[cfg(test)]
    pub(crate) fn net(&self) -> &net::Mpsenet { &self.net }

    /// What the network predicts for one analysed window.
    pub(super) fn predict(
        &self,
        spectrum: &Spectrum,
    ) -> Result<Prediction, EnhanceError> {
        let (mask, phase_real, phase_imag) = self
            .net
            .predict(&spectrum.magnitude, &spectrum.phase, spectrum.frames)
            .map_err(|e| EnhanceError::Compute(e.to_string()))?;
        Ok(Prediction {
            mask,
            phase_real,
            phase_imag,
        })
    }
}

impl EnhanceModel for CandleModel {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        _lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        enhance::enhance_window(samples, |spectrum| {
            Ok(stft::apply(spectrum, &self.predict(spectrum)?))
        })
    }

    fn runtime(&self) -> Runtime { Runtime::Candle }
}
