//! The burn-backed UniPASE runtime.
//!
//! An alternative [`EnhanceModel`] implementation on [burn](burn), selectable
//! at run time next to the candle one. Backends: ndarray on the CPU, and wgpu
//! with MSL-compiled kernels on Metal, with operator fusion and autotuning.
//!
//! **This runtime computes in f32 only**, on either device: the ndarray backend
//! has no half-precision element at all, and the published checkpoints are f32
//! throughout, so full precision is also the mode the two runtimes are compared
//! in.
//!
//! The checkpoint is the same converted file the candle runtime loads: tensors
//! are read through candle's safetensors reader and converted through f32 into
//! tensors of the target backend. The windowing, the packet-loss detector and
//! the inverse transform are shared host-side code, so the two runtimes cannot
//! disagree about the model's input or about the last step of its output.
//!
//! The backend choice is erased behind a boxed [`EnhanceModel`], so the driver
//! sees one type.

pub mod net;
#[cfg(test)]
mod tests;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{Tensor, TensorData, backend::Backend},
};

use super::{
    config::{HOP, PAD_REMAINDER},
    download::WEIGHTS_FILE,
    error::UnipaseError,
    istft,
    model::{EnhanceModel, Precision, Runtime},
    runtime::model_err,
};

/// Lazy access to the converted checkpoint through candle's safetensors reader.
pub struct Weights(candle_core::safetensors::MmapedSafetensors);

impl Weights {
    /// Opens the converted checkpoint.
    fn open(path: &Path) -> Result<Self, UnipaseError> {
        if !path.is_file() {
            return Err(UnipaseError::Checkpoint(format!(
                "no {}",
                path.display()
            )));
        }
        // SAFETY: the checkpoint is memory-mapped read-only; candle requires
        // the file not to be mutated while mapped, which nothing here does.
        let inner =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
                .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self(inner))
    }

    /// The named tensor as f32 values and its shape.
    pub fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), UnipaseError> {
        let tensor = self
            .0
            .load(key, &candle_core::Device::Cpu)
            .map_err(|e| model_err(key, e))?;
        let dims = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err(key, e))?;
        Ok((values, dims))
    }
}

/// The pipeline on one burn backend.
pub struct BurnModel<B: Backend> {
    encoder: net::Encoder<B>,
    adapter: net::Adapter<B>,
    vocoder: net::Vocoder<B>,
    device: B::Device,
}

impl<B: Backend> BurnModel<B> {
    /// Loads the converted checkpoint from `dir` onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`UnipaseError::Checkpoint`] when the weights are missing or do
    /// not match the geometry the engine expects.
    pub fn load(dir: &Path, device: B::Device) -> Result<Self, UnipaseError> {
        let weights = Weights::open(&dir.join(WEIGHTS_FILE))?;
        Ok(Self {
            encoder: net::Encoder::load(&weights, &device)?,
            adapter: net::Adapter::load(&weights, &device)?,
            vocoder: net::Vocoder::load(&weights, &device)?,
            device,
        })
    }
}

impl<B: Backend> EnhanceModel for BurnModel<B> {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        lost: &[bool],
    ) -> Result<Vec<f32>, UnipaseError> {
        debug_assert_eq!(samples.len() % HOP, PAD_REMAINDER);
        let frames = samples.len() / HOP;
        let input: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(samples.to_vec(), [1, samples.len()]),
            &self.device,
        );
        let (acoustic, phonetic) =
            self.encoder.features(input, lost, &self.device);
        let adapted = self.adapter.forward(acoustic, phonetic);
        let head = self.vocoder.spectrum(adapted);
        Ok(istft::spectrum_to_wave(&head, frames))
    }

    fn runtime(&self) -> Runtime { Runtime::Burn }
}

/// Loads the pipeline on the burn CPU backend (ndarray).
///
/// # Errors
///
/// Returns [`UnipaseError::InvalidOptions`] for half precision; otherwise see
/// [`BurnModel::load`].
pub fn load_cpu(
    dir: &Path,
    precision: Precision,
) -> Result<Box<dyn EnhanceModel>, UnipaseError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnModel::<NdArray<f32>>::load(
            dir,
            NdArrayDevice::Cpu,
        )?)),
        Precision::F16 => Err(half_precision_unavailable()),
    }
}

/// Loads the pipeline on the burn Metal backend (wgpu with MSL-compiled
/// kernels).
///
/// # Errors
///
/// Returns [`UnipaseError::InvalidOptions`] for half precision; otherwise see
/// [`BurnModel::load`].
pub fn load_metal(
    dir: &Path,
    precision: Precision,
) -> Result<Box<dyn EnhanceModel>, UnipaseError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnModel::<Metal<f32>>::load(
            dir,
            WgpuDevice::default(),
        )?)),
        Precision::F16 => Err(half_precision_unavailable()),
    }
}

/// Reports the one precision this runtime does not serve.
fn half_precision_unavailable() -> UnipaseError {
    UnipaseError::InvalidOptions(
        "the burn runtime computes in f32 only; use `--precision f32`, or \
         `--runtime candle` for f16"
            .into(),
    )
}
