//! The burn-backed stress runtime.
//!
//! An alternative [`StressModel`] implementation on [burn](burn), selectable at
//! run time next to the candle one. Backends: ndarray on the CPU (**f32
//! only**), and wgpu with MSL-compiled kernels on Metal (f16 and f32), with
//! operator fusion and autotuning.
//!
//! The weights are the same converted `model.safetensors` the candle runtime
//! loads: tensors are read through candle's safetensors reader and converted
//! through f32 into tensors of the target backend. The backend choice is erased
//! behind [`StressBurnRuntime`], so the pipeline sees one type.

pub mod net;

use std::{collections::HashMap, path::Path};

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{Int, Tensor, TensorData, backend::Backend, f16},
};

use super::{
    error::StressError,
    model::{Config, HomographContext, StressModel, WEIGHTS_FILE, WordScores},
    runtime::{Precision, model_err, pad_bags, pad_contexts},
};

/// The converted weights, read once on the host.
pub(super) struct Weights(HashMap<String, candle_core::Tensor>);

impl Weights {
    /// Opens `model.safetensors` in `model_dir`.
    fn open(model_dir: &Path) -> Result<Self, StressError> {
        let path = model_dir.join(WEIGHTS_FILE);
        if !path.is_file() {
            return Err(StressError::InvalidModel(format!(
                "no {WEIGHTS_FILE} in {}",
                model_dir.display()
            )));
        }
        let tensors =
            candle_core::safetensors::load(&path, &candle_core::Device::Cpu)
                .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self(tensors))
    }

    fn tensor(&self, key: &str) -> Result<&candle_core::Tensor, StressError> {
        self.0
            .get(key)
            .ok_or_else(|| model_err(key, "not in the weights"))
    }

    /// The named tensor as f32 values and its shape.
    pub(super) fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), StressError> {
        let tensor = self.tensor(key)?;
        let dims = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err(key, e))?;
        Ok((values, dims))
    }

    /// The named tensor's stored bytes — for the quantized embedding table,
    /// which is kept byte-sized on disk.
    pub(super) fn bytes(
        &self,
        key: &str,
    ) -> Result<(Vec<u8>, Vec<usize>), StressError> {
        let tensor = self.tensor(key)?;
        if tensor.dtype() != candle_core::DType::U8 {
            return Err(model_err(
                key,
                format!("stored as {:?}, expected bytes", tensor.dtype()),
            ));
        }
        let dims = tensor.dims().to_vec();
        let values = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<u8>())
            .map_err(|e| model_err(key, e))?;
        Ok((values, dims))
    }

    /// The one scalar values stored as a length-one tensor.
    pub(super) fn scalar(&self, key: &str) -> Result<f32, StressError> {
        let (values, _) = self.parts(key)?;
        values
            .first()
            .copied()
            .ok_or_else(|| model_err(key, "empty scalar"))
    }
}

/// The two networks on one burn backend.
pub struct StressBurnModel<B: Backend> {
    device: B::Device,
    model: net::StressNet<B>,
}

impl<B: Backend> StressBurnModel<B> {
    /// Loads the converted weights onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`StressError::InvalidModel`] when the weights are missing,
    /// malformed, or do not match the declared geometry.
    pub fn load(
        model_dir: &Path,
        device: B::Device,
    ) -> Result<Self, StressError> {
        let cfg = Config::silero_ru();
        let weights = Weights::open(model_dir)?;
        let model = net::StressNet::load(&weights, &cfg, &device)?;
        Ok(Self { device, model })
    }

    fn accentor(
        &self,
        bags: &[Vec<u32>],
    ) -> Result<Vec<WordScores>, StressError> {
        let (indices, mask, width) = pad_bags(bags);
        let batch = bags.len();
        let indices: Vec<i64> = indices.into_iter().map(i64::from).collect();
        let indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(indices, [batch, width]),
            &self.device,
        );
        let mask = Tensor::<B, 2>::from_data(
            TensorData::new(mask, [batch, width]),
            &self.device,
        );
        self.model.accentor(indices, mask)
    }

    fn homographs(
        &self,
        contexts: &[HomographContext],
        pad: u32,
    ) -> Result<Vec<usize>, StressError> {
        let (ids, width) = pad_contexts(contexts, pad);
        let batch = contexts.len();
        let ids: Vec<i64> = ids.into_iter().map(i64::from).collect();
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids, [batch, width]),
            &self.device,
        );
        self.model.homographs(ids, contexts)
    }
}

/// The burn-backed runtime with the backend chosen at load time.
pub enum StressBurnRuntime {
    /// ndarray on the CPU, f32.
    Cpu(Box<StressBurnModel<NdArray<f32>>>),
    /// wgpu/MSL on Metal, f16.
    MetalF16(Box<StressBurnModel<Metal<f16>>>),
    /// wgpu/MSL on Metal, f32.
    MetalF32(Box<StressBurnModel<Metal<f32>>>),
}

impl StressBurnRuntime {
    /// Loads the model on the burn CPU backend (ndarray).
    ///
    /// # Errors
    ///
    /// [`StressError::InvalidOptions`] for [`Precision::F16`] — the ndarray
    /// backend computes in f32 only; otherwise see [`StressBurnModel::load`].
    pub fn load_cpu(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StressError> {
        match precision {
            Precision::F32 => Ok(Self::Cpu(Box::new(StressBurnModel::load(
                model_dir,
                NdArrayDevice::Cpu,
            )?))),
            Precision::F16 => Err(StressError::InvalidOptions(
                "the burn runtime computes in f32 on the CPU; use f32 \
                 precision (or the metal device)"
                    .into(),
            )),
        }
    }

    /// Loads the model on the burn Metal backend at the requested precision.
    ///
    /// # Errors
    ///
    /// See [`StressBurnModel::load`].
    pub fn load_metal(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StressError> {
        let device = WgpuDevice::default();
        match precision {
            Precision::F16 => Ok(Self::MetalF16(Box::new(
                StressBurnModel::load(model_dir, device)?,
            ))),
            Precision::F32 => Ok(Self::MetalF32(Box::new(
                StressBurnModel::load(model_dir, device)?,
            ))),
        }
    }
}

impl StressModel for StressBurnRuntime {
    fn accentor(
        &self,
        bags: &[Vec<u32>],
    ) -> Result<Vec<WordScores>, StressError> {
        if bags.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Self::Cpu(model) => model.accentor(bags),
            Self::MetalF16(model) => model.accentor(bags),
            Self::MetalF32(model) => model.accentor(bags),
        }
    }

    fn homographs(
        &self,
        contexts: &[HomographContext],
        pad: u32,
    ) -> Result<Vec<usize>, StressError> {
        if contexts.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Self::Cpu(model) => model.homographs(contexts, pad),
            Self::MetalF16(model) => model.homographs(contexts, pad),
            Self::MetalF32(model) => model.homographs(contexts, pad),
        }
    }
}

#[cfg(test)]
mod tests;
