//! The burn-backed punctuation runtime.
//!
//! An alternative [`PunctCapSegModel`] implementation on [burn](burn),
//! selectable at run time next to the candle one. Backends: ndarray on the CPU
//! (**f32 only**), and wgpu with MSL-compiled kernels on Metal (f16 and f32),
//! with operator fusion and autotuning.
//!
//! The checkpoint is the same `model_weights.ckpt` the candle runtime loads:
//! tensors are read through candle's pickle reader and converted through f32
//! into tensors of the target backend. The backend choice is erased behind
//! [`PunctBurnRuntime`], so the pipeline sees one type.

pub mod net;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{Int, Tensor, TensorData, backend::Backend, f16},
};

use super::{
    error::PunctuateError,
    model::Config,
    runtime::{Precision, PunctCapSegModel, WEIGHTS_FILE, assemble, model_err},
    segment::TokenPred,
};

/// Lazy access to the checkpoint's tensors through candle's pickle reader.
pub(super) struct Weights(candle_core::pickle::PthTensors);

impl Weights {
    /// Opens `model_weights.ckpt` in `model_dir`.
    fn open(model_dir: &Path) -> Result<Self, PunctuateError> {
        let ckpt = model_dir.join(WEIGHTS_FILE);
        if !ckpt.is_file() {
            return Err(PunctuateError::InvalidModel(format!(
                "no {WEIGHTS_FILE} in {}",
                model_dir.display()
            )));
        }
        let inner = candle_core::pickle::PthTensors::new(&ckpt, None)
            .map_err(|e| model_err(&ckpt.display().to_string(), e))?;
        Ok(Self(inner))
    }

    /// The named tensor as f32 values and its shape.
    pub(super) fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), PunctuateError> {
        let tensor = self
            .0
            .get(key)
            .map_err(|e| model_err(key, e))?
            .ok_or_else(|| model_err(key, "tensor not found"))?;
        let dims = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err(key, e))?;
        Ok((values, dims))
    }
}

/// The punctuation network running on one burn backend.
pub struct PunctBurnModel<B: Backend> {
    device: B::Device,
    model: net::PunctModel<B>,
}

impl<B: Backend> PunctBurnModel<B> {
    /// Loads the checkpoint onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`PunctuateError::InvalidModel`] when the checkpoint is missing
    /// or malformed, or the weights do not match the declared geometry.
    pub fn load(
        model_dir: &Path,
        device: B::Device,
    ) -> Result<Self, PunctuateError> {
        let cfg = Config::xlmr_47lang();
        let weights = Weights::open(model_dir)?;
        let model = net::PunctModel::load(&weights, &cfg, &device)?;
        Ok(Self { device, model })
    }

    /// Forwards one batch of equal-length windows.
    fn forward_batch(
        &self,
        windows: &[Vec<u32>],
    ) -> Result<Vec<Vec<TokenPred>>, PunctuateError> {
        let Some(first) = windows.first() else {
            return Ok(Vec::new());
        };
        let (batch, seq) = (windows.len(), first.len());
        let ids: Vec<i64> = windows
            .iter()
            .flat_map(|w| w.iter().map(|&id| i64::from(id)))
            .collect();
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids, [batch, seq]),
            &self.device,
        );
        let raw = self.model.forward(input)?;
        Ok(assemble(&raw))
    }
}

/// The burn-backed punctuation runtime with the backend chosen at load time.
pub enum PunctBurnRuntime {
    /// ndarray on the CPU, f32.
    Cpu(Box<PunctBurnModel<NdArray<f32>>>),
    /// wgpu/MSL on Metal, f16.
    MetalF16(Box<PunctBurnModel<Metal<f16>>>),
    /// wgpu/MSL on Metal, f32.
    MetalF32(Box<PunctBurnModel<Metal<f32>>>),
}

impl PunctBurnRuntime {
    /// Loads a checkpoint on the burn CPU backend (ndarray).
    ///
    /// # Errors
    ///
    /// [`PunctuateError::InvalidOptions`] for [`Precision::F16`] — the ndarray
    /// backend computes in f32 only; otherwise see [`PunctBurnModel::load`].
    pub fn load_cpu(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, PunctuateError> {
        match precision {
            Precision::F32 => Ok(Self::Cpu(Box::new(PunctBurnModel::load(
                model_dir,
                NdArrayDevice::Cpu,
            )?))),
            Precision::F16 => Err(PunctuateError::InvalidOptions(
                "the burn runtime computes in f32 on the CPU; use f32 \
                 precision (or the metal device)"
                    .into(),
            )),
        }
    }

    /// Loads a checkpoint on the burn Metal backend at the requested precision.
    ///
    /// # Errors
    ///
    /// See [`PunctBurnModel::load`].
    pub fn load_metal(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, PunctuateError> {
        let device = WgpuDevice::default();
        match precision {
            Precision::F16 => Ok(Self::MetalF16(Box::new(
                PunctBurnModel::load(model_dir, device)?,
            ))),
            Precision::F32 => Ok(Self::MetalF32(Box::new(
                PunctBurnModel::load(model_dir, device)?,
            ))),
        }
    }
}

impl PunctCapSegModel for PunctBurnRuntime {
    fn forward_batch(
        &self,
        windows: &[Vec<u32>],
    ) -> Result<Vec<Vec<TokenPred>>, PunctuateError> {
        match self {
            Self::Cpu(model) => model.forward_batch(windows),
            Self::MetalF16(model) => model.forward_batch(windows),
            Self::MetalF32(model) => model.forward_batch(windows),
        }
    }
}
