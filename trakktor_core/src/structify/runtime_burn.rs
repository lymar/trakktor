//! The burn-backed SaT runtime.
//!
//! An alternative [`BoundaryModel`] implementation on [burn](burn), selectable
//! at run time next to the candle one. Backends: ndarray on the CPU (**f32
//! only** — the ndarray backend has no half-precision element), and wgpu with
//! MSL-compiled kernels on Metal (f16 and f32), with operator fusion and
//! autotuning.
//!
//! Checkpoints are the same published directories the candle runtime loads:
//! tensors are read through candle's safetensors reader (the `-sm` models) or
//! its pickle reader (the base models' `pytorch_model.bin`) and converted
//! through f32 into tensors of the target backend. The backend choice is
//! erased behind [`SatBurnRuntime`], so the segmentation pipeline sees one
//! type.

pub mod net;
#[cfg(test)]
mod tests;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{Int, Tensor, TensorData, backend::Backend, f16},
};

use super::{
    error::StructifyError,
    runtime::{BoundaryModel, Precision, model_err, parse_config},
    segment,
};

/// Lazy access to the checkpoint's tensors — safetensors (`-sm` models) or a
/// PyTorch pickle (base models) behind one f32 reader.
pub(super) enum Weights {
    Safetensors(candle_core::safetensors::MmapedSafetensors),
    Pickle(candle_core::pickle::PthTensors),
}

impl Weights {
    /// Opens the checkpoint in `model_dir`, probing the same weight files as
    /// the candle runtime.
    fn open(model_dir: &Path) -> Result<Self, StructifyError> {
        let safetensors = model_dir.join("model.safetensors");
        let pth = model_dir.join("pytorch_model.bin");
        if safetensors.is_file() {
            // Safety: the checkpoint file is mapped read-only and must not be
            // modified while the weights are being read.
            let inner = unsafe {
                candle_core::safetensors::MmapedSafetensors::new(&safetensors)
            }
            .map_err(|e| model_err(&safetensors.display().to_string(), e))?;
            Ok(Self::Safetensors(inner))
        } else if pth.is_file() {
            let inner = candle_core::pickle::PthTensors::new(&pth, None)
                .map_err(|e| model_err(&pth.display().to_string(), e))?;
            Ok(Self::Pickle(inner))
        } else {
            Err(StructifyError::InvalidModel(format!(
                "no model.safetensors or pytorch_model.bin in {}",
                model_dir.display()
            )))
        }
    }

    /// The named tensor as f32 values and its shape.
    pub(super) fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), StructifyError> {
        let tensor = match self {
            Self::Safetensors(inner) => inner
                .load(key, &candle_core::Device::Cpu)
                .map_err(|e| model_err(key, e))?,
            Self::Pickle(inner) => inner
                .get(key)
                .map_err(|e| model_err(key, e))?
                .ok_or_else(|| model_err(key, "tensor not found"))?,
        };
        let dims = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err(key, e))?;
        Ok((values, dims))
    }
}

/// The SaT network running on one burn backend.
pub struct SatBurnModel<B: Backend> {
    device: B::Device,
    model: net::SatModel<B>,
}

impl<B: Backend> SatBurnModel<B> {
    /// Loads a checkpoint directory (`config.json` + weights) onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError::InvalidModel`] when the files are missing or
    /// malformed, or the weights do not match the declared geometry.
    pub fn load(
        model_dir: &Path,
        device: B::Device,
    ) -> Result<Self, StructifyError> {
        let config_path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|e| model_err(&config_path.display().to_string(), e))?;
        let config = parse_config(&raw)?;
        let weights = Weights::open(model_dir)?;
        let model = net::SatModel::load(&weights, &config, &device)?;
        Ok(Self { device, model })
    }

    /// Forwards one batch of full windows and returns each window's logits
    /// with the `CLS`/`SEP` positions dropped.
    fn window_batch_logits(
        &self,
        buffer: &[u32],
        n_batch: usize,
        seq_len: usize,
    ) -> Result<Vec<Vec<f32>>, StructifyError> {
        let ids: Vec<i64> = buffer.iter().map(|&id| i64::from(id)).collect();
        let input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids, [n_batch, seq_len]),
            &self.device,
        );
        let logits = self.model.forward(input); // (batch, seq) f32
        // Drop CLS (col 0) and SEP (last col): keep the real tokens.
        let block = seq_len - 2;
        let data = logits
            .narrow(1, 1, block)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| model_err("reading logits", format!("{e:?}")))?;
        Ok(data.chunks(block).map(<[f32]>::to_vec).collect())
    }
}

/// The burn-backed SaT runtime with the backend chosen at load time.
pub enum SatBurnRuntime {
    /// ndarray on the CPU, f32.
    Cpu(Box<SatBurnModel<NdArray<f32>>>),
    /// wgpu/MSL on Metal, f16.
    MetalF16(Box<SatBurnModel<Metal<f16>>>),
    /// wgpu/MSL on Metal, f32.
    MetalF32(Box<SatBurnModel<Metal<f32>>>),
}

impl SatBurnRuntime {
    /// Loads a checkpoint on the burn CPU backend (ndarray).
    ///
    /// # Errors
    ///
    /// [`StructifyError::InvalidOptions`] for [`Precision::F16`] — the
    /// ndarray backend computes in f32 only; otherwise see
    /// [`SatBurnModel::load`].
    pub fn load_cpu(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StructifyError> {
        match precision {
            Precision::F32 => Ok(Self::Cpu(Box::new(SatBurnModel::load(
                model_dir,
                NdArrayDevice::Cpu,
            )?))),
            Precision::F16 => Err(StructifyError::InvalidOptions(
                "the burn runtime computes in f32 on the CPU; use f32 \
                 precision (or the metal device)"
                    .into(),
            )),
        }
    }

    /// Loads a checkpoint on the burn Metal backend (wgpu with MSL-compiled
    /// kernels), at the requested precision.
    ///
    /// # Errors
    ///
    /// See [`SatBurnModel::load`].
    pub fn load_metal(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StructifyError> {
        let device = WgpuDevice::default();
        match precision {
            Precision::F16 => Ok(Self::MetalF16(Box::new(SatBurnModel::load(
                model_dir, device,
            )?))),
            Precision::F32 => Ok(Self::MetalF32(Box::new(SatBurnModel::load(
                model_dir, device,
            )?))),
        }
    }
}

impl BoundaryModel for SatBurnRuntime {
    fn token_boundary_logits(
        &self,
        ids: &[u32],
        cls_id: u32,
        sep_id: u32,
        stride: usize,
        batch_size: usize,
        weighting: segment::Weighting,
    ) -> Result<Vec<f32>, StructifyError> {
        segment::windowed_logits(
            ids,
            cls_id,
            sep_id,
            stride,
            batch_size,
            weighting,
            |buffer, n_batch, seq_len| match self {
                Self::Cpu(model) => {
                    model.window_batch_logits(buffer, n_batch, seq_len)
                },
                Self::MetalF16(model) => {
                    model.window_batch_logits(buffer, n_batch, seq_len)
                },
                Self::MetalF32(model) => {
                    model.window_batch_logits(buffer, n_batch, seq_len)
                },
            },
        )
    }
}
