//! The candle-backed SaT runtime.
//!
//! Loads a checkpoint (a directory with `config.json` and `model.safetensors`
//! in the published `sat-*-sm` layout) at a selectable [`Precision`] and runs
//! the batched window forward over the vendored network in [`net`]. Devices:
//! CPU always; Metal and CUDA behind the corresponding cargo features.

pub(super) mod net;

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
pub use net::NEWLINE_INDEX;

use super::{error::StructifyError, segment};

/// Maps any backend failure onto the feature's model error.
pub(super) fn model_err(
    context: &str,
    e: impl std::fmt::Display,
) -> StructifyError {
    StructifyError::InvalidModel(format!("{context}: {e}"))
}

/// A loaded SaT network, ready to score token boundaries — the seam behind
/// which the runtimes (candle, burn) are interchangeable. The windowing,
/// batching, and overlap stitching around the forward are shared
/// ([`segment::windowed_logits`]); implementations supply only the batched
/// window forward.
pub trait BoundaryModel {
    /// Runs the model over all windows of the token stream and returns one
    /// averaged boundary logit per token (`ids.len()` values).
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError`] on a backend failure.
    fn token_boundary_logits(
        &self,
        ids: &[u32],
        cls_id: u32,
        sep_id: u32,
        stride: usize,
        batch_size: usize,
        weighting: segment::Weighting,
    ) -> Result<Vec<f32>, StructifyError>;
}

/// Compute precision of the runtime.
///
/// The published checkpoints ship in full precision. [`F16`](Precision::F16) —
/// half the memory and faster matmuls — is the default; [`F32`](Precision::F32)
/// keeps full precision for reference parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Half precision (`f16`).
    #[default]
    F16,
    /// Full precision (`f32`).
    F32,
}

impl Precision {
    fn dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::F32 => DType::F32,
        }
    }
}

/// The SaT network running on candle.
pub struct SatRuntime {
    device: Device,
    model: net::SatModel,
}

impl SatRuntime {
    /// Loads a checkpoint directory (`config.json` + `model.safetensors`) onto
    /// `device`, converting the weights to `precision`.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError::InvalidModel`] when the files are missing or
    /// malformed, or the weights do not match the declared geometry.
    pub fn load(
        model_dir: &Path,
        device: Device,
        precision: Precision,
    ) -> Result<Self, StructifyError> {
        let config_path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|e| model_err(&config_path.display().to_string(), e))?;
        let config = parse_config(&raw)?;

        let dtype = precision.dtype();
        // The `-sm` models ship safetensors (memory-mapped); the base models
        // ship a PyTorch checkpoint, read through candle's pickle loader.
        let safetensors = model_dir.join("model.safetensors");
        let pth = model_dir.join("pytorch_model.bin");
        let vb = if safetensors.is_file() {
            // Safety: the checkpoint file is mapped read-only and must not be
            // modified while the runtime is alive.
            unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[&safetensors],
                    dtype,
                    &device,
                )
            }
            .map_err(|e| model_err(&safetensors.display().to_string(), e))?
        } else if pth.is_file() {
            VarBuilder::from_pth(&pth, dtype, &device)
                .map_err(|e| model_err(&pth.display().to_string(), e))?
        } else {
            return Err(StructifyError::InvalidModel(format!(
                "no model.safetensors or pytorch_model.bin in {}",
                model_dir.display()
            )));
        };

        let model = net::SatModel::load(vb, &config)
            .map_err(|e| model_err("loading model weights", e))?;
        Ok(Self { device, model })
    }

    /// The device this runtime computes on.
    pub fn device(&self) -> &Device { &self.device }

    /// [`load`](Self::load) on the CPU.
    ///
    /// # Errors
    ///
    /// See [`load`](Self::load).
    pub fn load_cpu(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StructifyError> {
        Self::load(model_dir, Device::Cpu, precision)
    }

    /// [`load`](Self::load) on the first Metal device.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError::InvalidModel`] when no Metal device is
    /// available; otherwise see [`load`](Self::load).
    #[cfg(feature = "structify-metal")]
    pub fn load_metal(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StructifyError> {
        let device = Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))?;
        Self::load(model_dir, device, precision)
    }

    /// [`load`](Self::load) on the first CUDA device.
    ///
    /// # Errors
    ///
    /// Returns [`StructifyError::InvalidModel`] when no CUDA device is
    /// available; otherwise see [`load`](Self::load).
    #[cfg(feature = "structify-cuda")]
    pub fn load_cuda(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StructifyError> {
        let device = Device::new_cuda(0)
            .map_err(|e| model_err("creating the cuda device", e))?;
        Self::load(model_dir, device, precision)
    }

    /// Forwards one batch of full windows and returns each window's logits
    /// with the `CLS`/`SEP` positions dropped.
    fn window_batch_logits(
        &self,
        buffer: &[u32],
        n_batch: usize,
        seq_len: usize,
    ) -> Result<Vec<Vec<f32>>, StructifyError> {
        let input =
            Tensor::from_vec(buffer.to_vec(), (n_batch, seq_len), &self.device)
                .map_err(|e| model_err("building input tensor", e))?;
        let mask = Tensor::ones((n_batch, seq_len), DType::F32, &self.device)
            .map_err(|e| model_err("building attention mask", e))?;
        let logits = self
            .model
            .forward(&input, &mask)
            .map_err(|e| model_err("model forward", e))?;
        // Drop CLS (col 0) and SEP (last col): keep the real tokens.
        logits
            .narrow(1, 1, seq_len - 2)
            .and_then(|t| t.to_vec2::<f32>())
            .map_err(|e| model_err("reading logits", e))
    }
}

impl BoundaryModel for SatRuntime {
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
            |buffer, n_batch, seq_len| {
                self.window_batch_logits(buffer, n_batch, seq_len)
            },
        )
    }
}

/// Reads the model geometry from the checkpoint's `config.json`.
pub(super) fn parse_config(raw: &str) -> Result<net::Config, StructifyError> {
    let value: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| model_err("config.json", e))?;
    let field = |name: &str| -> Result<usize, StructifyError> {
        value[name].as_u64().map(|v| v as usize).ok_or_else(|| {
            StructifyError::InvalidModel(format!(
                "config.json: missing `{name}`"
            ))
        })
    };
    // Limited-lookahead base models (e.g. `sat-12l`, `lookahead != null`) need
    // a banded attention mask this port does not build yet; reject them clearly
    // rather than return silently wrong boundaries. The full-context base model
    // (`sat-12l-no-limited-lookahead`, `lookahead: null`) and the `-sm` models
    // are supported.
    if value["lookahead"].is_number() {
        return Err(StructifyError::InvalidModel(format!(
            "model uses limited lookahead (lookahead={}), not yet supported; \
             use sat-12l-no-limited-lookahead or an -sm model",
            value["lookahead"]
        )));
    }
    // num_labels is not an explicit field; HF derives it from `id2label`.
    let num_labels = value["id2label"]
        .as_object()
        .map_or(1, |labels| labels.len().max(1));
    Ok(net::Config {
        hidden_size: field("hidden_size")?,
        num_attention_heads: field("num_attention_heads")?,
        intermediate_size: field("intermediate_size")?,
        num_hidden_layers: field("num_hidden_layers")?,
        vocab_size: field("vocab_size")?,
        max_position_embeddings: field("max_position_embeddings")?,
        type_vocab_size: field("type_vocab_size")?,
        pad_token_id: value["pad_token_id"].as_u64().unwrap_or(1) as u32,
        layer_norm_eps: value["layer_norm_eps"].as_f64().unwrap_or(1e-5),
        num_labels,
    })
}
