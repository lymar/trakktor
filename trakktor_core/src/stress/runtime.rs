//! The candle-backed stress runtime.
//!
//! Loads the converted weights (`model.safetensors`) at a selectable
//! [`Precision`] and runs both networks: the n-gram accentor, which scores a
//! batch of words, and the homograph encoder, which picks a reading for a batch
//! of marked sentences.
//!
//! Devices: CPU always; Metal and CUDA behind the corresponding cargo features.

mod net;

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::{
    error::StressError,
    model::{Config, HomographContext, StressModel, WEIGHTS_FILE, WordScores},
};

/// Maps any backend failure onto the feature's model error.
pub(super) fn model_err(
    context: &str,
    e: impl std::fmt::Display,
) -> StressError {
    StressError::InvalidModel(format!("{context}: {e}"))
}

/// Compute precision of the runtime.
///
/// Unlike the other text models, the default here is **f32**: both networks are
/// small enough that it costs nothing, and every decision they feed is a
/// threshold (`softmax > 0.5`, `sigmoid > 0`) that half precision can flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Full precision — the default.
    #[default]
    F32,
    /// Half precision: less memory, faster on a GPU.
    F16,
}

impl Precision {
    fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
        }
    }

    /// The value reported in the output contract.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

/// The two networks running on candle.
pub struct StressRuntime {
    device: Device,
    model: net::StressNet,
}

impl StressRuntime {
    /// Loads a converted model directory onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`StressError::InvalidModel`] when the weights are missing,
    /// malformed, or do not match the declared geometry.
    pub fn load(
        model_dir: &Path,
        device: Device,
        precision: Precision,
    ) -> Result<Self, StressError> {
        let cfg = Config::silero_ru();
        let weights = model_dir.join(WEIGHTS_FILE);
        if !weights.is_file() {
            return Err(StressError::InvalidModel(format!(
                "no {WEIGHTS_FILE} in {}",
                model_dir.display()
            )));
        }
        let model =
            net::StressNet::load(&weights, &cfg, &device, precision.dtype())?;
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
    ) -> Result<Self, StressError> {
        Self::load(model_dir, Device::Cpu, precision)
    }

    /// [`load`](Self::load) on the first Metal device.
    ///
    /// # Errors
    ///
    /// Returns [`StressError::InvalidModel`] when no Metal device is
    /// available; otherwise see [`load`](Self::load).
    #[cfg(feature = "stress-metal")]
    pub fn load_metal(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StressError> {
        let device = Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))?;
        Self::load(model_dir, device, precision)
    }

    /// [`load`](Self::load) on the first CUDA device.
    ///
    /// # Errors
    ///
    /// Returns [`StressError::InvalidModel`] when no CUDA device is
    /// available; otherwise see [`load`](Self::load).
    #[cfg(feature = "stress-cuda")]
    pub fn load_cuda(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, StressError> {
        let device = Device::new_cuda(0)
            .map_err(|e| model_err("creating the cuda device", e))?;
        Self::load(model_dir, device, precision)
    }
}

impl StressModel for StressRuntime {
    fn accentor(
        &self,
        bags: &[Vec<u32>],
    ) -> Result<Vec<WordScores>, StressError> {
        if bags.is_empty() {
            return Ok(Vec::new());
        }
        let (indices, counts, width) = pad_bags(bags);
        let batch = bags.len();
        let indices = Tensor::from_vec(indices, (batch, width), &self.device)
            .map_err(|e| model_err("building the n-gram batch", e))?;
        let counts = Tensor::from_vec(counts, (batch, width, 1), &self.device)
            .map_err(|e| model_err("building the n-gram mask", e))?;
        self.model.accentor(&indices, &counts)
    }

    fn homographs(
        &self,
        contexts: &[HomographContext],
        pad: u32,
    ) -> Result<Vec<usize>, StressError> {
        if contexts.is_empty() {
            return Ok(Vec::new());
        }
        let (ids, width) = pad_contexts(contexts, pad);
        let batch = contexts.len();
        let ids = Tensor::from_vec(ids, (batch, width), &self.device)
            .map_err(|e| model_err("building the context batch", e))?;
        self.model.homographs(&ids, contexts)
    }
}

/// Pads a batch of n-gram bags to a rectangle: the row indices, a `1.0`/`0.0`
/// mask over them, and the common width.
///
/// The mask is what makes the padding free — a padded slot contributes a zero
/// row to the sum and nothing to the count.
pub(super) fn pad_bags(bags: &[Vec<u32>]) -> (Vec<u32>, Vec<f32>, usize) {
    let width = bags.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut indices = Vec::with_capacity(bags.len() * width);
    let mut mask = Vec::with_capacity(bags.len() * width);
    for bag in bags {
        for slot in 0..width {
            match bag.get(slot) {
                Some(row) => {
                    indices.push(*row);
                    mask.push(1.0);
                },
                None => {
                    indices.push(0);
                    mask.push(0.0);
                },
            }
        }
    }
    (indices, mask, width)
}

/// Pads a batch of homograph contexts to the longest one.
///
/// There is deliberately **no attention mask**: the reference pads the same way
/// and lets the encoder attend to the padding, so masking it away would change
/// the answers it gives.
pub(super) fn pad_contexts(
    contexts: &[HomographContext],
    pad: u32,
) -> (Vec<u32>, usize) {
    let width = contexts
        .iter()
        .map(|context| context.ids.len())
        .max()
        .unwrap_or(1);
    let mut ids = Vec::with_capacity(contexts.len() * width);
    for context in contexts {
        ids.extend_from_slice(&context.ids);
        ids.extend(std::iter::repeat_n(pad, width - context.ids.len()));
    }
    (ids, width)
}

#[cfg(test)]
pub(super) mod tests;
