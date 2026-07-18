//! The candle-backed GigaAM runtime.
//!
//! Loads a checkpoint (`.ckpt`, a PyTorch/Lightning archive) and runs the
//! Conformer encoder and CTC / RNN-T heads on [candle](candle_core). Devices:
//! CPU always; Metal behind the `gigaam-metal` cargo feature.

#[cfg(test)]
mod bench;
pub mod net;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use net::{ConformerEncoder, CtcHead};

use super::{
    config::EncoderConfig,
    error::GigaamError,
    feature::{FeatureExtractor, Mel, MelConfig},
};

/// Maps any backend failure onto the engine's model error.
fn model_err(context: &str, e: impl std::fmt::Display) -> GigaamError {
    GigaamError::InvalidModel(format!("{context}: {e}"))
}

/// Compute precision of the runtime.
///
/// The published checkpoints ship their encoder weights in half precision, and
/// the reference runs its GPU path in half precision too.
/// [`F16`](Precision::F16) matches that — half the memory, faster matmuls — and
/// is the default on GPU. [`F32`](Precision::F32) keeps full precision for
/// bit-close reference parity.
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

/// Names of the checkpoint's feature-extractor buffers.
const FB_KEY: &str = "preprocessor.featurizer.0.mel_scale.fb";
const WINDOW_KEY: &str = "preprocessor.featurizer.0.spectrogram.window";

/// A loaded GigaAM CTC model: feature extractor, Conformer encoder, and CTC
/// head, all on one device at one precision.
pub struct GigaamModel {
    device: Device,
    dtype: DType,
    feature: FeatureExtractor,
    encoder: ConformerEncoder,
    ctc_head: CtcHead,
    n_mels: usize,
}

impl GigaamModel {
    /// Loads a CTC checkpoint (`.ckpt`) with the given geometry.
    pub fn load_ctc(
        ckpt: &std::path::Path,
        encoder_cfg: EncoderConfig,
        mel_cfg: MelConfig,
        num_classes: usize,
        device: Device,
        precision: Precision,
    ) -> Result<Self, GigaamError> {
        let dtype = precision.dtype();
        let tensors = load_state_dict(ckpt, &device)?;
        let map: HashMap<String, Tensor> = tensors.into_iter().collect();

        // The mel filterbank and STFT window are checkpoint buffers; extract
        // them as f32 for the (f32 CPU) feature extractor.
        let fb = tensor_to_f32_vec(&map, FB_KEY)?;
        let window = tensor_to_f32_vec(&map, WINDOW_KEY)?;
        let feature = FeatureExtractor::new(mel_cfg, &fb, &window);

        let vb = VarBuilder::from_tensors(map, dtype, &device);
        let encoder = ConformerEncoder::load(&encoder_cfg, vb.pp("encoder"))
            .map_err(|e| model_err("loading encoder weights", e))?;
        let ctc_head =
            CtcHead::load(encoder_cfg.d_model, num_classes, vb.pp("head"))
                .map_err(|e| model_err("loading head weights", e))?;

        Ok(Self {
            device,
            dtype,
            feature,
            encoder,
            ctc_head,
            n_mels: mel_cfg.n_mels,
        })
    }

    /// [`load_ctc`](Self::load_ctc) on the CPU.
    pub fn load_ctc_cpu(
        ckpt: &std::path::Path,
        encoder_cfg: EncoderConfig,
        mel_cfg: MelConfig,
        num_classes: usize,
        precision: Precision,
    ) -> Result<Self, GigaamError> {
        Self::load_ctc(
            ckpt,
            encoder_cfg,
            mel_cfg,
            num_classes,
            Device::Cpu,
            precision,
        )
    }

    /// [`load_ctc`](Self::load_ctc) on the first Metal device.
    #[cfg(feature = "gigaam-metal")]
    pub fn load_ctc_metal(
        ckpt: &std::path::Path,
        encoder_cfg: EncoderConfig,
        mel_cfg: MelConfig,
        num_classes: usize,
        precision: Precision,
    ) -> Result<Self, GigaamError> {
        let device = Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))?;
        Self::load_ctc(
            ckpt,
            encoder_cfg,
            mel_cfg,
            num_classes,
            device,
            precision,
        )
    }

    /// The feature extractor.
    pub fn feature(&self) -> &FeatureExtractor { &self.feature }

    /// The compute device.
    pub fn device(&self) -> &Device { &self.device }

    /// Runs the encoder over a single chunk's log-mel features, returning the
    /// encoded output `[T', d_model]`.
    pub fn encode(&self, mel: &Mel) -> Result<Tensor, GigaamError> {
        let n_frames = mel.n_frames();
        let input = Tensor::from_slice(
            mel.data(),
            (1, self.n_mels, n_frames),
            &self.device,
        )
        .and_then(|t| t.to_dtype(self.dtype))
        .map_err(|e| model_err("building mel tensor", e))?;
        let encoded = self
            .encoder
            .forward(&input)
            .map_err(|e| model_err("encoder forward", e))?;
        encoded.i(0).map_err(|e| model_err("encoder output", e))
    }

    /// CTC logits `[T', num_classes]` for an encoded chunk `[T', d_model]`.
    pub fn ctc_logits(&self, encoded: &Tensor) -> Result<Tensor, GigaamError> {
        let batched = encoded
            .unsqueeze(0)
            .map_err(|e| model_err("ctc input", e))?;
        let logits = self
            .ctc_head
            .logits(&batched)
            .map_err(|e| model_err("ctc head", e))?;
        logits.i(0).map_err(|e| model_err("ctc logits", e))
    }
}

/// Reads a checkpoint tensor as a flat f32 vector.
fn tensor_to_f32_vec(
    map: &HashMap<String, Tensor>,
    key: &str,
) -> Result<Vec<f32>, GigaamError> {
    let tensor = map
        .get(key)
        .ok_or_else(|| model_err("checkpoint", format!("missing `{key}`")))?;
    tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| model_err(key, e))
}

/// Reads the tensors under a checkpoint's `state_dict` into name/tensor pairs
/// on `device`. GigaAM checkpoints are Lightning archives whose weights live
/// under the `state_dict` key alongside a (non-tensor) config; candle's pickle
/// reader extracts just the tensors at that key.
pub(crate) fn load_state_dict(
    path: &std::path::Path,
    device: &Device,
) -> Result<Vec<(String, Tensor)>, GigaamError> {
    let tensors =
        candle_core::pickle::read_all_with_key(path, Some("state_dict"))
            .map_err(|e| model_err("reading checkpoint", e))?;
    let mut out = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let tensor = tensor
            .to_device(device)
            .map_err(|e| model_err("moving weights to device", e))?;
        out.push((name, tensor));
    }
    Ok(out)
}
