//! The candle-backed punctuation runtime.
//!
//! Loads the NeMo checkpoint (`model_weights.ckpt`, a PyTorch state dict read
//! through candle's pickle loader) at a selectable [`Precision`] and runs the
//! batched window forward over the vendored network in [`net`]. Devices: CPU
//! always; Metal and CUDA behind the corresponding cargo features.
//!
//! This module also defines the seam shared by both runtimes: the
//! [`PunctCapSegModel`] trait (a batch of equal-length windows → per-position
//! predictions), the raw host-side head outputs ([`RawOutputs`]), and the
//! thresholding that turns them into [`TokenPred`]s ([`assemble`]).

mod net;

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::{
    error::PunctuateError,
    model::{Config, MAX_SUBWORD_LEN, SEG_THRESHOLD},
    segment::TokenPred,
};

/// Maps any backend failure onto the feature's model error.
pub(super) fn model_err(
    context: &str,
    e: impl std::fmt::Display,
) -> PunctuateError {
    PunctuateError::InvalidModel(format!("{context}: {e}"))
}

/// The raw per-window head outputs, pulled to the host by a runtime's forward:
/// argmax pre/post ids, the full-stop probability, and the per-character
/// upper-case probabilities. Indexed `[window][position]` (`cap` adds
/// `[character]`). Shared by the candle and burn nets.
pub(crate) struct RawOutputs {
    pub pre: Vec<Vec<u32>>,
    pub post: Vec<Vec<u32>>,
    pub seg_prob1: Vec<Vec<f32>>,
    pub cap_prob: Vec<Vec<Vec<f32>>>,
}

/// Thresholds the raw head outputs into per-window [`TokenPred`]s: full stop at
/// `softmax(seg)[FULLSTOP] > SEG_THRESHOLD`, upper case at `sigmoid(cap) >
/// 0.5`. Both runtimes end here, so the decision rule lives once.
pub(crate) fn assemble(raw: &RawOutputs) -> Vec<Vec<TokenPred>> {
    raw.pre
        .iter()
        .enumerate()
        .map(|(w, pre)| {
            pre.iter()
                .enumerate()
                .map(|(t, &pre_id)| {
                    let mut cap = [false; MAX_SUBWORD_LEN];
                    // `zip` bounds by both the head width and the array.
                    for (slot, &p) in cap.iter_mut().zip(&raw.cap_prob[w][t]) {
                        *slot = p > 0.5;
                    }
                    TokenPred {
                        pre: pre_id as u8,
                        post: raw.post[w][t] as u8,
                        sbd: raw.seg_prob1[w][t] > SEG_THRESHOLD,
                        cap,
                    }
                })
                .collect()
        })
        .collect()
}

/// A loaded punctuation network, ready to score windows — the seam behind which
/// the runtimes (candle, burn) are interchangeable. The windowing, batching,
/// and overlap stitching around the forward are shared
/// ([`super::segment::windowed_predictions`]); implementations supply only the
/// batched window forward.
pub trait PunctCapSegModel {
    /// Forwards a batch of equal-length windows (each `[bos] + content +
    /// [eos]`) and returns, per window, the prediction for every position
    /// (BOS/EOS included).
    ///
    /// # Errors
    ///
    /// Returns [`PunctuateError`] on a backend failure.
    fn forward_batch(
        &self,
        windows: &[Vec<u32>],
    ) -> Result<Vec<Vec<TokenPred>>, PunctuateError>;
}

/// Compute precision of the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Half precision (`f16`) — the default: half the memory, faster on GPU.
    #[default]
    F16,
    /// Full precision (`f32`) — reproducible, for reference parity.
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

/// The name of the weight file inside a model directory.
pub(super) const WEIGHTS_FILE: &str = "model_weights.ckpt";

/// The punctuation network running on candle.
pub struct PunctRuntime {
    device: Device,
    model: net::PunctModel,
}

impl PunctRuntime {
    /// Loads a model directory (`model_weights.ckpt`) onto `device`, converting
    /// the weights to `precision`.
    ///
    /// # Errors
    ///
    /// Returns [`PunctuateError::InvalidModel`] when the checkpoint is missing
    /// or malformed, or the weights do not match the declared geometry.
    pub fn load(
        model_dir: &Path,
        device: Device,
        precision: Precision,
    ) -> Result<Self, PunctuateError> {
        let cfg = Config::xlmr_47lang();
        let ckpt = model_dir.join(WEIGHTS_FILE);
        if !ckpt.is_file() {
            return Err(PunctuateError::InvalidModel(format!(
                "no {WEIGHTS_FILE} in {}",
                model_dir.display()
            )));
        }
        let vb = VarBuilder::from_pth(&ckpt, precision.dtype(), &device)
            .map_err(|e| model_err(&ckpt.display().to_string(), e))?;
        let model = net::PunctModel::load(vb, &cfg)
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
    ) -> Result<Self, PunctuateError> {
        Self::load(model_dir, Device::Cpu, precision)
    }

    /// [`load`](Self::load) on the first Metal device.
    ///
    /// # Errors
    ///
    /// Returns [`PunctuateError::InvalidModel`] when no Metal device is
    /// available; otherwise see [`load`](Self::load).
    #[cfg(feature = "punctuate-metal")]
    pub fn load_metal(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, PunctuateError> {
        let device = Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))?;
        Self::load(model_dir, device, precision)
    }

    /// [`load`](Self::load) on the first CUDA device.
    ///
    /// # Errors
    ///
    /// Returns [`PunctuateError::InvalidModel`] when no CUDA device is
    /// available; otherwise see [`load`](Self::load).
    #[cfg(feature = "punctuate-cuda")]
    pub fn load_cuda(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, PunctuateError> {
        let device = Device::new_cuda(0)
            .map_err(|e| model_err("creating the cuda device", e))?;
        Self::load(model_dir, device, precision)
    }
}

impl PunctCapSegModel for PunctRuntime {
    fn forward_batch(
        &self,
        windows: &[Vec<u32>],
    ) -> Result<Vec<Vec<TokenPred>>, PunctuateError> {
        let Some(first) = windows.first() else {
            return Ok(Vec::new());
        };
        let (batch, seq) = (windows.len(), first.len());
        let flat: Vec<u32> =
            windows.iter().flat_map(|w| w.iter().copied()).collect();
        let input = Tensor::from_vec(flat, (batch, seq), &self.device)
            .map_err(|e| model_err("building input tensor", e))?;
        let raw = self
            .model
            .forward(&input)
            .map_err(|e| model_err("model forward", e))?;
        Ok(assemble(&raw))
    }
}
