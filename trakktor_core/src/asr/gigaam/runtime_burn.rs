//! The burn-backed GigaAM runtime.
//!
//! An alternative [`CtcModel`] implementation on [burn](burn), selectable at
//! run time next to the candle one. Backends: ndarray on the CPU (**f32
//! only** — the ndarray backend has no half-precision element), and wgpu with
//! MSL-compiled kernels on Metal (f16 and f32), with operator fusion and
//! autotuning.
//!
//! Checkpoints are read by the same pickle reader as the candle runtime
//! ([`load_state_dict`]); every weight is converted through f32 into a tensor
//! of the target backend, shape-checked against the checkpoint on the way.

pub mod net;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{Tensor, TensorData, backend::Backend, f16},
};
use net::{ConformerEncoder, CtcHead};

use super::{
    config::EncoderConfig,
    error::GigaamError,
    feature::{FeatureExtractor, Mel, MelConfig},
    runtime::{
        CtcModel, FB_KEY, Precision, WINDOW_KEY, load_state_dict, model_err,
        tensor_to_f32_parts,
    },
};

/// A loaded GigaAM CTC model on a burn backend: feature extractor, Conformer
/// encoder, and CTC head, all on one device at the backend's compute dtype.
pub struct GigaamBurnModel<B: Backend> {
    device: B::Device,
    feature: FeatureExtractor,
    encoder: ConformerEncoder<B>,
    ctc_head: CtcHead<B>,
    n_mels: usize,
}

impl<B: Backend> GigaamBurnModel<B> {
    /// Loads a CTC checkpoint (`.ckpt`) with the given geometry onto `device`.
    pub fn load_ctc(
        ckpt: &std::path::Path,
        encoder_cfg: EncoderConfig,
        mel_cfg: MelConfig,
        num_classes: usize,
        device: B::Device,
    ) -> Result<Self, GigaamError> {
        let tensors = load_state_dict(ckpt, &candle_core::Device::Cpu)?;
        let map: HashMap<String, candle_core::Tensor> =
            tensors.into_iter().collect();

        // The mel filterbank and STFT window are checkpoint buffers; the
        // (f32 CPU) feature extractor uses them verbatim.
        let (fb, _) = tensor_to_f32_parts(&map, FB_KEY)?;
        let (window, _) = tensor_to_f32_parts(&map, WINDOW_KEY)?;
        let feature = FeatureExtractor::new(mel_cfg, &fb, &window);

        let encoder = ConformerEncoder::load(&encoder_cfg, &map, &device)?;
        let ctc_head =
            CtcHead::load(encoder_cfg.d_model, num_classes, &map, &device)?;
        Ok(Self {
            device,
            feature,
            encoder,
            ctc_head,
            n_mels: mel_cfg.n_mels,
        })
    }

    /// One chunk's log-mel features as the encoder's `[1, n_mels, T]` input.
    fn mel_input(&self, mel: &Mel) -> Tensor<B, 3> {
        Tensor::from_data(
            TensorData::new(
                mel.data().to_vec(),
                [1, self.n_mels, mel.n_frames()],
            ),
            &self.device,
        )
    }

    /// Runs the encoder over a single chunk's log-mel features, returning the
    /// encoded output `[T', d_model]`.
    pub fn encode(&self, mel: &Mel) -> Tensor<B, 2> {
        let out = self.encoder.forward(self.mel_input(mel)); // [1, T', D]
        let [_, t, d] = out.dims();
        out.reshape([t, d])
    }

    /// CTC logits `[T', num_classes]` for an encoded chunk `[T', d_model]`.
    pub fn ctc_logits(&self, encoded: Tensor<B, 2>) -> Tensor<B, 2> {
        let [t, d] = encoded.dims();
        let logits = self.ctc_head.logits(encoded.reshape([1, t, d]));
        let [_, _, c] = logits.dims();
        logits.reshape([t, c])
    }
}

impl<B: Backend> CtcModel for GigaamBurnModel<B> {
    fn feature(&self) -> &FeatureExtractor { &self.feature }

    fn ctc_labels(&self, mel: &Mel) -> Result<Vec<u32>, GigaamError> {
        let encoded = self.encoder.forward(self.mel_input(mel));
        let logits = self.ctc_head.logits(encoded); // [1, T', C]
        let labels = logits.argmax(2);
        labels
            .into_data()
            .convert::<u32>()
            .to_vec::<u32>()
            .map_err(|e| model_err("argmax readback", format!("{e:?}")))
    }
}

/// Loads a CTC checkpoint on the burn CPU backend (ndarray).
///
/// # Errors
///
/// [`GigaamError::InvalidOptions`] for [`Precision::F16`] — the ndarray
/// backend computes in f32 only.
pub fn load_ctc_cpu(
    ckpt: &std::path::Path,
    encoder_cfg: EncoderConfig,
    mel_cfg: MelConfig,
    num_classes: usize,
    precision: Precision,
) -> Result<Box<dyn CtcModel>, GigaamError> {
    match precision {
        Precision::F32 => {
            Ok(Box::new(GigaamBurnModel::<NdArray<f32>>::load_ctc(
                ckpt,
                encoder_cfg,
                mel_cfg,
                num_classes,
                NdArrayDevice::Cpu,
            )?))
        },
        Precision::F16 => Err(GigaamError::InvalidOptions(
            "the burn runtime computes in f32 on the CPU; use f32 precision \
             (or the metal device)"
                .into(),
        )),
    }
}

/// Loads a CTC checkpoint on the burn Metal backend (wgpu with MSL-compiled
/// kernels), at the requested precision.
pub fn load_ctc_metal(
    ckpt: &std::path::Path,
    encoder_cfg: EncoderConfig,
    mel_cfg: MelConfig,
    num_classes: usize,
    precision: Precision,
) -> Result<Box<dyn CtcModel>, GigaamError> {
    let device = WgpuDevice::default();
    match precision {
        Precision::F16 => {
            Ok(Box::new(GigaamBurnModel::<Metal<f16>>::load_ctc(
                ckpt,
                encoder_cfg,
                mel_cfg,
                num_classes,
                device,
            )?))
        },
        Precision::F32 => {
            Ok(Box::new(GigaamBurnModel::<Metal<f32>>::load_ctc(
                ckpt,
                encoder_cfg,
                mel_cfg,
                num_classes,
                device,
            )?))
        },
    }
}
