//! The burn-backed GigaAM runtime.
//!
//! An alternative [`AsrModel`] implementation on [burn](burn), selectable at
//! run time next to the candle one. Backends: ndarray on the CPU (**f32
//! only** — the ndarray backend has no half-precision element), and wgpu with
//! MSL-compiled kernels on Metal (f16 and f32), with operator fusion and
//! autotuning.
//!
//! Checkpoints are read by the same pickle reader as the candle runtime
//! ([`load_state_dict`]); every weight is converted through f32 into a tensor
//! of the target backend, shape-checked against the checkpoint on the way.
//! The RNN-T head is the exception: it always runs in `f32` on the CPU (see
//! [`rnnt`](super::rnnt)), identical code and weights across runtimes.

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
    config::{ModelClass, ModelConfig},
    decode,
    error::GigaamError,
    feature::{FeatureExtractor, Mel},
    rnnt::RnntHead,
    runtime::{
        AsrModel, Emissions, FB_KEY, Precision, WINDOW_KEY, load_state_dict,
        model_err, tensor_to_f32_parts,
    },
};

/// A model's decoding head: the CTC projection lives on the backend, the
/// RNN-T head always runs on the CPU (see [`rnnt`](super::rnnt)).
enum Head<B: Backend> {
    Ctc(CtcHead<B>),
    Rnnt(RnntHead),
}

/// A loaded GigaAM model on a burn backend: feature extractor, Conformer
/// encoder, and decoding head, the backend-resident parts on one device at
/// the backend's compute dtype.
pub struct GigaamBurnModel<B: Backend> {
    device: B::Device,
    feature: FeatureExtractor,
    encoder: ConformerEncoder<B>,
    head: Head<B>,
    n_mels: usize,
    blank_id: u32,
}

impl<B: Backend> GigaamBurnModel<B> {
    /// Loads a checkpoint (`.ckpt`) with the given model config onto `device`.
    pub fn load(
        ckpt: &std::path::Path,
        config: &ModelConfig,
        device: B::Device,
    ) -> Result<Self, GigaamError> {
        let tensors = load_state_dict(ckpt, &candle_core::Device::Cpu)?;
        let map: HashMap<String, candle_core::Tensor> =
            tensors.into_iter().collect();

        // The mel filterbank and STFT window are checkpoint buffers; the
        // (f32 CPU) feature extractor uses them verbatim.
        let (fb, _) = tensor_to_f32_parts(&map, FB_KEY)?;
        let (window, _) = tensor_to_f32_parts(&map, WINDOW_KEY)?;
        let feature = FeatureExtractor::new(config.mel, &fb, &window);

        let encoder = ConformerEncoder::load(&config.encoder, &map, &device)?;
        let head = match config.model_class {
            ModelClass::Ctc => Head::Ctc(CtcHead::load(
                config.encoder.d_model,
                config.num_classes,
                &map,
                &device,
            )?),
            ModelClass::Rnnt => {
                let rnnt_cfg = config.rnnt.as_ref().ok_or_else(|| {
                    model_err("config", "rnnt model without rnnt geometry")
                })?;
                Head::Rnnt(RnntHead::load(
                    &map,
                    config.encoder.d_model,
                    config.num_classes,
                    rnnt_cfg,
                )?)
            },
        };
        Ok(Self {
            device,
            feature,
            encoder,
            head,
            n_mels: config.mel.n_mels,
            blank_id: config.blank_id,
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
    /// Panics on an RNN-T model (test/bench helper).
    pub fn ctc_logits(&self, encoded: Tensor<B, 2>) -> Tensor<B, 2> {
        let Head::Ctc(ctc_head) = &self.head else {
            panic!("ctc_logits on a non-CTC model");
        };
        let [t, d] = encoded.dims();
        let logits = ctc_head.logits(encoded.reshape([1, t, d]));
        let [_, _, c] = logits.dims();
        logits.reshape([t, c])
    }
}

impl<B: Backend> AsrModel for GigaamBurnModel<B> {
    fn feature(&self) -> &FeatureExtractor { &self.feature }

    fn emissions(&self, mel: &Mel) -> Result<Emissions, GigaamError> {
        let encoded = self.encoder.forward(self.mel_input(mel)); // [1, T', D]
        match &self.head {
            Head::Ctc(ctc_head) => {
                let logits = ctc_head.logits(encoded); // [1, T', C]
                let labels: Vec<u32> = logits
                    .argmax(2)
                    .into_data()
                    .convert::<u32>()
                    .to_vec::<u32>()
                    .map_err(|e| {
                        model_err("argmax readback", format!("{e:?}"))
                    })?;
                let (token_ids, token_frames) =
                    decode::ctc_greedy(&labels, labels.len(), self.blank_id);
                Ok(Emissions {
                    token_ids,
                    token_frames,
                    enc_frames: labels.len(),
                })
            },
            Head::Rnnt(rnnt_head) => {
                let [_, enc_frames, _] = encoded.dims();
                let flat: Vec<f32> = encoded
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .map_err(|e| {
                        model_err("encoder read-back", format!("{e:?}"))
                    })?;
                let (token_ids, token_frames) =
                    rnnt_head.greedy(&flat, enc_frames);
                Ok(Emissions {
                    token_ids,
                    token_frames,
                    enc_frames,
                })
            },
        }
    }
}

/// Loads a checkpoint on the burn CPU backend (ndarray).
///
/// # Errors
///
/// [`GigaamError::InvalidOptions`] for [`Precision::F16`] — the ndarray
/// backend computes in f32 only.
pub fn load_cpu(
    ckpt: &std::path::Path,
    config: &ModelConfig,
    precision: Precision,
) -> Result<Box<dyn AsrModel>, GigaamError> {
    match precision {
        Precision::F32 => Ok(Box::new(GigaamBurnModel::<NdArray<f32>>::load(
            ckpt,
            config,
            NdArrayDevice::Cpu,
        )?)),
        Precision::F16 => Err(GigaamError::InvalidOptions(
            "the burn runtime computes in f32 on the CPU; use f32 precision \
             (or the metal device)"
                .into(),
        )),
    }
}

/// Loads a checkpoint on the burn Metal backend (wgpu with MSL-compiled
/// kernels), at the requested precision.
pub fn load_metal(
    ckpt: &std::path::Path,
    config: &ModelConfig,
    precision: Precision,
) -> Result<Box<dyn AsrModel>, GigaamError> {
    let device = WgpuDevice::default();
    match precision {
        Precision::F16 => Ok(Box::new(GigaamBurnModel::<Metal<f16>>::load(
            ckpt, config, device,
        )?)),
        Precision::F32 => Ok(Box::new(GigaamBurnModel::<Metal<f32>>::load(
            ckpt, config, device,
        )?)),
    }
}
