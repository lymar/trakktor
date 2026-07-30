//! The GigaAM runtime seam and its candle implementation.
//!
//! [`AsrModel`] is the narrow interface the rest of the engine sees: a loaded
//! model turns one chunk's log-mel features into emitted tokens (ids with
//! their encoder frames). [`GigaamModel`] implements it on
//! [candle](candle_core), loading a checkpoint (`.ckpt`, a PyTorch/Lightning
//! archive) and running the Conformer encoder with a CTC or RNN-T head.
//! Devices: CPU always; Metal behind the `gigaam-metal` cargo feature. An
//! alternative burn-backed implementation lives in `runtime_burn` (the
//! `gigaam-burn` cargo feature).
//!
//! Decoding stays off the device either way: the chunk ends in a single
//! synchronization (the CTC path reads back per-frame argmax labels, the
//! RNN-T path reads back the encoder output for the CPU-side transducer loop
//! in [`rnnt`](super::rnnt)), and the token emission logic is shared across
//! runtimes. The seam splits there: [`AsrModel::encode_chunk`] is the device
//! half ending in that read-back, and the [`EncodedChunk`] it returns decodes
//! on the CPU with no device or model in sight — so a caller may decode one
//! chunk on another thread while the device already encodes the next.

#[cfg(test)]
mod bench;
pub mod net;
#[cfg(test)]
mod tests;

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use net::{ConformerEncoder, CtcHead};

use super::{
    config::{ModelClass, ModelConfig},
    decode,
    error::GigaamError,
    feature::{FeatureExtractor, Mel},
    rnnt::RnntHead,
};

/// Maps any backend failure onto the engine's model error.
pub(crate) fn model_err(
    context: &str,
    e: impl std::fmt::Display,
) -> GigaamError {
    GigaamError::InvalidModel(format!("{context}: {e}"))
}

/// The decoded token emissions of one chunk: token ids with the encoder frame
/// each was emitted at, and the chunk's encoder frame count (the scale that
/// turns frames into seconds).
#[derive(Debug, Clone)]
pub struct Emissions {
    pub token_ids: Vec<u32>,
    pub token_frames: Vec<usize>,
    pub enc_frames: usize,
}

/// A loaded model behind a runtime-agnostic seam: everything transcription
/// needs is the feature extractor matching the model's mel geometry and the
/// map from one chunk's log-mel features to emitted tokens (greedy CTC
/// collapse or the greedy transducer loop, by the model's head). The rest of
/// the pipeline (speech segmentation, tokenization, timestamps) is shared
/// across runtimes.
pub trait AsrModel {
    /// The feature extractor.
    fn feature(&self) -> &FeatureExtractor;

    /// The device half of one chunk: the encoder forward (plus, for a CTC
    /// head, its projection and argmax), ending in the chunk's single
    /// read-back. Everything still to do is pure CPU, packaged as an
    /// [`EncodedChunk`].
    fn encode_chunk(&self, mel: &Mel) -> Result<EncodedChunk, GigaamError>;

    /// Both halves in sequence: [`encode_chunk`](Self::encode_chunk), then
    /// [`EncodedChunk::decode`].
    fn emissions(&self, mel: &Mel) -> Result<Emissions, GigaamError> {
        Ok(self.encode_chunk(mel)?.decode())
    }
}

/// The read-back device output of one chunk, one token-emission step short of
/// [`Emissions`].
///
/// The value is detached from the device and the model — plain buffers plus a
/// shared handle to the transducer head — so [`decode`](Self::decode) can run
/// on another thread while the device starts on the next chunk. Decoding is
/// where the RNN-T head spends its sequential CPU loop; for CTC it is only
/// the collapse of per-frame labels.
pub struct EncodedChunk {
    inner: EncodedInner,
}

enum EncodedInner {
    /// Per-frame argmax labels awaiting the CTC collapse.
    Ctc { labels: Vec<u32>, blank_id: u32 },
    /// The encoder output `[T', d_model]` awaiting the greedy transducer
    /// loop of the shared head.
    Rnnt {
        head: Arc<RnntHead>,
        encoded: Vec<f32>,
        enc_frames: usize,
    },
}

impl EncodedChunk {
    /// Read-back CTC labels awaiting the collapse.
    pub(crate) fn ctc(labels: Vec<u32>, blank_id: u32) -> Self {
        Self {
            inner: EncodedInner::Ctc { labels, blank_id },
        }
    }

    /// Read-back encoder output awaiting the transducer loop.
    pub(crate) fn rnnt(
        head: Arc<RnntHead>,
        encoded: Vec<f32>,
        enc_frames: usize,
    ) -> Self {
        Self {
            inner: EncodedInner::Rnnt {
                head,
                encoded,
                enc_frames,
            },
        }
    }

    /// The pure-CPU half: token emission by the model's head.
    #[must_use]
    pub fn decode(self) -> Emissions {
        match self.inner {
            EncodedInner::Ctc { labels, blank_id } => {
                let (token_ids, token_frames) =
                    decode::ctc_greedy(&labels, labels.len(), blank_id);
                Emissions {
                    token_ids,
                    token_frames,
                    enc_frames: labels.len(),
                }
            },
            EncodedInner::Rnnt {
                head,
                encoded,
                enc_frames,
            } => {
                let (token_ids, token_frames) =
                    head.greedy(&encoded, enc_frames);
                Emissions {
                    token_ids,
                    token_frames,
                    enc_frames,
                }
            },
        }
    }
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
pub(crate) const FB_KEY: &str = "preprocessor.featurizer.0.mel_scale.fb";
pub(crate) const WINDOW_KEY: &str =
    "preprocessor.featurizer.0.spectrogram.window";

/// A model's decoding head: the CTC projection stays on the device, the
/// RNN-T head always runs on the CPU (see [`rnnt`](super::rnnt)).
enum Head {
    Ctc(CtcHead),
    Rnnt(Arc<RnntHead>),
}

/// A loaded GigaAM model: feature extractor, Conformer encoder, and decoding
/// head, the device-resident parts on one device at one precision.
pub struct GigaamModel {
    device: Device,
    dtype: DType,
    feature: FeatureExtractor,
    encoder: ConformerEncoder,
    head: Head,
    n_mels: usize,
    blank_id: u32,
}

impl GigaamModel {
    /// Loads a checkpoint (`.ckpt`) with the given model config.
    pub fn load(
        ckpt: &std::path::Path,
        config: &ModelConfig,
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
        let feature = FeatureExtractor::new(config.mel, &fb, &window);

        // The RNN-T head reads its (CPU, f32) weights straight off the map,
        // before the map moves into the device VarBuilder.
        let head = match config.model_class {
            ModelClass::Ctc => None,
            ModelClass::Rnnt => {
                let rnnt_cfg = config.rnnt.as_ref().ok_or_else(|| {
                    model_err("config", "rnnt model without rnnt geometry")
                })?;
                Some(Head::Rnnt(Arc::new(RnntHead::load(
                    &map,
                    config.encoder.d_model,
                    config.num_classes,
                    rnnt_cfg,
                )?)))
            },
        };

        let vb = VarBuilder::from_tensors(map, dtype, &device);
        let encoder = ConformerEncoder::load(&config.encoder, vb.pp("encoder"))
            .map_err(|e| model_err("loading encoder weights", e))?;
        let head = match head {
            Some(head) => head,
            None => Head::Ctc(
                CtcHead::load(
                    config.encoder.d_model,
                    config.num_classes,
                    vb.pp("head"),
                )
                .map_err(|e| model_err("loading head weights", e))?,
            ),
        };

        Ok(Self {
            device,
            dtype,
            feature,
            encoder,
            head,
            n_mels: config.mel.n_mels,
            blank_id: config.blank_id,
        })
    }

    /// [`load`](Self::load) on the CPU.
    pub fn load_cpu(
        ckpt: &std::path::Path,
        config: &ModelConfig,
        precision: Precision,
    ) -> Result<Self, GigaamError> {
        Self::load(ckpt, config, Device::Cpu, precision)
    }

    /// [`load`](Self::load) on the first Metal device.
    #[cfg(feature = "gigaam-metal")]
    pub fn load_metal(
        ckpt: &std::path::Path,
        config: &ModelConfig,
        precision: Precision,
    ) -> Result<Self, GigaamError> {
        let device = Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))?;
        Self::load(ckpt, config, device, precision)
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
    /// Errors on an RNN-T model.
    pub fn ctc_logits(&self, encoded: &Tensor) -> Result<Tensor, GigaamError> {
        let Head::Ctc(ctc_head) = &self.head else {
            return Err(model_err("ctc head", "not a CTC model"));
        };
        let batched = encoded
            .unsqueeze(0)
            .map_err(|e| model_err("ctc input", e))?;
        let logits = ctc_head
            .logits(&batched)
            .map_err(|e| model_err("ctc head", e))?;
        logits.i(0).map_err(|e| model_err("ctc logits", e))
    }

    /// The encoded chunk read back as row-major `[T', d_model]` `f32`.
    fn encoded_to_f32(
        &self,
        encoded: &Tensor,
    ) -> Result<(Vec<f32>, usize), GigaamError> {
        let enc_frames =
            encoded.dim(0).map_err(|e| model_err("encoder output", e))?;
        let flat = encoded
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err("encoder read-back", e))?;
        Ok((flat, enc_frames))
    }
}

impl AsrModel for GigaamModel {
    fn feature(&self) -> &FeatureExtractor { &self.feature }

    fn encode_chunk(&self, mel: &Mel) -> Result<EncodedChunk, GigaamError> {
        let encoded = self.encode(mel)?;
        let chunk = match &self.head {
            Head::Ctc(_) => {
                let logits = self.ctc_logits(&encoded)?;
                let labels = logits
                    .argmax(candle_core::D::Minus1)
                    .and_then(|t| t.to_vec1::<u32>())
                    .map_err(|e| model_err("argmax", e))?;
                EncodedChunk::ctc(labels, self.blank_id)
            },
            Head::Rnnt(rnnt_head) => {
                let (encoded, enc_frames) = self.encoded_to_f32(&encoded)?;
                EncodedChunk::rnnt(Arc::clone(rnnt_head), encoded, enc_frames)
            },
        };
        Ok(chunk)
    }
}

/// Reads a checkpoint tensor as a flat f32 vector.
fn tensor_to_f32_vec(
    map: &HashMap<String, Tensor>,
    key: &str,
) -> Result<Vec<f32>, GigaamError> {
    Ok(tensor_to_f32_parts(map, key)?.0)
}

/// Reads a checkpoint tensor as a flat f32 vector plus its shape — the
/// runtime-neutral form other backends build their tensors from.
pub(crate) fn tensor_to_f32_parts(
    map: &HashMap<String, Tensor>,
    key: &str,
) -> Result<(Vec<f32>, Vec<usize>), GigaamError> {
    let tensor = map
        .get(key)
        .ok_or_else(|| model_err("checkpoint", format!("missing `{key}`")))?;
    let values = tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| model_err(key, e))?;
    Ok((values, tensor.dims().to_vec()))
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
