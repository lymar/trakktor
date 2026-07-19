//! The burn-backed Whisper runtime.
//!
//! An alternative [`ForwardProvider`] implementation on [burn](burn),
//! selectable at run time next to the candle one. Backends: ndarray on the
//! CPU (**f32 only** — the ndarray backend has no half-precision element),
//! and wgpu with MSL-compiled kernels on Metal (f16 and f32), with operator
//! fusion and autotuning.
//!
//! Checkpoints are the same published directories the candle runtime loads
//! (`config.json` + `model.safetensors`); tensors are read through candle's
//! safetensors reader and converted through f32 into tensors of the target
//! backend. The backend choice is erased behind [`BurnRuntime`], so the
//! transcription pipeline monomorphizes over burn once.

pub mod net;
#[cfg(test)]
mod tests;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{DType, Tensor, TensorData, backend::Backend, f16},
};

use super::{
    constants::N_FRAMES,
    error::WhisperError,
    feature::MelWindow,
    model::{CrossQk, ForwardProvider, Logits, ModelDims},
    runtime::{Precision, model_err, parse_config},
    tokenizer::TokenId,
};

/// Lazy access to the checkpoint's tensors: every weight is read on demand
/// and handed over as f32 values plus its shape.
pub(super) struct Weights {
    inner: candle_core::safetensors::MmapedSafetensors,
}

impl Weights {
    fn open(path: &Path) -> Result<Self, WhisperError> {
        // Safety: the checkpoint file is mapped read-only and must not be
        // modified while the weights are being read.
        let inner =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
                .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self { inner })
    }

    /// The named tensor as f32 values and its shape.
    pub(super) fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), WhisperError> {
        let tensor = self
            .inner
            .load(key, &candle_core::Device::Cpu)
            .map_err(|e| model_err(key, e))?;
        let dims = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err(key, e))?;
        Ok((values, dims))
    }
}

/// One decoding session: the per-layer K/V caches and how many positions are
/// already cached.
struct Session<B: Backend> {
    n_batch: usize,
    offset: usize,
    caches: Vec<net::LayerKv<B>>,
}

/// The Whisper network running on one burn backend.
pub struct BurnModel<B: Backend> {
    dims: ModelDims,
    device: B::Device,
    encoder: net::AudioEncoder<B>,
    decoder: net::TextDecoder<B>,
    session: Option<Session<B>>,
}

impl<B: Backend> BurnModel<B> {
    /// Loads a checkpoint directory (`config.json` + `model.safetensors`)
    /// onto `device`.
    pub fn load(
        model_dir: &Path,
        device: B::Device,
    ) -> Result<Self, WhisperError> {
        let config_path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|e| model_err(&config_path.display().to_string(), e))?;
        let dims = parse_config(&raw)?;

        let weights = Weights::open(&model_dir.join("model.safetensors"))?;
        let encoder = net::AudioEncoder::load(&weights, &dims, &device)?;
        let decoder = net::TextDecoder::load(&weights, &dims, &device)?;
        Ok(Self {
            dims,
            device,
            encoder,
            decoder,
            session: None,
        })
    }

    fn encode(
        &mut self,
        mel_window: &MelWindow,
    ) -> Result<Tensor<B, 3>, WhisperError> {
        if mel_window.n_mels() != self.dims.n_mels {
            return Err(WhisperError::InvalidModel(format!(
                "mel window has {} bands, model expects {}",
                mel_window.n_mels(),
                self.dims.n_mels
            )));
        }
        let data = mel_window.data();
        if data.len() != self.dims.n_mels * N_FRAMES {
            return Err(WhisperError::InvalidModel(format!(
                "mel window has {} values, expected {}",
                data.len(),
                self.dims.n_mels * N_FRAMES
            )));
        }
        let mel = Tensor::from_data(
            TensorData::new(data.to_vec(), [1, self.dims.n_mels, N_FRAMES]),
            &self.device,
        );
        Ok(self.encoder.forward(mel))
    }

    fn begin_decode(
        &mut self,
        n_batch: usize,
        features: &Tensor<B, 3>,
    ) -> Result<(), WhisperError> {
        self.session = Some(Session {
            n_batch,
            offset: 0,
            caches: self.decoder.begin_session(features),
        });
        Ok(())
    }

    fn decode_step(
        &mut self,
        step_tokens: &[TokenId],
        n_batch: usize,
    ) -> Result<Logits, WhisperError> {
        let session = self.session.as_mut().ok_or_else(|| {
            WhisperError::InvalidModel("decode_step outside a session".into())
        })?;
        if session.n_batch != n_batch || step_tokens.len() % n_batch != 0 {
            return Err(WhisperError::InvalidModel(format!(
                "step of {} tokens does not fit {} sequences",
                step_tokens.len(),
                n_batch
            )));
        }
        let n_step = step_tokens.len() / n_batch;

        // Diagnostics: attribute the step's wall time across the decoder
        // forward, the vocabulary projection, and the read-back. A device
        // sync between phases is needed so the async backend does not roll
        // one phase's time into the next; only taken when tracing.
        let trace_on = super::trace::on();
        let t0 = trace_on.then(std::time::Instant::now);
        let hidden = self.decoder.forward_session(
            step_tokens,
            n_batch,
            session.offset,
            &mut session.caches,
            &self.device,
        );
        let t1 = t0.map(|_| {
            let _ = B::sync(&self.device);
            std::time::Instant::now()
        });
        let logits = self.decoder.logits(hidden);
        let t2 = t1.map(|_| {
            let _ = B::sync(&self.device);
            std::time::Instant::now()
        });
        session.offset += n_step;

        let data = logits
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| model_err("reading logits", format!("{e:?}")))?;

        if let (Some(t0), Some(t1), Some(t2)) = (t0, t1, t2) {
            let read_end = std::time::Instant::now();
            super::trace::emit(format_args!(
                "        step[n={n_step}]: fwd={:.1} lin={:.1} read={:.1}ms",
                (t1 - t0).as_secs_f64() * 1000.0,
                (t2 - t1).as_secs_f64() * 1000.0,
                (read_end - t2).as_secs_f64() * 1000.0,
            ));
        }
        Ok(Logits::new(n_batch, n_step, self.dims.n_vocab, data))
    }

    fn rearrange_kv_cache(
        &mut self,
        source_indices: &[usize],
    ) -> Result<(), WhisperError> {
        let session = self.session.as_mut().ok_or_else(|| {
            WhisperError::InvalidModel(
                "rearrange_kv_cache outside a session".into(),
            )
        })?;
        // The identity permutation is a no-op, as in the reference.
        if source_indices.iter().enumerate().all(|(i, &s)| i == s) {
            return Ok(());
        }
        let indices: Vec<i64> =
            source_indices.iter().map(|&i| i as i64).collect();
        let indices = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(indices, [source_indices.len()]),
            &self.device,
        );
        for cache in &mut session.caches {
            // The batch-1 cross K/V are shared by every row, so any
            // permutation leaves them as they are; only self K/V move.
            if let Some((k, v)) = cache.self_kv.take() {
                cache.self_kv = Some((
                    k.select(0, indices.clone()),
                    v.select(0, indices.clone()),
                ));
            }
        }
        Ok(())
    }

    fn end_decode(&mut self) { self.session = None; }

    fn forward_with_cross_qk(
        &mut self,
        tokens: &[TokenId],
        features: &Tensor<B, 3>,
    ) -> Result<(Logits, CrossQk), WhisperError> {
        let (hidden, captured) =
            self.decoder.forward_capture(tokens, features, &self.device);
        let logits = self
            .decoder
            .logits(hidden)
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| model_err("reading logits", format!("{e:?}")))?;

        // Per-layer tensors are (1, n_heads, n_tokens, n_frames); stack them
        // into the contract's [layer][head][token][frame] buffer.
        let n_layers = captured.len();
        let [_, n_heads, n_tokens, n_frames] = captured
            .first()
            .ok_or_else(|| {
                WhisperError::InvalidModel("decoder has no layers".into())
            })?
            .dims();
        let mut data =
            Vec::with_capacity(n_layers * n_heads * n_tokens * n_frames);
        for qk in captured {
            data.extend(
                qk.cast(DType::F32).into_data().to_vec::<f32>().map_err(
                    |e| model_err("reading cross-attention", format!("{e:?}")),
                )?,
            );
        }

        Ok((
            Logits::new(1, tokens.len(), self.dims.n_vocab, logits),
            CrossQk::new(n_layers, n_heads, n_tokens, n_frames, data),
        ))
    }
}

/// Audio features of [`BurnRuntime`]: the encoder output on whichever
/// backend the runtime runs.
pub enum BurnAudioFeatures {
    /// Features on the CPU backend.
    Cpu(Tensor<NdArray<f32>, 3>),
    /// Features on the Metal backend in f16.
    MetalF16(Tensor<Metal<f16>, 3>),
    /// Features on the Metal backend in f32.
    MetalF32(Tensor<Metal<f32>, 3>),
}

/// The burn-backed Whisper runtime with the backend chosen at load time.
pub enum BurnRuntime {
    /// ndarray on the CPU, f32.
    Cpu(Box<BurnModel<NdArray<f32>>>),
    /// wgpu/MSL on Metal, f16.
    MetalF16(Box<BurnModel<Metal<f16>>>),
    /// wgpu/MSL on Metal, f32.
    MetalF32(Box<BurnModel<Metal<f32>>>),
}

impl BurnRuntime {
    /// Loads a checkpoint on the burn CPU backend (ndarray).
    ///
    /// # Errors
    ///
    /// [`WhisperError::InvalidOptions`] for [`Precision::F16`] — the ndarray
    /// backend computes in f32 only; otherwise see [`BurnModel::load`].
    pub fn load_cpu(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, WhisperError> {
        match precision {
            Precision::F32 => Ok(Self::Cpu(Box::new(BurnModel::load(
                model_dir,
                NdArrayDevice::Cpu,
            )?))),
            Precision::F16 => Err(WhisperError::InvalidOptions(
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
    /// See [`BurnModel::load`].
    pub fn load_metal(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, WhisperError> {
        let device = WgpuDevice::default();
        match precision {
            Precision::F16 => Ok(Self::MetalF16(Box::new(BurnModel::load(
                model_dir, device,
            )?))),
            Precision::F32 => Ok(Self::MetalF32(Box::new(BurnModel::load(
                model_dir, device,
            )?))),
        }
    }
}

/// The features handed in were produced by a different backend variant —
/// impossible through the transcription pipeline, which never mixes
/// runtimes.
fn foreign_features() -> WhisperError {
    WhisperError::InvalidModel(
        "audio features come from a different backend".into(),
    )
}

impl ForwardProvider for BurnRuntime {
    type AudioFeatures = BurnAudioFeatures;

    fn dims(&self) -> &ModelDims {
        match self {
            Self::Cpu(model) => &model.dims,
            Self::MetalF16(model) => &model.dims,
            Self::MetalF32(model) => &model.dims,
        }
    }

    fn encode(
        &mut self,
        mel_window: &MelWindow,
    ) -> Result<Self::AudioFeatures, WhisperError> {
        match self {
            Self::Cpu(model) => {
                Ok(BurnAudioFeatures::Cpu(model.encode(mel_window)?))
            },
            Self::MetalF16(model) => {
                Ok(BurnAudioFeatures::MetalF16(model.encode(mel_window)?))
            },
            Self::MetalF32(model) => {
                Ok(BurnAudioFeatures::MetalF32(model.encode(mel_window)?))
            },
        }
    }

    fn begin_decode(
        &mut self,
        n_batch: usize,
        features: &Self::AudioFeatures,
    ) -> Result<(), WhisperError> {
        match (self, features) {
            (Self::Cpu(model), BurnAudioFeatures::Cpu(features)) => {
                model.begin_decode(n_batch, features)
            },
            (Self::MetalF16(model), BurnAudioFeatures::MetalF16(features)) => {
                model.begin_decode(n_batch, features)
            },
            (Self::MetalF32(model), BurnAudioFeatures::MetalF32(features)) => {
                model.begin_decode(n_batch, features)
            },
            _ => Err(foreign_features()),
        }
    }

    fn decode_step(
        &mut self,
        step_tokens: &[TokenId],
        n_batch: usize,
    ) -> Result<Logits, WhisperError> {
        match self {
            Self::Cpu(model) => model.decode_step(step_tokens, n_batch),
            Self::MetalF16(model) => model.decode_step(step_tokens, n_batch),
            Self::MetalF32(model) => model.decode_step(step_tokens, n_batch),
        }
    }

    fn rearrange_kv_cache(
        &mut self,
        source_indices: &[usize],
    ) -> Result<(), WhisperError> {
        match self {
            Self::Cpu(model) => model.rearrange_kv_cache(source_indices),
            Self::MetalF16(model) => model.rearrange_kv_cache(source_indices),
            Self::MetalF32(model) => model.rearrange_kv_cache(source_indices),
        }
    }

    fn end_decode(&mut self) {
        match self {
            Self::Cpu(model) => model.end_decode(),
            Self::MetalF16(model) => model.end_decode(),
            Self::MetalF32(model) => model.end_decode(),
        }
    }

    fn forward_with_cross_qk(
        &mut self,
        tokens: &[TokenId],
        features: &Self::AudioFeatures,
    ) -> Result<(Logits, CrossQk), WhisperError> {
        match (self, features) {
            (Self::Cpu(model), BurnAudioFeatures::Cpu(features)) => {
                model.forward_with_cross_qk(tokens, features)
            },
            (Self::MetalF16(model), BurnAudioFeatures::MetalF16(features)) => {
                model.forward_with_cross_qk(tokens, features)
            },
            (Self::MetalF32(model), BurnAudioFeatures::MetalF32(features)) => {
                model.forward_with_cross_qk(tokens, features)
            },
            _ => Err(foreign_features()),
        }
    }
}
