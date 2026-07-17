//! The candle-backed Whisper runtime.
//!
//! Loads a checkpoint (a directory holding `config.json` and
//! `model.safetensors` in the published layout) at a selectable
//! [`Precision`] and implements [`ForwardProvider`] on top of the vendored
//! network in [`net`]. Devices: CPU always; Metal and CUDA behind the
//! corresponding cargo features.

mod net;
#[cfg(test)]
mod tests;

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::{
    constants::N_FRAMES,
    error::WhisperError,
    feature::MelWindow,
    model::{CrossQk, ForwardProvider, Logits, ModelDims},
    tokenizer::TokenId,
};

/// Maps any backend failure onto the engine's model error.
fn model_err(context: &str, e: impl std::fmt::Display) -> WhisperError {
    WhisperError::InvalidModel(format!("{context}: {e}"))
}

/// One decoding session: the (batch-expanded) audio features and how many
/// positions are already cached.
struct Session {
    n_batch: usize,
    features: Tensor,
    offset: usize,
}

/// Compute precision of the runtime.
///
/// The published checkpoints ship in half precision, and the reference runs
/// its GPU path in half precision too. [`F16`](Precision::F16) matches that —
/// half the memory and roughly twice the matmul throughput — and is the
/// default. [`F32`](Precision::F32) keeps full precision for bit-exact
/// reference parity, at double the weight memory.
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

/// The Whisper network running on candle.
pub struct CandleRuntime {
    dims: ModelDims,
    device: Device,
    dtype: DType,
    encoder: net::AudioEncoder,
    decoder: net::TextDecoder,
    session: Option<Session>,
}

impl CandleRuntime {
    /// Loads a checkpoint directory (`config.json` + `model.safetensors`)
    /// onto `device`, converting the weights to `precision`.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidModel`] when the files are missing or
    /// malformed, or the weights do not match the declared geometry.
    pub fn load(
        model_dir: &Path,
        device: Device,
        precision: Precision,
    ) -> Result<Self, WhisperError> {
        let config_path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|e| model_err(&config_path.display().to_string(), e))?;
        let dims = parse_config(&raw)?;

        let weights = model_dir.join("model.safetensors");
        let dtype = precision.dtype();
        // Safety: the checkpoint file is mapped read-only and must not be
        // modified while the runtime is alive.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights], dtype, &device)
        }
        .map_err(|e| model_err(&weights.display().to_string(), e))?;

        let encoder = net::AudioEncoder::load(vb.pp("model.encoder"), &dims)
            .map_err(|e| model_err("loading encoder weights", e))?;
        let decoder = net::TextDecoder::load(vb.pp("model.decoder"), &dims)
            .map_err(|e| model_err("loading decoder weights", e))?;

        Ok(Self {
            dims,
            device,
            dtype,
            encoder,
            decoder,
            session: None,
        })
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
    ) -> Result<Self, WhisperError> {
        Self::load(model_dir, Device::Cpu, precision)
    }

    /// [`load`](Self::load) on the first Metal device.
    ///
    /// # Errors
    ///
    /// Returns [`WhisperError::InvalidModel`] when no Metal device is
    /// available; otherwise see [`load`](Self::load).
    #[cfg(feature = "whisper-metal")]
    pub fn load_metal(
        model_dir: &Path,
        precision: Precision,
    ) -> Result<Self, WhisperError> {
        let device = Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))?;
        Self::load(model_dir, device, precision)
    }
}

/// Reads the model geometry from the checkpoint's `config.json`.
fn parse_config(raw: &str) -> Result<ModelDims, WhisperError> {
    let value: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| model_err("config.json", e))?;
    let field = |name: &str| -> Result<usize, WhisperError> {
        value[name].as_u64().map(|v| v as usize).ok_or_else(|| {
            WhisperError::InvalidModel(format!("config.json: missing `{name}`"))
        })
    };
    Ok(ModelDims {
        n_mels: field("num_mel_bins")?,
        n_audio_ctx: field("max_source_positions")?,
        n_audio_state: field("d_model")?,
        n_audio_head: field("encoder_attention_heads")?,
        n_audio_layer: field("encoder_layers")?,
        n_vocab: field("vocab_size")?,
        n_text_ctx: field("max_target_positions")?,
        // The published checkpoints use one width for both stacks.
        n_text_state: field("d_model")?,
        n_text_head: field("decoder_attention_heads")?,
        n_text_layer: field("decoder_layers")?,
    })
}

impl ForwardProvider for CandleRuntime {
    type AudioFeatures = Tensor;

    fn dims(&self) -> &ModelDims { &self.dims }

    fn encode(
        &mut self,
        mel_window: &MelWindow,
    ) -> Result<Self::AudioFeatures, WhisperError> {
        if mel_window.n_mels() != self.dims.n_mels {
            return Err(WhisperError::InvalidModel(format!(
                "mel window has {} bands, model expects {}",
                mel_window.n_mels(),
                self.dims.n_mels
            )));
        }
        let mel = Tensor::from_slice(
            mel_window.data(),
            (1, self.dims.n_mels, N_FRAMES),
            &self.device,
        )
        .and_then(|t| t.to_dtype(self.dtype))
        .map_err(|e| model_err("building mel tensor", e))?;
        self.encoder
            .forward(&mel)
            .map_err(|e| model_err("encoder forward", e))
    }

    fn begin_decode(
        &mut self,
        n_batch: usize,
        features: &Self::AudioFeatures,
    ) -> Result<(), WhisperError> {
        self.decoder.reset_cache();
        let features = if n_batch == 1 {
            features.clone()
        } else {
            // The reference replicates the features across the group; the
            // backend materializes the same replication.
            let (_, n_ctx, n_state) = features
                .dims3()
                .map_err(|e| model_err("audio features shape", e))?;
            features
                .broadcast_as((n_batch, n_ctx, n_state))
                .and_then(|t| t.contiguous())
                .map_err(|e| model_err("replicating audio features", e))?
        };
        self.session = Some(Session {
            n_batch,
            features,
            offset: 0,
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

        let tokens =
            Tensor::from_slice(step_tokens, (n_batch, n_step), &self.device)
                .map_err(|e| model_err("building token tensor", e))?;
        let hidden = self
            .decoder
            .forward_session(&tokens, &session.features, session.offset)
            .map_err(|e| model_err("decoder forward", e))?;
        let logits = self
            .decoder
            .final_linear(&hidden)
            .map_err(|e| model_err("logits projection", e))?;
        session.offset += n_step;

        let data = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err("reading logits", e))?;
        Ok(Logits::new(n_batch, n_step, self.dims.n_vocab, data))
    }

    fn rearrange_kv_cache(
        &mut self,
        source_indices: &[usize],
    ) -> Result<(), WhisperError> {
        if self.session.is_none() {
            return Err(WhisperError::InvalidModel(
                "rearrange_kv_cache outside a session".into(),
            ));
        }
        // The identity permutation is a no-op, as in the reference.
        if source_indices.iter().enumerate().all(|(i, &s)| i == s) {
            return Ok(());
        }
        let indices: Vec<u32> =
            source_indices.iter().map(|&i| i as u32).collect();
        let indices = Tensor::from_slice(&indices, indices.len(), &self.device)
            .map_err(|e| model_err("building index tensor", e))?;
        self.decoder
            .rearrange_cache(&indices)
            .map_err(|e| model_err("rearranging kv cache", e))
    }

    fn end_decode(&mut self) {
        self.session = None;
        self.decoder.reset_cache();
    }

    fn forward_with_cross_qk(
        &mut self,
        tokens: &[TokenId],
        features: &Self::AudioFeatures,
    ) -> Result<(Logits, CrossQk), WhisperError> {
        let tokens_t =
            Tensor::from_slice(tokens, (1, tokens.len()), &self.device)
                .map_err(|e| model_err("building token tensor", e))?;
        let (hidden, cross_qks) = self
            .decoder
            .forward_stateless(&tokens_t, features)
            .map_err(|e| model_err("decoder forward", e))?;
        let logits = self
            .decoder
            .final_linear(&hidden)
            .map_err(|e| model_err("logits projection", e))?;
        let logits_data = logits
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err("reading logits", e))?;

        // Per-layer tensors are (1, n_heads, n_tokens, n_frames); stack them
        // into the contract's [layer][head][token][frame] buffer.
        let n_layers = cross_qks.len();
        let (_, n_heads, n_tokens, n_frames) = cross_qks
            .first()
            .ok_or_else(|| {
                WhisperError::InvalidModel("decoder has no layers".into())
            })?
            .dims4()
            .map_err(|e| model_err("cross-attention shape", e))?;
        let mut data =
            Vec::with_capacity(n_layers * n_heads * n_tokens * n_frames);
        for qk in &cross_qks {
            data.extend(
                qk.to_dtype(DType::F32)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| model_err("reading cross-attention", e))?,
            );
        }

        Ok((
            Logits::new(1, tokens.len(), self.dims.n_vocab, logits_data),
            CrossQk::new(n_layers, n_heads, n_tokens, n_frames, data),
        ))
    }
}
