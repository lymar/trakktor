//! The burn-backed Qwen3-TTS runtime.
//!
//! An alternative [`SpeechModel`] implementation on [burn](burn), selectable at
//! run time next to the candle one. Backends: ndarray on the CPU, and wgpu with
//! MSL-compiled kernels on Metal, with operator fusion and autotuning.
//!
//! **This runtime computes in f32 only**, on either device. The ndarray backend
//! has no half-precision element at all, and on Metal the half-precision this
//! engine needs is unavailable from below: `bf16` kernels — down to a plain
//! unary elementwise one — are rejected by the shader compiler, and `f16` is
//! excluded by the model itself (see [`Precision`], whose narrow exponent sends
//! generation off the rails). `--precision bf16 --runtime burn` is therefore a
//! validation error rather than a silent downgrade; full precision on Metal is
//! what this runtime is good at, and it is the mode the two runtimes are
//! compared in.
//!
//! The checkpoint is the same published directory the candle runtime loads:
//! tensors are read through candle's safetensors reader and converted through
//! f32 into tensors of the target backend.
//!
//! The backend choice is erased behind a boxed [`SpeechModel`], so the
//! synthesis driver sees one type.

pub mod codec;
pub mod net;
#[cfg(test)]
mod tests;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::backend::Backend,
};

use super::{
    Precision,
    config::{CODEC_WEIGHTS, CodecConfig, ModelConfig, TALKER_WEIGHTS},
    error::Qwen3TtsError,
    model::SpeechModel,
    prompt::Position,
    runtime::model_err,
};

/// Lazy access to a checkpoint's tensors through candle's safetensors reader.
pub(super) struct Weights(candle_core::safetensors::MmapedSafetensors);

impl Weights {
    /// Opens one safetensors file.
    fn open(path: &Path) -> Result<Self, Qwen3TtsError> {
        if !path.is_file() {
            return Err(Qwen3TtsError::InvalidModel(format!(
                "no {} in the checkpoint",
                path.display()
            )));
        }
        // SAFETY: the checkpoint is memory-mapped read-only; candle requires
        // the file not to be mutated while mapped, which nothing here does.
        let inner =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
                .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self(inner))
    }

    /// The named tensor as f32 values and its shape.
    pub(super) fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), Qwen3TtsError> {
        let tensor = self
            .0
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

/// Blocked out-of-place transpose of a row-major `[rows, cols]` matrix.
pub(super) fn transpose_2d(
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    const TILE: usize = 64;
    let mut out = vec![0.0f32; values.len()];
    for row0 in (0..rows).step_by(TILE) {
        for col0 in (0..cols).step_by(TILE) {
            for row in row0..(row0 + TILE).min(rows) {
                for col in col0..(col0 + TILE).min(cols) {
                    out[col * rows + row] = values[row * cols + col];
                }
            }
        }
    }
    out
}

/// The engine's networks on one burn backend.
pub struct BurnSpeech<B: Backend> {
    config: ModelConfig,
    talker: net::Talker<B>,
    codec: codec::CodecDecoder<B>,
}

impl<B: Backend> BurnSpeech<B> {
    /// Loads a checkpoint directory onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the checkpoint is missing
    /// files, malformed, or does not match its declared geometry.
    pub fn load(
        model_dir: &Path,
        device: B::Device,
    ) -> Result<Self, Qwen3TtsError> {
        let config = ModelConfig::read(model_dir)?;
        let talker = net::Talker::load(
            &Weights::open(&model_dir.join(TALKER_WEIGHTS))?,
            &device,
            &config.talker,
            config.tts_pad_token_id,
        )?;
        let codec = codec::CodecDecoder::load(
            &Weights::open(&model_dir.join(CODEC_WEIGHTS))?,
            &device,
            &CodecConfig::read(model_dir)?,
        )?;
        Ok(Self {
            config,
            talker,
            codec,
        })
    }
}

impl<B: Backend> SpeechModel for BurnSpeech<B> {
    fn config(&self) -> &ModelConfig { &self.config }

    fn sample_rate(&self) -> u32 { self.codec.sample_rate() }

    fn prime(
        &mut self,
        positions: &[Position],
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        self.talker.prime(positions)
    }

    fn predict_residuals(
        &mut self,
        first_code: u32,
        pick: &mut dyn FnMut(&[f32], usize) -> u32,
    ) -> Result<Vec<u32>, Qwen3TtsError> {
        self.talker.predict_residuals(first_code, pick)
    }

    fn advance(&mut self, frame: &[u32]) -> Result<Vec<f32>, Qwen3TtsError> {
        self.talker.advance(frame)
    }

    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>, Qwen3TtsError> {
        self.codec.decode(frames)
    }
}

/// Reports the one precision this runtime does not serve.
fn half_precision_unavailable() -> Qwen3TtsError {
    Qwen3TtsError::InvalidOptions(
        "the burn runtime computes in f32 only; use `--precision f32`, or \
         `--runtime candle` for bf16"
            .into(),
    )
}

/// Loads a checkpoint on the burn CPU backend (ndarray).
///
/// # Errors
///
/// Returns [`Qwen3TtsError::InvalidOptions`] for [`Precision::Bf16`];
/// otherwise see [`BurnSpeech::load`].
pub fn load_cpu(
    model_dir: &Path,
    precision: Precision,
) -> Result<Box<dyn SpeechModel>, Qwen3TtsError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnSpeech::<NdArray<f32>>::load(
            model_dir,
            NdArrayDevice::Cpu,
        )?)),
        Precision::Bf16 => Err(half_precision_unavailable()),
    }
}

/// Loads a checkpoint on the burn Metal backend (wgpu with MSL-compiled
/// kernels).
///
/// # Errors
///
/// Returns [`Qwen3TtsError::InvalidOptions`] for [`Precision::Bf16`];
/// otherwise see [`BurnSpeech::load`].
pub fn load_metal(
    model_dir: &Path,
    precision: Precision,
) -> Result<Box<dyn SpeechModel>, Qwen3TtsError> {
    match precision {
        Precision::F32 => Ok(Box::new(BurnSpeech::<Metal<f32>>::load(
            model_dir,
            WgpuDevice::default(),
        )?)),
        Precision::Bf16 => Err(half_precision_unavailable()),
    }
}
