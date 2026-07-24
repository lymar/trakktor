//! The candle-backed runtime.
//!
//! Holds the networks of the engine and the plumbing around them: device
//! selection, checkpoint loading, and the precision each stage runs at. The
//! codec decoder always runs in full precision — it is the stage whose output
//! must reproduce exactly — while the talker and the code predictor follow the
//! requested [`Precision`].

mod codec;
mod layers;
mod talker;

use std::path::Path;

use candle_core::DType;
pub use candle_core::Device;
use candle_nn::VarBuilder;
pub use codec::CodecDecoder;
pub use talker::Talker;

use super::{Precision, config::CodecConfig, error::Qwen3TtsError};

/// Maps any backend failure onto the engine's model error.
pub(super) fn model_err(
    context: &str,
    e: impl std::fmt::Display,
) -> Qwen3TtsError {
    Qwen3TtsError::InvalidModel(format!("{context}: {e}"))
}

impl Precision {
    /// The tensor type this precision loads weights as.
    pub(super) fn dtype(self) -> DType {
        match self {
            Precision::Bf16 => DType::BF16,
            Precision::F32 => DType::F32,
        }
    }
}

/// The file holding the codec's weights inside a checkpoint directory.
const CODEC_WEIGHTS: &str = "speech_tokenizer/model.safetensors";
/// The file holding the codec's geometry inside a checkpoint directory.
const CODEC_CONFIG: &str = "speech_tokenizer/config.json";

/// Loads the codec decoder from a checkpoint directory.
///
/// # Errors
///
/// Returns [`Qwen3TtsError::InvalidModel`] when the codec files are missing or
/// malformed, or when the weights do not match the declared geometry.
pub fn load_codec(
    model_dir: &Path,
    device: Device,
) -> Result<CodecDecoder, Qwen3TtsError> {
    let config_path = model_dir.join(CODEC_CONFIG);
    let weights_path = model_dir.join(CODEC_WEIGHTS);
    for path in [&config_path, &weights_path] {
        if !path.is_file() {
            return Err(Qwen3TtsError::InvalidModel(format!(
                "no {} in the checkpoint",
                path.display()
            )));
        }
    }

    let config = std::fs::read_to_string(&config_path)
        .map_err(|e| model_err(&config_path.display().to_string(), e))?;
    let cfg = CodecConfig::parse(&config)?;
    if cfg.total_upsample() != cfg.decode_upsample_rate {
        return Err(Qwen3TtsError::InvalidModel(format!(
            "codec upsampling stages multiply to {} but the config declares {}",
            cfg.total_upsample(),
            cfg.decode_upsample_rate
        )));
    }

    // SAFETY: the checkpoint is memory-mapped read-only; candle requires the
    // file not to be mutated while mapped, which nothing here does.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[&weights_path],
            DType::F32,
            &device,
        )
        .map_err(|e| model_err(&weights_path.display().to_string(), e))?
    };
    CodecDecoder::load(&cfg, vb, device)
        .map_err(|e| model_err("loading the codec decoder", e))
}

/// Creates the compute device for a run.
///
/// # Errors
///
/// Returns [`Qwen3TtsError::InvalidOptions`] when the requested device is not
/// available in this build.
pub fn device(metal: bool) -> Result<Device, Qwen3TtsError> {
    if !metal {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "tts-metal")]
    {
        Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))
    }
    #[cfg(not(feature = "tts-metal"))]
    {
        Err(Qwen3TtsError::InvalidOptions(
            "this build has no Metal support; install or build trakktor with \
             the `metal` feature"
                .into(),
        ))
    }
}
