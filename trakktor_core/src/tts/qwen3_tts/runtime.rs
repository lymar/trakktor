//! The candle-backed runtime.
//!
//! Holds the networks of the engine and the plumbing around them: device
//! selection, checkpoint loading, and the precision each stage runs at. The
//! codec decoder always runs in full precision — it is the stage whose output
//! must reproduce exactly — while the talker and the code predictor follow the
//! requested [`Precision`].

mod codec;
pub(super) mod layers;
mod talker;

use std::path::Path;

use candle_core::DType;
pub use candle_core::Device;
use candle_nn::VarBuilder;
pub use codec::CodecDecoder;
pub use talker::Talker;

use super::{
    Precision,
    config::{CODEC_WEIGHTS, CodecConfig, ModelConfig, TALKER_WEIGHTS},
    error::Qwen3TtsError,
    model::SpeechModel,
    prompt::Position,
};

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

/// The engine's networks on candle.
pub struct CandleSpeech {
    config: ModelConfig,
    talker: Talker,
    codec: CodecDecoder,
    /// The text track's padding, embedded once: every generated frame reads it
    /// and it never changes.
    pad_embed: candle_core::Tensor,
    /// The talker's state for the frame the code predictor is filling.
    state: Option<candle_core::Tensor>,
}

/// Loads a checkpoint directory onto `device`.
///
/// The talker and the code predictor run at `precision`; the codec decoder
/// always runs in full precision.
///
/// # Errors
///
/// Returns [`Qwen3TtsError::InvalidModel`] when the checkpoint is missing
/// files, malformed, or does not match its declared geometry.
pub fn load(
    model_dir: &Path,
    device: Device,
    precision: Precision,
) -> Result<CandleSpeech, Qwen3TtsError> {
    let config = ModelConfig::read(model_dir)?;
    let talker = load_talker(model_dir, &config, &device, precision)?;
    let codec = load_codec(model_dir, device)?;
    let pad_embed = talker
        .embed_text(&[config.tts_pad_token_id])
        .map_err(|e| model_err("embedding the text padding", e))?;
    Ok(CandleSpeech {
        config,
        talker,
        codec,
        pad_embed,
        state: None,
    })
}

impl SpeechModel for CandleSpeech {
    fn config(&self) -> &ModelConfig { &self.config }

    fn sample_rate(&self) -> u32 { self.codec.sample_rate() }

    fn samples_per_frame(&self) -> usize { self.codec.samples_per_frame() }

    fn prime(
        &mut self,
        positions: &[Position],
    ) -> Result<Vec<f32>, Qwen3TtsError> {
        self.talker.reset();
        let embeds = self
            .embed_prompt(positions)
            .map_err(|e| model_err("embedding the prompt", e))?;
        let (logits, state) = self
            .talker
            .forward(&embeds)
            .map_err(|e| model_err("priming the talker", e))?;
        self.state = Some(state);
        logits
            .to_vec1::<f32>()
            .map_err(|e| model_err("reading the logits", e))
    }

    fn predict_residuals(
        &mut self,
        first_code: u32,
        pick: &mut dyn FnMut(&[f32], usize) -> u32,
    ) -> Result<Vec<u32>, Qwen3TtsError> {
        let state = self.state.as_ref().ok_or_else(|| {
            model_err("predicting the residual codebooks", "no primed state")
        })?;
        self.talker
            .predict_residuals(state, first_code, |logits, step| {
                Ok(pick(&logits.to_vec1::<f32>()?, step))
            })
            .map_err(|e| model_err("predicting the residual codebooks", e))
    }

    fn advance(&mut self, frame: &[u32]) -> Result<Vec<f32>, Qwen3TtsError> {
        // The finished frame folds into one embedding and, with the text
        // track's padding, becomes the talker's next input.
        let next = self
            .talker
            .fold_frame(frame)
            .and_then(|folded| folded + &self.pad_embed)
            .map_err(|e| model_err("folding the frame", e))?;
        let (logits, state) = self
            .talker
            .forward(&next)
            .map_err(|e| model_err("advancing the talker", e))?;
        self.state = Some(state);
        logits
            .to_vec1::<f32>()
            .map_err(|e| model_err("reading the logits", e))
    }

    fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>, Qwen3TtsError> {
        self.codec
            .decode(frames)
            .map_err(|e| model_err("decoding the codec frames", e))
    }
}

impl CandleSpeech {
    /// Embeds a laid-out prompt into `[1, positions, hidden]`.
    fn embed_prompt(
        &self,
        positions: &[Position],
    ) -> candle_core::Result<candle_core::Tensor> {
        // Every position drives the text track; only the opening role, a
        // prefix, leaves the codec track silent.
        let text_ids: Vec<u32> = positions
            .iter()
            .filter_map(|position| position.text)
            .collect();
        let lead = positions
            .iter()
            .take_while(|position| position.codec.is_none())
            .count();
        let codec_ids: Vec<u32> = positions
            .iter()
            .filter_map(|position| position.codec)
            .collect();
        debug_assert_eq!(text_ids.len(), positions.len());
        debug_assert_eq!(codec_ids.len() + lead, positions.len());

        let text = self.talker.embed_text(&text_ids)?;
        let codec = self.talker.embed_codec(&codec_ids)?;
        let hidden = text.dim(2)?;
        let silent = candle_core::Tensor::zeros(
            (1, lead, hidden),
            codec.dtype(),
            codec.device(),
        )?;
        text + candle_core::Tensor::cat(&[&silent, &codec], 1)?
    }
}

/// Loads the talker from a checkpoint directory.
fn load_talker(
    model_dir: &Path,
    config: &ModelConfig,
    device: &Device,
    precision: Precision,
) -> Result<Talker, Qwen3TtsError> {
    let weights = model_dir.join(TALKER_WEIGHTS);
    if !weights.is_file() {
        return Err(Qwen3TtsError::InvalidModel(format!(
            "no {} in the checkpoint",
            weights.display()
        )));
    }
    // SAFETY: the checkpoint is memory-mapped read-only; candle requires the
    // file not to be mutated while mapped, which nothing here does.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[&weights],
            precision.dtype(),
            device,
        )
        .map_err(|e| model_err(&weights.display().to_string(), e))?
    };
    Talker::load(&config.talker, vb, device.clone(), precision.dtype())
        .map_err(|e| model_err("loading the talker", e))
}

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
    let weights_path = model_dir.join(CODEC_WEIGHTS);
    if !weights_path.is_file() {
        return Err(Qwen3TtsError::InvalidModel(format!(
            "no {} in the checkpoint",
            weights_path.display()
        )));
    }
    let cfg = CodecConfig::read(model_dir)?;

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
