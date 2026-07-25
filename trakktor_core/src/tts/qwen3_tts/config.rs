//! Model configuration, read from the checkpoint's `config.json`.
//!
//! Unlike the engines whose geometry is pinned to a handful of known
//! checkpoints, Qwen3-TTS ships a full `config.json` next to the weights, so
//! the geometry, the special-token ids, and the speaker/language tables are all
//! read from the checkpoint rather than embedded here. That keeps one code path
//! serving every published variant.

use std::{collections::BTreeMap, path::Path};

use serde_json::Value;

use super::error::Qwen3TtsError;

/// Reads a file from a checkpoint directory.
pub(super) fn read(
    model_dir: &Path,
    name: &str,
) -> Result<String, Qwen3TtsError> {
    let path = model_dir.join(name);
    std::fs::read_to_string(&path).map_err(|e| {
        Qwen3TtsError::InvalidModel(format!("reading {}: {e}", path.display()))
    })
}

/// Which conditioning path a checkpoint was trained for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Named preset speakers.
    CustomVoice,
    /// A voice described in natural language.
    VoiceDesign,
    /// Voice cloning from reference audio.
    Base,
}

impl ModelType {
    /// Parses the `tts_model_type` field.
    fn parse(value: &str) -> Result<Self, Qwen3TtsError> {
        match value {
            "custom_voice" => Ok(ModelType::CustomVoice),
            "voice_design" => Ok(ModelType::VoiceDesign),
            "base" => Ok(ModelType::Base),
            other => Err(Qwen3TtsError::InvalidModel(format!(
                "unknown tts_model_type `{other}`"
            ))),
        }
    }
}

/// Geometry of the residual code predictor (the "MTP" head) that fills
/// codebooks 1..N of every frame.
#[derive(Debug, Clone)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

/// Geometry and token tables of the talker — the autoregressive backbone that
/// predicts codebook 0 of every frame.
#[derive(Debug, Clone)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    /// Size of the codec-token vocabulary (codes plus control ids).
    pub vocab_size: usize,
    /// Size of the text vocabulary the text track embeds from.
    pub text_vocab_size: usize,
    /// Width of the text embeddings, before they are projected into
    /// [`hidden_size`](Self::hidden_size).
    pub text_hidden_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Codebooks per frame (1 from the talker plus the rest from the code
    /// predictor).
    pub num_code_groups: usize,
    /// Per-section split of the rotary dimensions.
    pub mrope_section: Vec<usize>,
    /// Whether the rotary sections are interleaved rather than concatenated.
    pub mrope_interleaved: bool,

    /// Codec-track control ids.
    pub codec_pad_id: u32,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,

    /// Language name → codec-track token id.
    pub codec_language_id: BTreeMap<String, u32>,
    /// Speaker name → codec-track token id.
    pub spk_id: BTreeMap<String, u32>,
    /// Speaker name → the dialect it implies, when it implies one.
    pub spk_is_dialect: BTreeMap<String, Option<String>>,

    pub code_predictor: CodePredictorConfig,
}

impl TalkerConfig {
    /// Number of query heads per key/value head (the grouped-query factor).
    #[must_use]
    pub fn num_key_value_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Resolves a speaker name (case-insensitive) to its codec-track token id.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::UnsupportedVoice`] when the model has no such
    /// speaker.
    pub fn speaker_id(&self, name: &str) -> Result<u32, Qwen3TtsError> {
        let key = name.to_lowercase();
        self.spk_id.get(&key).copied().ok_or_else(|| {
            Qwen3TtsError::UnsupportedVoice(format!(
                "`{name}` is not a voice of this model (available: {})",
                self.speakers().join(", ")
            ))
        })
    }

    /// The preset speaker names this model offers, sorted.
    #[must_use]
    pub fn speakers(&self) -> Vec<&str> {
        self.spk_id.keys().map(String::as_str).collect()
    }

    /// The language names this model knows, sorted, excluding dialects (those
    /// are reached through the speakers that imply them).
    #[must_use]
    pub fn languages(&self) -> Vec<&str> {
        self.codec_language_id
            .keys()
            .filter(|name| !name.contains("dialect"))
            .map(String::as_str)
            .collect()
    }
}

/// Everything read from a checkpoint's `config.json`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: ModelType,
    /// The published size tag (`0b6`, `1b7`).
    pub model_size: String,
    /// Text-track control ids, shared by every conditioning path.
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    /// Chat-markup ids the prompt's opening role is built from.
    pub im_start_token_id: u32,
    pub assistant_token_id: u32,
    pub talker: TalkerConfig,
}

impl ModelConfig {
    /// Reads and parses a checkpoint's `config.json`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the file is missing or
    /// malformed; see [`parse`](Self::parse).
    pub fn read(model_dir: &Path) -> Result<Self, Qwen3TtsError> {
        Self::parse(&read(model_dir, "config.json")?)
    }

    /// Parses a checkpoint's `config.json`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the JSON is malformed or a
    /// field the pipeline needs is missing or of the wrong type.
    pub fn parse(json: &str) -> Result<Self, Qwen3TtsError> {
        let root: Value = serde_json::from_str(json).map_err(|e| {
            Qwen3TtsError::InvalidModel(format!("config.json: {e}"))
        })?;

        let talker_json = object(&root, "talker_config")?;
        let predictor_json = object(talker_json, "code_predictor_config")?;
        let rope = object(talker_json, "rope_scaling")?;

        let code_predictor = CodePredictorConfig {
            hidden_size: usize_at(predictor_json, "hidden_size")?,
            num_hidden_layers: usize_at(predictor_json, "num_hidden_layers")?,
            num_attention_heads: usize_at(
                predictor_json,
                "num_attention_heads",
            )?,
            num_key_value_heads: usize_at(
                predictor_json,
                "num_key_value_heads",
            )?,
            head_dim: usize_at(predictor_json, "head_dim")?,
            intermediate_size: usize_at(predictor_json, "intermediate_size")?,
            vocab_size: usize_at(predictor_json, "vocab_size")?,
            rms_norm_eps: f64_at(predictor_json, "rms_norm_eps")?,
            rope_theta: f64_at(predictor_json, "rope_theta")?,
        };

        let talker = TalkerConfig {
            hidden_size: usize_at(talker_json, "hidden_size")?,
            num_hidden_layers: usize_at(talker_json, "num_hidden_layers")?,
            num_attention_heads: usize_at(talker_json, "num_attention_heads")?,
            num_key_value_heads: usize_at(talker_json, "num_key_value_heads")?,
            head_dim: usize_at(talker_json, "head_dim")?,
            intermediate_size: usize_at(talker_json, "intermediate_size")?,
            vocab_size: usize_at(talker_json, "vocab_size")?,
            text_vocab_size: usize_at(talker_json, "text_vocab_size")?,
            text_hidden_size: usize_at(talker_json, "text_hidden_size")?,
            rms_norm_eps: f64_at(talker_json, "rms_norm_eps")?,
            rope_theta: f64_at(talker_json, "rope_theta")?,
            num_code_groups: usize_at(talker_json, "num_code_groups")?,
            mrope_section: usize_array(rope, "mrope_section")?,
            mrope_interleaved: rope
                .get("interleaved")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            codec_pad_id: u32_at(talker_json, "codec_pad_id")?,
            codec_bos_id: u32_at(talker_json, "codec_bos_id")?,
            codec_eos_token_id: u32_at(talker_json, "codec_eos_token_id")?,
            codec_think_id: u32_at(talker_json, "codec_think_id")?,
            codec_nothink_id: u32_at(talker_json, "codec_nothink_id")?,
            codec_think_bos_id: u32_at(talker_json, "codec_think_bos_id")?,
            codec_think_eos_id: u32_at(talker_json, "codec_think_eos_id")?,
            codec_language_id: u32_map(talker_json, "codec_language_id")?,
            spk_id: u32_map(talker_json, "spk_id")?,
            spk_is_dialect: dialect_map(talker_json)?,
            code_predictor,
        };

        Ok(ModelConfig {
            model_type: ModelType::parse(string_at(&root, "tts_model_type")?)?,
            model_size: string_at(&root, "tts_model_size")?.to_owned(),
            tts_bos_token_id: u32_at(&root, "tts_bos_token_id")?,
            tts_eos_token_id: u32_at(&root, "tts_eos_token_id")?,
            tts_pad_token_id: u32_at(&root, "tts_pad_token_id")?,
            im_start_token_id: u32_at(&root, "im_start_token_id")?,
            assistant_token_id: u32_at(&root, "assistant_token_id")?,
            talker,
        })
    }

    /// Resolves a target language (case-insensitive) to its codec-track token
    /// id. `auto` — and an absent language — leave the choice to the model,
    /// reported as `None`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::UnsupportedLanguage`] when the model has no
    /// such language.
    pub fn language_id(
        &self,
        language: Option<&str>,
    ) -> Result<Option<u32>, Qwen3TtsError> {
        let Some(language) = language else {
            return Ok(None);
        };
        let key = language.to_lowercase();
        if key == "auto" {
            return Ok(None);
        }
        self.talker
            .codec_language_id
            .get(&key)
            .copied()
            .map(Some)
            .ok_or_else(|| {
                Qwen3TtsError::UnsupportedLanguage(format!(
                    "`{language}` is not a language of this model (available: \
                     auto, {})",
                    self.talker.languages().join(", ")
                ))
            })
    }
}

/// The file holding the codec's geometry inside a checkpoint directory.
pub const CODEC_CONFIG: &str = "speech_tokenizer/config.json";
/// The file holding the codec's weights inside a checkpoint directory.
pub const CODEC_WEIGHTS: &str = "speech_tokenizer/model.safetensors";
/// The file holding the talker's weights inside a checkpoint directory.
pub const TALKER_WEIGHTS: &str = "model.safetensors";

/// Geometry of the codec decoder, read from `speech_tokenizer/config.json`.
///
/// Only the decoding half is modelled: turning frames of codes back into a
/// waveform. The encoder is needed for voice cloning and is not parsed here.
#[derive(Debug, Clone)]
pub struct CodecConfig {
    /// Sample rate of the waveform the decoder produces.
    pub output_sample_rate: u32,
    /// Waveform samples produced per frame of codes.
    pub decode_upsample_rate: usize,
    /// Codebooks per frame the decoder expects.
    pub num_quantizers: usize,
    /// How many of those are semantic (the rest are acoustic residuals).
    pub num_semantic_quantizers: usize,
    /// Entries per codebook.
    pub codebook_size: usize,
    /// Width of a codebook entry after projection.
    pub codebook_dim: usize,
    /// Width the quantizer works in, before projecting out.
    pub quantizer_dim: usize,
    /// Width the transformer stack reads and writes.
    pub latent_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Attention window of the transformer stack, in frames.
    pub sliding_window: usize,
    pub layer_scale_initial_scale: f64,
    /// Channel width the convolutional decoder starts from.
    pub decoder_dim: usize,
    /// Upsampling factors of the convolutional decoder blocks.
    pub upsample_rates: Vec<usize>,
    /// Upsampling factors applied before the convolutional decoder.
    pub upsampling_ratios: Vec<usize>,
}

impl CodecConfig {
    /// Reads and parses the codec's `speech_tokenizer/config.json`, checking
    /// that its upsampling stages account for the declared frame rate.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the file is missing or
    /// malformed, or the stages do not multiply out.
    pub fn read(model_dir: &Path) -> Result<Self, Qwen3TtsError> {
        let cfg = Self::parse(&read(model_dir, CODEC_CONFIG)?)?;
        if cfg.total_upsample() != cfg.decode_upsample_rate {
            return Err(Qwen3TtsError::InvalidModel(format!(
                "codec upsampling stages multiply to {} but the config \
                 declares {}",
                cfg.total_upsample(),
                cfg.decode_upsample_rate
            )));
        }
        Ok(cfg)
    }

    /// Parses `speech_tokenizer/config.json`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the JSON is malformed or a
    /// field the decoder needs is missing.
    pub fn parse(json: &str) -> Result<Self, Qwen3TtsError> {
        let root: Value = serde_json::from_str(json).map_err(|e| {
            Qwen3TtsError::InvalidModel(format!(
                "speech_tokenizer/config.json: {e}"
            ))
        })?;
        let decoder = object(&root, "decoder_config")?;

        let codebook_dim = usize_at(decoder, "codebook_dim")?;
        Ok(CodecConfig {
            output_sample_rate: u32_at(&root, "output_sample_rate")?,
            decode_upsample_rate: usize_at(&root, "decode_upsample_rate")?,
            num_quantizers: usize_at(decoder, "num_quantizers")?,
            num_semantic_quantizers: usize_at(
                decoder,
                "num_semantic_quantizers",
            )?,
            codebook_size: usize_at(decoder, "codebook_size")?,
            codebook_dim,
            // The split quantizer runs at half the codebook width and projects
            // back out to it.
            quantizer_dim: codebook_dim / 2,
            latent_dim: usize_at(decoder, "latent_dim")?,
            hidden_size: usize_at(decoder, "hidden_size")?,
            intermediate_size: usize_at(decoder, "intermediate_size")?,
            num_hidden_layers: usize_at(decoder, "num_hidden_layers")?,
            num_attention_heads: usize_at(decoder, "num_attention_heads")?,
            num_key_value_heads: usize_at(decoder, "num_key_value_heads")?,
            head_dim: usize_at(decoder, "head_dim")?,
            rms_norm_eps: f64_at(decoder, "rms_norm_eps")?,
            rope_theta: f64_at(decoder, "rope_theta")?,
            sliding_window: usize_at(decoder, "sliding_window")?,
            layer_scale_initial_scale: f64_at(
                decoder,
                "layer_scale_initial_scale",
            )?,
            decoder_dim: usize_at(decoder, "decoder_dim")?,
            upsample_rates: usize_array(decoder, "upsample_rates")?,
            upsampling_ratios: usize_array(decoder, "upsampling_ratios")?,
        })
    }

    /// Total waveform samples produced per frame of codes — the product of
    /// every upsampling stage, which must match `decode_upsample_rate`.
    #[must_use]
    pub fn total_upsample(&self) -> usize {
        self.upsample_rates
            .iter()
            .chain(&self.upsampling_ratios)
            .product()
    }
}

/// Sampling defaults read from `generation_config.json`.
///
/// Sampling runs at two levels — the talker and the code predictor — each with
/// its own knobs; the reference enables both by default.
#[derive(Debug, Clone, Copy)]
pub struct GenerationDefaults {
    pub do_sample: bool,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub predictor_do_sample: bool,
    pub predictor_top_k: usize,
    pub predictor_top_p: f32,
    pub predictor_temperature: f32,
    pub max_new_tokens: usize,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            do_sample: true,
            top_k: 50,
            top_p: 1.0,
            temperature: 0.9,
            repetition_penalty: 1.05,
            predictor_do_sample: true,
            predictor_top_k: 50,
            predictor_top_p: 1.0,
            predictor_temperature: 0.9,
            max_new_tokens: 2048,
        }
    }
}

impl GenerationDefaults {
    /// Reads and parses a checkpoint's `generation_config.json`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the file is missing or
    /// malformed.
    pub fn read(model_dir: &Path) -> Result<Self, Qwen3TtsError> {
        Self::parse(&read(model_dir, "generation_config.json")?)
    }

    /// Parses `generation_config.json`, falling back to the reference defaults
    /// for anything it does not state.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the JSON is malformed.
    pub fn parse(json: &str) -> Result<Self, Qwen3TtsError> {
        let root: Value = serde_json::from_str(json).map_err(|e| {
            Qwen3TtsError::InvalidModel(format!("generation_config.json: {e}"))
        })?;
        let mut defaults = GenerationDefaults::default();
        let bool_at = |key: &str, fallback: bool| {
            root.get(key).and_then(Value::as_bool).unwrap_or(fallback)
        };
        let num_at = |key: &str, fallback: f64| {
            root.get(key).and_then(Value::as_f64).unwrap_or(fallback)
        };

        defaults.do_sample = bool_at("do_sample", defaults.do_sample);
        defaults.top_k = num_at("top_k", defaults.top_k as f64) as usize;
        defaults.top_p = num_at("top_p", f64::from(defaults.top_p)) as f32;
        defaults.temperature =
            num_at("temperature", f64::from(defaults.temperature)) as f32;
        defaults.repetition_penalty = num_at(
            "repetition_penalty",
            f64::from(defaults.repetition_penalty),
        ) as f32;
        defaults.predictor_do_sample =
            bool_at("subtalker_dosample", defaults.predictor_do_sample);
        defaults.predictor_top_k =
            num_at("subtalker_top_k", defaults.predictor_top_k as f64) as usize;
        defaults.predictor_top_p =
            num_at("subtalker_top_p", f64::from(defaults.predictor_top_p))
                as f32;
        defaults.predictor_temperature = num_at(
            "subtalker_temperature",
            f64::from(defaults.predictor_temperature),
        ) as f32;
        defaults.max_new_tokens =
            num_at("max_new_tokens", defaults.max_new_tokens as f64) as usize;
        Ok(defaults)
    }
}

/// Reports a field that is missing or of an unexpected type.
fn missing(key: &str) -> Qwen3TtsError {
    Qwen3TtsError::InvalidModel(format!("missing or malformed `{key}`"))
}

/// Borrows a nested object.
fn object<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Qwen3TtsError> {
    value
        .get(key)
        .filter(|found| found.is_object())
        .ok_or_else(|| missing(key))
}

/// Reads a string field.
fn string_at<'a>(
    value: &'a Value,
    key: &str,
) -> Result<&'a str, Qwen3TtsError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| missing(key))
}

/// Reads an unsigned integer field as `usize`.
fn usize_at(value: &Value, key: &str) -> Result<usize, Qwen3TtsError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|found| found as usize)
        .ok_or_else(|| missing(key))
}

/// Reads an unsigned integer field as `u32`.
fn u32_at(value: &Value, key: &str) -> Result<u32, Qwen3TtsError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|found| u32::try_from(found).ok())
        .ok_or_else(|| missing(key))
}

/// Reads a floating-point field (integers in the JSON are accepted).
fn f64_at(value: &Value, key: &str) -> Result<f64, Qwen3TtsError> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| missing(key))
}

/// Reads an array of unsigned integers.
fn usize_array(value: &Value, key: &str) -> Result<Vec<usize>, Qwen3TtsError> {
    let array = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| missing(key))?;
    array
        .iter()
        .map(|item| {
            item.as_u64()
                .map(|found| found as usize)
                .ok_or_else(|| missing(key))
        })
        .collect()
}

/// Reads a `{ name: id }` table, lowercasing the names so lookups are
/// case-insensitive.
fn u32_map(
    value: &Value,
    key: &str,
) -> Result<BTreeMap<String, u32>, Qwen3TtsError> {
    let table = value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| missing(key))?;
    table
        .iter()
        .map(|(name, id)| {
            id.as_u64()
                .and_then(|found| u32::try_from(found).ok())
                .map(|found| (name.to_lowercase(), found))
                .ok_or_else(|| missing(key))
        })
        .collect()
}

/// Reads `spk_is_dialect`, whose values are either `false` or a dialect name.
fn dialect_map(
    value: &Value,
) -> Result<BTreeMap<String, Option<String>>, Qwen3TtsError> {
    const KEY: &str = "spk_is_dialect";
    let Some(table) = value.get(KEY).and_then(Value::as_object) else {
        // Absent for checkpoints without preset speakers.
        return Ok(BTreeMap::new());
    };
    Ok(table
        .iter()
        .map(|(name, dialect)| {
            (name.to_lowercase(), dialect.as_str().map(str::to_owned))
        })
        .collect())
}

#[cfg(test)]
mod tests;
