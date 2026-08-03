//! The two published JSON descriptions the engine reads.
//!
//! `config.json` describes the network's geometry and
//! `preprocessor_config.json` how a picture becomes patches. Both are read as
//! plain JSON, key by key, the way the rest of the tree reads a published
//! description — no serde derives, and no attempt to model fields the port does
//! not use.
//!
//! **One field is deliberately not read: `use_cache`.** The checkpoint ships
//! `"use_cache": false`, which describes a default of the framework it was
//! exported from, not a property of the model: honouring it makes decoding
//! quadratic and nine times slower for byte-identical output. Everything this
//! module takes from the file is geometry.

use std::path::Path;

use serde_json::Value;

use crate::ocr::error::OcrError;

/// The description of the network.
pub const CONFIG_FILE: &str = "config.json";

/// The description of the image preprocessing.
pub const PREPROCESSOR_FILE: &str = "preprocessor_config.json";

/// The tokenizer, in the `tokenizers` crate's own format.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// The weights.
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// The vision tower's geometry.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Side of the square the learned position grid was trained at; the grid
    /// is `(image_size / patch_size)²` slots and is interpolated onto the
    /// actual patch grid of each input.
    pub image_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    /// How many patches per side the projector folds into one token.
    pub spatial_merge_size: usize,
    pub layer_norm_eps: f64,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// The decoder's geometry, plus the token ids that mark an image.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    /// Not `hidden_size / num_attention_heads`: the queries are projected
    /// *wider* than the residual stream (16 heads × 128 = 2048 out of 1024).
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// How the head dimension is split between the three position axes.
    pub mrope_section: Vec<usize>,
    pub image_token_id: u32,
    pub vision: VisionConfig,
}

impl ModelConfig {
    /// Reads `config.json` from a model directory.
    pub fn load(dir: &Path) -> Result<Self, OcrError> {
        let value = read_json(&dir.join(CONFIG_FILE))?;
        let vision = value
            .get("vision_config")
            .ok_or_else(|| bad("the model description has no vision_config"))?;
        Ok(Self {
            hidden_size: usize_at(&value, "hidden_size")?,
            intermediate_size: usize_at(&value, "intermediate_size")?,
            num_attention_heads: usize_at(&value, "num_attention_heads")?,
            num_hidden_layers: usize_at(&value, "num_hidden_layers")?,
            num_key_value_heads: usize_at(&value, "num_key_value_heads")?,
            head_dim: usize_at(&value, "head_dim")?,
            vocab_size: usize_at(&value, "vocab_size")?,
            rms_norm_eps: f64_at(&value, "rms_norm_eps")?,
            rope_theta: f64_at(&value, "rope_theta")?,
            mrope_section: value
                .pointer("/rope_scaling/mrope_section")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("no rope_scaling.mrope_section"))?
                .iter()
                .map(|v| {
                    v.as_u64().map(|n| n as usize).ok_or_else(|| {
                        bad("rope_scaling.mrope_section is not a list of \
                             numbers")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            image_token_id: usize_at(&value, "image_token_id")? as u32,
            vision: VisionConfig {
                image_size: usize_at(vision, "image_size")?,
                hidden_size: usize_at(vision, "hidden_size")?,
                intermediate_size: usize_at(vision, "intermediate_size")?,
                num_attention_heads: usize_at(vision, "num_attention_heads")?,
                num_hidden_layers: usize_at(vision, "num_hidden_layers")?,
                num_channels: usize_at(vision, "num_channels")?,
                patch_size: usize_at(vision, "patch_size")?,
                spatial_merge_size: usize_at(vision, "spatial_merge_size")?,
                layer_norm_eps: f64_at(vision, "layer_norm_eps")?,
            },
        })
    }
}

/// How a picture becomes patches.
#[derive(Debug, Clone)]
pub struct ImageConfig {
    /// Smallest area, in pixels, a picture is scaled up to.
    pub min_pixels: u32,
    /// Largest area, in pixels, a picture is scaled down to.
    pub max_pixels: u32,
    pub patch_size: u32,
    pub merge_size: u32,
    pub rescale_factor: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl ImageConfig {
    /// Reads `preprocessor_config.json` from a model directory.
    pub fn load(dir: &Path) -> Result<Self, OcrError> {
        let value = read_json(&dir.join(PREPROCESSOR_FILE))?;
        Ok(Self {
            min_pixels: usize_at(&value, "min_pixels")? as u32,
            max_pixels: usize_at(&value, "max_pixels")? as u32,
            patch_size: usize_at(&value, "patch_size")? as u32,
            merge_size: usize_at(&value, "merge_size")? as u32,
            rescale_factor: f64_at(&value, "rescale_factor")? as f32,
            image_mean: triple(&value, "image_mean")?,
            image_std: triple(&value, "image_std")?,
        })
    }

    /// Both sides of a resized picture are multiples of this.
    pub fn factor(&self) -> u32 { self.patch_size * self.merge_size }
}

fn read_json(path: &Path) -> Result<Value, OcrError> {
    let bytes = std::fs::read(path).map_err(|source| OcrError::ModelFile {
        path: path.display().to_string(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| {
        OcrError::Artifact(format!("{} is not valid JSON: {e}", path.display()))
    })
}

fn usize_at(value: &Value, key: &str) -> Result<usize, OcrError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| bad(&format!("`{key}` is missing or not a number")))
}

fn f64_at(value: &Value, key: &str) -> Result<f64, OcrError> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| bad(&format!("`{key}` is missing or not a number")))
}

fn triple(value: &Value, key: &str) -> Result<[f32; 3], OcrError> {
    let list = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| bad(&format!("`{key}` is missing or not a list")))?;
    if list.len() != 3 {
        return Err(bad(&format!("`{key}` needs three values")));
    }
    let mut out = [0f32; 3];
    for (slot, item) in out.iter_mut().zip(list) {
        *slot = item
            .as_f64()
            .ok_or_else(|| bad(&format!("`{key}` is not a list of numbers")))?
            as f32;
    }
    Ok(out)
}

fn bad(what: &str) -> OcrError { OcrError::Artifact(what.to_string()) }
