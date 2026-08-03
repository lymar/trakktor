//! The pre/post-processing description shipped with each PaddleOCR model.
//!
//! Every published model directory carries the same description twice:
//! `inference.yml` and `config.json`, key for key. trakktor reads the JSON —
//! it is the same contract without a YAML parser in the dependency tree — and
//! downloads only that half of the pair.
//!
//! The description is a *description*, not the runtime's configuration: the
//! reference pipeline takes its thresholds from its own defaults and reads
//! this file only for the model name, the recognizer's character dictionary
//! and the classifier's labels. This port does the same, and uses the rest of
//! the file as a consistency check on the artifact it just downloaded.
//!
//! Two shapes have to be tolerated. `scale` is a number for some models and
//! the string `"1./255."` for others (upstream passes it through `eval`), and
//! `PostProcess` is flat with a `name` for the detector and the recognizer but
//! a single-key map for the text-line orientation classifier. The transform
//! list also carries training-pipeline entries (`DetLabelEncode`,
//! `MultiLabelEncode`, `KeepKeys`) that leaked in from the training config;
//! they have no meaning at inference and are skipped.

#[cfg(test)]
mod tests;

use std::path::Path;

use serde_json::Value;

use crate::ocr::error::OcrError;

/// The file name every published model directory carries.
pub const CONFIG_FILE: &str = "config.json";

/// What the model directory says about the model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Upstream's own name for the model, e.g. `eslav_PP-OCRv5_mobile_rec`.
    pub model_name: String,
    /// ImageNet-style normalization, when the model declares one.
    pub normalize: Option<Normalize>,
    /// `[channels, height, width]` a recognizer was exported for.
    pub rec_image_shape: Option<[usize; 3]>,
    /// `[width, height]` a classifier resizes its input to.
    pub cls_image_size: Option<[usize; 2]>,
    pub post: PostProcess,
}

/// `(pixel * scale - mean) / std`, per channel.
#[derive(Debug, Clone)]
pub struct Normalize {
    pub scale: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// The post-processing the model was exported for.
#[derive(Debug, Clone)]
pub enum PostProcess {
    /// Differentiable-binarization text detection.
    Db(DbParams),
    /// CTC text recognition over a character dictionary.
    Ctc { characters: Vec<String> },
    /// Top-1 classification over a label list.
    Topk { labels: Vec<String> },
}

/// Thresholds the detector's published description carries.
#[derive(Debug, Clone)]
pub struct DbParams {
    pub thresh: f32,
    pub box_thresh: f32,
    pub max_candidates: usize,
    pub unclip_ratio: f32,
}

impl ModelConfig {
    /// Reads `config.json` from a model directory.
    pub fn load(dir: &Path) -> Result<Self, OcrError> {
        let path = dir.join(CONFIG_FILE);
        let bytes =
            std::fs::read(&path).map_err(|source| OcrError::ModelFile {
                path: path.display().to_string(),
                source,
            })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| {
            OcrError::Artifact(format!(
                "{} is not valid JSON: {e}",
                path.display()
            ))
        })?;
        Self::parse(&value)
    }

    fn parse(value: &Value) -> Result<Self, OcrError> {
        let model_name = value
            .pointer("/Global/model_name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| bad("the model description carries no name"))?
            .to_string();

        let mut normalize = None;
        let mut rec_image_shape = None;
        let mut cls_image_size = None;
        let ops = value
            .pointer("/PreProcess/transform_ops")
            .and_then(|o| o.as_array())
            .map(Vec::as_slice)
            .unwrap_or_default();
        for op in ops {
            let Some((name, params)) =
                op.as_object().and_then(|o| o.iter().next())
            else {
                continue;
            };
            match name.as_str() {
                "NormalizeImage" => normalize = Some(parse_normalize(params)?),
                "RecResizeImg" => {
                    let shape = int_array(params.get("image_shape"))?;
                    let [c, h, w] = shape[..] else {
                        return Err(bad(
                            "a recognizer's image shape is not three numbers"
                        ));
                    };
                    rec_image_shape = Some([c, h, w]);
                },
                // Named for the tensor layout it produces, this one is
                // `[width, height]` — the opposite order of everything else in
                // the same file.
                "ResizeImage" => {
                    let size = int_array(params.get("size"))?;
                    let [w, h] = size[..] else {
                        return Err(bad(
                            "a classifier's image size is not two numbers"
                        ));
                    };
                    cls_image_size = Some([w, h]);
                },
                _ => {},
            }
        }

        let post = value.get("PostProcess").ok_or_else(|| {
            bad("the model description has no post-processing")
        })?;
        let post = parse_post(post)?;

        Ok(Self {
            model_name,
            normalize,
            rec_image_shape,
            cls_image_size,
            post,
        })
    }

    /// The character dictionary of a recognizer, if this is one.
    pub fn characters(&self) -> Option<&[String]> {
        match &self.post {
            PostProcess::Ctc { characters } => Some(characters),
            _ => None,
        }
    }
}

fn parse_post(value: &Value) -> Result<PostProcess, OcrError> {
    if let Some(name) = value.get("name").and_then(|n| n.as_str()) {
        return match name {
            "DBPostProcess" => Ok(PostProcess::Db(DbParams {
                thresh: number(value.get("thresh"), 0.3)?,
                box_thresh: number(value.get("box_thresh"), 0.6)?,
                max_candidates: number(value.get("max_candidates"), 1000.0)?
                    as usize,
                unclip_ratio: number(value.get("unclip_ratio"), 1.5)?,
            })),
            "CTCLabelDecode" => {
                let characters = value
                    .get("character_dict")
                    .and_then(|d| d.as_array())
                    .ok_or_else(|| {
                        bad("a recognizer carries no character dictionary")
                    })?
                    .iter()
                    .map(|c| {
                        c.as_str().map(str::to_string).ok_or_else(|| {
                            bad("a dictionary entry is not a string")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if characters.is_empty() {
                    return Err(bad("the character dictionary is empty"));
                }
                Ok(PostProcess::Ctc { characters })
            },
            other => Err(OcrError::Artifact(format!(
                "unsupported post-processing `{other}`"
            ))),
        };
    }

    // The classifier's shape: a single-key map naming the class instead.
    if let Some(topk) = value.get("Topk") {
        let labels = topk
            .get("label_list")
            .and_then(|l| l.as_array())
            .ok_or_else(|| bad("a classifier carries no labels"))?
            .iter()
            .map(|l| {
                l.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| bad("a label is not a string"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(PostProcess::Topk { labels });
    }

    Err(bad("unrecognized post-processing"))
}

fn parse_normalize(params: &Value) -> Result<Normalize, OcrError> {
    Ok(Normalize {
        scale: scale(params.get("scale"))?,
        mean: triple(params.get("mean"), "mean")?,
        std: triple(params.get("std"), "std")?,
    })
}

/// `scale` is a number in some descriptions and an unevaluated Python
/// expression (`"1./255."`) in others.
fn scale(value: Option<&Value>) -> Result<f32, OcrError> {
    match value {
        Some(Value::Number(n)) => Ok(n.as_f64().unwrap_or_default() as f32),
        Some(Value::String(text)) => {
            let (num, den) = text.split_once('/').unwrap_or((text, "1"));
            let parse = |part: &str| {
                part.trim().trim_end_matches('.').parse::<f64>().ok()
            };
            match (parse(num), parse(den)) {
                (Some(num), Some(den)) if den != 0.0 => Ok((num / den) as f32),
                _ => Err(OcrError::Artifact(format!(
                    "cannot read the normalization scale `{text}`"
                ))),
            }
        },
        _ => Err(bad("a normalization carries no scale")),
    }
}

fn triple(value: Option<&Value>, what: &str) -> Result<[f32; 3], OcrError> {
    let items = value.and_then(|v| v.as_array()).ok_or_else(|| {
        OcrError::Artifact(format!("normalization {what} is missing"))
    })?;
    let [a, b, c] = &items[..] else {
        return Err(OcrError::Artifact(format!(
            "normalization {what} is not three numbers"
        )));
    };
    let as_f32 = |v: &Value| v.as_f64().map(|f| f as f32);
    match (as_f32(a), as_f32(b), as_f32(c)) {
        (Some(a), Some(b), Some(c)) => Ok([a, b, c]),
        _ => Err(OcrError::Artifact(format!(
            "normalization {what} is not numeric"
        ))),
    }
}

fn int_array(value: Option<&Value>) -> Result<Vec<usize>, OcrError> {
    value
        .and_then(|v| v.as_array())
        .ok_or_else(|| bad("expected a list of numbers"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| bad("expected a whole number"))
        })
        .collect()
}

fn number(value: Option<&Value>, default: f32) -> Result<f32, OcrError> {
    match value {
        None => Ok(default),
        Some(Value::Number(n)) => Ok(n.as_f64().unwrap_or_default() as f32),
        Some(_) => Err(bad("expected a number")),
    }
}

fn bad(message: &str) -> OcrError { OcrError::Artifact(message.to_string()) }
