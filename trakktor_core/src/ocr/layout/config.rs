//! The description shipped with a layout model.
//!
//! A detection model's `config.json` is a *different shape* from a
//! recognizer's: the pre-processing is a flat `Preprocess` list rather than
//! `PreProcess.transform_ops`, and instead of a `PostProcess` block it carries
//! `label_list` — the class names, in class-index order.
//!
//! Two fields of it are read and one is deliberately not believed:
//!
//! - `label_list` gives the names, and their order **is** the class order of
//!   the network's output. The port checks the names against its own list
//!   rather than trusting the file blindly, because a silently reordered list
//!   would produce plausible nonsense rather than an error.
//! - `Resize.target_size` gives the square the page is squashed into, and
//!   `Resize.interp` the filter (2 = bicubic).
//! - `NormalizeImage` says `norm_type: none`, `mean: [0,0,0]`, `std: [1,1,1]` —
//!   and **is not the whole story**. Upstream's builder defaults the absent
//!   `is_scale` to true and rewrites `none` into `mean_std`, so the input is
//!   divided by 255 after all. Following the file literally feeds the network
//!   values a hundred times too large, and it answers with noise. The port
//!   divides.

#[cfg(test)]
mod tests;

use std::path::Path;

use serde_json::Value;

use crate::ocr::{
    error::OcrError,
    layout::region::{LABEL_NAMES, Label},
};

/// The file name every published model directory carries.
pub const CONFIG_FILE: &str = "config.json";

/// What a layout model's directory says about it.
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Upstream's own name for the model.
    pub model_name: String,
    /// `[height, width]` the page is resized to. Square in every published
    /// model of this family.
    pub target_size: [usize; 2],
    /// The resize filter, as OpenCV numbers them; 2 is bicubic.
    pub interp: u32,
    /// The class names, in class-index order.
    pub labels: Vec<String>,
}

impl LayoutConfig {
    /// Reads `config.json` from a model directory.
    pub fn load(dir: &Path) -> Result<Self, OcrError> {
        let path = dir.join(CONFIG_FILE);
        let bytes =
            std::fs::read(&path).map_err(|source| OcrError::ModelFile {
                path: path.display().to_string(),
                source,
            })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| {
            bad(format!("{} is not valid JSON: {e}", path.display()))
        })?;
        Self::parse(&value)
    }

    fn parse(value: &Value) -> Result<Self, OcrError> {
        let model_name = value
            .pointer("/Global/model_name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| bad("the model description carries no name".into()))?
            .to_string();

        let steps = value
            .get("Preprocess")
            .and_then(|p| p.as_array())
            .ok_or_else(|| {
                bad("the model description has no pre-processing".into())
            })?;
        let resize = steps
            .iter()
            .find(|step| {
                step.get("type").and_then(|t| t.as_str()) == Some("Resize")
            })
            .ok_or_else(|| bad("the pre-processing has no resize".into()))?;
        let size = resize
            .get("target_size")
            .and_then(|s| s.as_array())
            .ok_or_else(|| bad("the resize has no target size".into()))?;
        let [height, width] = &size[..] else {
            return Err(bad("the target size is not two numbers".into()));
        };
        let dimension = |v: &Value| {
            v.as_u64()
                .filter(|n| *n > 0)
                .map(|n| n as usize)
                .ok_or_else(|| bad("a target size is not a size".into()))
        };
        let target_size = [dimension(height)?, dimension(width)?];
        // Keeping the aspect ratio would mean padding, and the whole geometry
        // of the output — boxes are scaled straight back by the ratio the page
        // was squashed by — assumes it was not kept.
        if resize.get("keep_ratio").and_then(|k| k.as_bool()) == Some(true) {
            return Err(bad("this model keeps the aspect ratio on resize, \
                            which this port does not implement"
                .into()));
        }
        let interp =
            resize.get("interp").and_then(|i| i.as_u64()).unwrap_or(2) as u32;

        let labels = value
            .get("label_list")
            .and_then(|l| l.as_array())
            .ok_or_else(|| bad("the model description has no labels".into()))?
            .iter()
            .map(|l| {
                l.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| bad("a label is not a string".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            model_name,
            target_size,
            interp,
            labels,
        })
    }

    /// Checks that the published labels are the ones this port knows, in the
    /// order it knows them.
    ///
    /// The order is the whole of the class-index mapping, and nothing else in
    /// the artifact records it. A file that renamed or reordered them would
    /// otherwise be read as a working model that calls a picture "text".
    pub fn check_labels(&self) -> Result<(), OcrError> {
        if self.labels.len() != Label::COUNT {
            return Err(bad(format!(
                "the model has {} classes, this port knows {}",
                self.labels.len(),
                Label::COUNT
            )));
        }
        for (at, (published, known)) in
            self.labels.iter().zip(LABEL_NAMES).enumerate()
        {
            if published != known {
                return Err(bad(format!(
                    "class {at} is `{published}` in the model and `{known}` \
                     in this port"
                )));
            }
        }
        Ok(())
    }
}

fn bad(message: String) -> OcrError { OcrError::Artifact(message) }
