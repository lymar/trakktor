//! Fetching a published checkpoint and converting it once.
//!
//! Upstream carries both of its generators in its own repository — nine
//! megabytes each — pinned here to a commit rather than a branch, because a
//! branch is a moving target and these weights have no published digest of
//! their own.
//!
//! The conversion is deliberately thin: it lifts the `generator` state
//! dictionary out of the archive and writes it as safetensors. There is
//! nothing to fold. The normalizations in this network are *instance* norms —
//! they take their statistics from the window being enhanced, not from
//! training — so unlike a batch norm they are a computation and not a stored
//! affine map, and they have to happen at run time.

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use candle_core::Tensor;

use crate::{
    download::{Download, Progress},
    enhance::EnhanceError,
};

/// The converted checkpoint every runtime loads.
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// The commit the checkpoints are pinned to.
const PIN: &str = "89932cfe90d1dacb8e170e4a331d762462c21792";

/// One published generator.
struct Checkpoint {
    /// The name the model is asked for by.
    name: &'static str,
    /// The file in the upstream repository.
    file: &'static str,
    /// Its size, and our own digest — upstream publishes none.
    size: u64,
    blake3: &'static str,
}

/// The published generators. `dns` is trained on the DNS Challenge data and
/// `vb` on VoiceBank+DEMAND; the first is the wider corpus and the fairer match
/// to a recording that was not made in a studio.
const CHECKPOINTS: &[Checkpoint] = &[
    Checkpoint {
        name: "dns",
        file: "g_best_dns",
        size: 9_138_054,
        blake3:
            "0cba9b6c8c1e5a99a0bac9f1a973707990781324c393aeaf837e87ed31c9863e",
    },
    Checkpoint {
        name: "vb",
        file: "g_best_vb",
        size: 9_142_950,
        blake3:
            "223eb80a44d0cde46ffab617a1b6cb852085d2f53254066cb1d0a6bee3863d00",
    },
];

/// The model used when none is named.
pub const DEFAULT_MODEL: &str = "dns";

/// The published model names.
pub const KNOWN_MODELS: &[&str] = &["dns", "vb"];

/// Bytes fetched on a cold install of `model`.
#[must_use]
pub fn download_size(model: &str) -> u64 {
    find(model).map_or(0, |checkpoint| checkpoint.size)
}

/// The published checkpoint of that name, if there is one.
fn find(model: &str) -> Option<&'static Checkpoint> {
    CHECKPOINTS.iter().find(|entry| entry.name == model)
}

/// A resolved model: the directory holding its converted weights.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The directory with [`WEIGHTS_FILE`] in it.
    pub dir: PathBuf,
    /// The published name, when the model was named rather than pointed at.
    pub name: Option<String>,
}

/// Where a published model's converted weights live.
#[must_use]
pub fn model_dir(models_dir: &Path, model: &str) -> PathBuf {
    models_dir.join("enhance").join("mpsenet").join(model)
}

/// Resolves `model` to a directory holding converted weights, downloading and
/// converting the published checkpoint on first use.
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidModel`] for an unknown name,
/// [`EnhanceError::ModelDownload`] when the checkpoint cannot be fetched, and
/// [`EnhanceError::Checkpoint`] when it cannot be converted.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: Progress<'_>,
) -> Result<ResolvedModel, EnhanceError> {
    let as_path = Path::new(model);
    if as_path.is_dir() {
        if !as_path.join(WEIGHTS_FILE).is_file() {
            return Err(EnhanceError::InvalidModel(format!(
                "the directory `{model}` holds no {WEIGHTS_FILE}"
            )));
        }
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            name: None,
        });
    }

    let Some(checkpoint) = find(model) else {
        return Err(EnhanceError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a directory holding \
             {WEIGHTS_FILE})",
            KNOWN_MODELS.join(", ")
        )));
    };

    let dir = model_dir(models_dir, model);
    let weights = dir.join(WEIGHTS_FILE);
    if weights.is_file() {
        return Ok(ResolvedModel {
            dir,
            name: Some(model.to_owned()),
        });
    }

    crate::download::create_dir(&dir)?;
    let archive = dir.join(checkpoint.file);
    if !archive.is_file() {
        let url = format!(
            "https://raw.githubusercontent.com/yxlu-0102/MP-SENet/{PIN}/best_ckpt/{}",
            checkpoint.file
        );
        Download::new(&url, &archive)
            .label(checkpoint.file)
            .blake3(checkpoint.blake3)
            .fetch(progress)?;
    }
    convert(&archive, &weights)?;
    let _ = fs::remove_file(&archive);

    Ok(ResolvedModel {
        dir,
        name: Some(model.to_owned()),
    })
}

/// Tensors the checkpoint must have for it to be this network at all. One is
/// picked from each part, so a checkpoint of a different shape of model fails
/// here rather than half-way through a load.
const REQUIRED: &[&str] = &[
    "dense_encoder.dense_conv_1.0.weight",
    "dense_encoder.dense_block.dense_block.3.1.weight",
    "dense_encoder.dense_conv_2.0.weight",
    "TSTransformer.0.time_transformer.attention.in_proj_weight",
    "TSTransformer.3.freq_transformer.ffn.gru.weight_ih_l0_reverse",
    "mask_decoder.lsigmoid.slope",
    "phase_decoder.phase_conv_i.weight",
];

/// Converts the published archive into the file the runtimes load.
fn convert(archive: &Path, out: &Path) -> Result<(), EnhanceError> {
    let tensors =
        candle_core::pickle::read_all_with_key(archive, Some("generator"))
            .map_err(|e| {
                EnhanceError::Checkpoint(format!(
                    "{}: cannot read its `generator` state dictionary ({e})",
                    archive.display()
                ))
            })?;
    let all: HashMap<String, Tensor> = tensors.into_iter().collect();

    for required in REQUIRED {
        if !all.contains_key(*required) {
            return Err(EnhanceError::Checkpoint(format!(
                "the checkpoint has no `{required}`, so it is not MP-SENet"
            )));
        }
    }

    let temp = out.with_extension("partial");
    candle_core::safetensors::save(&all, &temp).map_err(|e| {
        EnhanceError::Checkpoint(format!("writing {}: {e}", temp.display()))
    })?;
    fs::rename(&temp, out).map_err(|e| {
        EnhanceError::Checkpoint(format!("finalizing {}: {e}", out.display()))
    })
}
