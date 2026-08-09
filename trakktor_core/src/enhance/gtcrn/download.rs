//! Fetching the published checkpoint and converting it once.
//!
//! Upstream carries the weights in its own repository — 580 KB, small enough
//! that the download is over before a progress line can draw. It is pinned to a
//! commit rather than a branch, because a branch is a moving target and these
//! weights have no published digest of their own.
//!
//! The conversion does one thing the runtimes are then spared: it **folds every
//! batch norm into the convolution before it**. In inference a batch norm is an
//! affine map with fixed statistics, so `w' = w·γ/√(σ²+ε)` and
//! `b' = (b−μ)·γ/√(σ²+ε) + β` turn the pair into one convolution. That removes
//! forty-five tensors, a whole layer type from both runtimes, and any chance of
//! the two disagreeing about the epsilon.

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use candle_core::Tensor;

use super::config::BN_EPS;
use crate::{
    download::{Download, Progress},
    enhance::EnhanceError,
};

/// The converted checkpoint every runtime loads.
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// The published checkpoint, pinned to a commit.
const CHECKPOINT_URL: &str = "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/checkpoints/model_trained_on_dns3.tar";
/// What it is called on the way in.
const CHECKPOINT_FILE: &str = "model_trained_on_dns3.tar";
/// Its size, and our own digest — upstream publishes none.
const CHECKPOINT_SIZE: u64 = 579_819;
const CHECKPOINT_BLAKE3: &str =
    "a431fc6d76ee7a580742fe29537d0712064b52abe0346183148fa8016b799218";

/// The only published model. Upstream also ships one trained on VCTK-DEMAND;
/// this is the one trained on DNS3, which is the wider corpus and the fair
/// comparison to the generative engine next door.
pub const DEFAULT_MODEL: &str = "dns3";

/// The published model names.
pub const KNOWN_MODELS: &[&str] = &[DEFAULT_MODEL];

/// Bytes fetched on a cold install.
#[must_use]
pub fn download_size() -> u64 { CHECKPOINT_SIZE }

/// A resolved model: the directory holding its converted weights.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The directory with [`WEIGHTS_FILE`] in it.
    pub dir: PathBuf,
    /// The published name, when the model was named rather than pointed at.
    pub name: Option<String>,
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

    if !KNOWN_MODELS.contains(&model) {
        return Err(EnhanceError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a directory holding \
             {WEIGHTS_FILE})",
            KNOWN_MODELS.join(", ")
        )));
    }

    let dir = models_dir.join("enhance").join("gtcrn");
    let weights = dir.join(WEIGHTS_FILE);
    if weights.is_file() {
        return Ok(ResolvedModel {
            dir,
            name: Some(model.to_owned()),
        });
    }

    crate::download::create_dir(&dir)?;
    let archive = dir.join(CHECKPOINT_FILE);
    if !archive.is_file() {
        Download::new(CHECKPOINT_URL, &archive)
            .label(CHECKPOINT_FILE)
            .blake3(CHECKPOINT_BLAKE3)
            .fetch(progress)?;
    }
    convert(&archive, &weights)?;
    let _ = fs::remove_file(&archive);

    Ok(ResolvedModel {
        dir,
        name: Some(model.to_owned()),
    })
}

/// Converts the published archive into the file the runtimes load.
fn convert(archive: &Path, out: &Path) -> Result<(), EnhanceError> {
    let tensors =
        candle_core::pickle::read_all_with_key(archive, Some("model"))
            .map_err(|e| {
                EnhanceError::Checkpoint(format!(
                    "{}: cannot read its `model` state dictionary ({e})",
                    archive.display()
                ))
            })?;
    let mut all: HashMap<String, Tensor> = tensors.into_iter().collect();

    fold_batch_norms(&mut all)?;
    // A counter torch keeps for its own bookkeeping; nothing reads it here.
    all.retain(|name, _| !name.ends_with("num_batches_tracked"));

    for required in [
        "erb.erb_fc.weight",
        "erb.ierb_fc.weight",
        "encoder.en_convs.0.conv.weight",
        "dpgrnn1.intra_rnn.rnn1.weight_ih_l0",
        "decoder.de_convs.4.conv.weight",
    ] {
        if !all.contains_key(required) {
            return Err(EnhanceError::Checkpoint(format!(
                "the checkpoint has no `{required}`, so it is not GTCRN"
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

/// The convolution each batch norm belongs to. The reference names them in
/// pairs, and the pairing is by name rather than by position, so it is written
/// out rather than guessed.
fn conv_for(norm: &str) -> Option<String> {
    for (norm_suffix, conv_suffix) in [
        (".bn", ".conv"),
        (".point_bn1", ".point_conv1"),
        (".point_bn2", ".point_conv2"),
        (".depth_bn", ".depth_conv"),
    ] {
        if let Some(stem) = norm.strip_suffix(norm_suffix) {
            return Some(format!("{stem}{conv_suffix}"));
        }
    }
    None
}

/// Folds every batch norm into the convolution before it.
fn fold_batch_norms(
    all: &mut HashMap<String, Tensor>,
) -> Result<(), EnhanceError> {
    let norms: Vec<String> = all
        .keys()
        .filter_map(|key| key.strip_suffix(".weight").map(str::to_owned))
        .filter(|stem| all.contains_key(&format!("{stem}.running_var")))
        .collect();

    for norm in norms {
        let conv = conv_for(&norm).ok_or_else(|| {
            EnhanceError::Checkpoint(format!(
                "`{norm}` is a batch norm with no convolution to fold into"
            ))
        })?;
        // Which axis the output channels live on cannot be guessed from the
        // shapes: a 16-to-16 transposed convolution has the same first
        // dimension either way, and guessing wrong scales the kernel across
        // its inputs instead of its outputs — which still runs, and still
        // produces plausible audio, and is wrong. The decoder is the
        // transposed half; that is a fact about the network, so it is read
        // from the name.
        let transposed = conv.starts_with("decoder.");
        let fold = || -> candle_core::Result<(Tensor, Tensor)> {
            let gamma = all[&format!("{norm}.weight")].clone();
            let beta = all[&format!("{norm}.bias")].clone();
            let mean = all[&format!("{norm}.running_mean")].clone();
            let var = all[&format!("{norm}.running_var")].clone();
            let scale = gamma.div(&(var + BN_EPS)?.sqrt()?)?;
            let out_channels = scale.dims1()?;

            let weight = all[&format!("{conv}.weight")].clone();
            let (first, second, height, width) = weight.dims4()?;
            let folded_weight = if transposed {
                // `[in, out/groups, kh, kw]`, and the group an input channel
                // belongs to decides which slice of the scale applies to it.
                let groups = out_channels / second;
                let per_group_in = first / groups;
                weight
                    .reshape((groups, per_group_in, second, height, width))?
                    .broadcast_mul(&scale.reshape((groups, 1, second, 1, 1))?)?
                    .reshape((first, second, height, width))?
            } else {
                // `[out, in/groups, kh, kw]`: the scale is the first axis.
                weight.broadcast_mul(&scale.reshape((first, 1, 1, 1))?)?
            };

            let bias = all[&format!("{conv}.bias")].clone();
            let folded_bias = ((bias - mean)?.mul(&scale)? + beta)?;
            Ok((folded_weight, folded_bias))
        };
        let (weight, bias) = fold().map_err(|e| {
            EnhanceError::Checkpoint(format!("folding `{norm}`: {e}"))
        })?;
        all.insert(format!("{conv}.weight"), weight);
        all.insert(format!("{conv}.bias"), bias);
        for suffix in ["weight", "bias", "running_mean", "running_var"] {
            all.remove(&format!("{norm}.{suffix}"));
        }
    }
    Ok(())
}
