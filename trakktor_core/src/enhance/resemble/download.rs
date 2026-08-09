//! Fetching the published checkpoint and converting it once.
//!
//! Upstream publishes **one** file for both networks — the enhancer's training
//! checkpoint, which carries the denoiser inside it because the enhancer uses
//! it as an input. So there is one download whichever engine is asked for, and
//! it is converted into two files: a small one the denoiser loads on its own,
//! and the whole pipeline.
//!
//! # Three things happen at conversion that cannot happen later
//!
//! - **Weight normalization is folded.** Most of the vocoder and both halves of
//!   the autoencoder store a direction and a length rather than a kernel; the
//!   reference multiplies them back together at inference and so does this,
//!   once, here.
//! - **The mel filterbank is lifted out.** It is a buffer rather than a
//!   parameter — torchaudio's Slaney-scaled matrix — so the published weights
//!   carry the exact matrix and nothing has to be rebuilt from a formula.
//! - **The checkpoint is fp16 and what is written is f32.** The reference loads
//!   those half-precision values into full-precision parameters and folds the
//!   weight normalization there, so folding in f32 is what reproduces it. The
//!   converted pipeline is therefore about twice the size of what was
//!   downloaded, which is the price of the fold being exact.

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Tensor};

use crate::{
    download::{Download, Progress},
    enhance::EnhanceError,
};

/// The converted denoiser, which is all `resemble-denoise` loads.
pub const DENOISER_FILE: &str = "denoiser.safetensors";

/// The converted pipeline.
pub const ENHANCER_FILE: &str = "enhancer.safetensors";

/// The published checkpoint, and its size — upstream publishes no digest, so
/// this one is ours.
const CHECKPOINT_URL: &str = "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt";

/// Bytes fetched on a cold install.
const CHECKPOINT_SIZE: u64 = 713_176_232;

/// The archive's name on disk while it is being converted.
const ARCHIVE: &str = "mp_rank_00_model_states.pt";

/// Bytes fetched on a cold install.
#[must_use]
pub fn download_size() -> u64 { CHECKPOINT_SIZE }

/// A resolved model: the directory holding the converted weights.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The directory with the converted files in it.
    pub dir: PathBuf,
    /// The published name, when the model was named rather than pointed at.
    pub name: Option<String>,
}

/// Where the converted weights live.
#[must_use]
pub fn model_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("enhance").join("resemble")
}

/// The name the published model is asked for by.
pub const DEFAULT_MODEL: &str = "resemble";

/// Resolves `model` to a directory holding converted weights, downloading and
/// converting the published checkpoint on first use.
///
/// `wanted` is the file the caller is about to load; both are written, but only
/// its presence decides whether the work has already been done.
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidModel`] for an unknown name,
/// [`EnhanceError::ModelDownload`] when the checkpoint cannot be fetched, and
/// [`EnhanceError::Checkpoint`] when it cannot be converted.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    wanted: &str,
    progress: Progress<'_>,
) -> Result<ResolvedModel, EnhanceError> {
    let as_path = Path::new(model);
    if as_path.is_dir() {
        if !as_path.join(wanted).is_file() {
            return Err(EnhanceError::InvalidModel(format!(
                "the directory `{model}` holds no {wanted}"
            )));
        }
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            name: None,
        });
    }
    if model != DEFAULT_MODEL {
        return Err(EnhanceError::InvalidModel(format!(
            "unknown model `{model}` (known: {DEFAULT_MODEL}; or pass a \
             directory holding {wanted})"
        )));
    }

    let dir = model_dir(models_dir);
    if dir.join(wanted).is_file() {
        return Ok(ResolvedModel {
            dir,
            name: Some(model.to_owned()),
        });
    }

    crate::download::create_dir(&dir)?;
    let archive = dir.join(ARCHIVE);
    if !archive.is_file() {
        Download::new(CHECKPOINT_URL, &archive)
            .label("the pipeline")
            .fetch(progress)?;
    }
    convert(&archive, &dir)?;
    let _ = fs::remove_file(&archive);

    Ok(ResolvedModel {
        dir,
        name: Some(model.to_owned()),
    })
}

/// Tensors the checkpoint must have for it to be this pipeline at all. One is
/// picked from each part, so a checkpoint of a different shape of model fails
/// here rather than half-way through a load.
const REQUIRED: &[&str] = &[
    "denoiser.net.input_proj.weight",
    "denoiser.net.encoder_blocks.3.res_block2.5.weight",
    "denoiser.net.head.2.weight",
    "lcfm.ae.encoder.0.weight",
    "lcfm.ae.decoder.5.weight",
    "lcfm.cfm.net.layers.29.dconv.weight",
    "vocoder.blocks.3.kernel_predictor.kernel_conv.bias",
    "vocoder.conv_post.1.bias",
    "mel_fn.melspec.mel_scale.fb",
    "mel_fn.melspec.spectrogram.window",
    "normalizer.running_mean_unsafe",
    "normalizer.running_var_unsafe",
];

/// The mel filterbank, under the name the runtimes read it by.
const FILTERBANK_KEY: &str = "mel_filterbank";

/// Its analysis window — the checkpoint's own, half-precision copy, which is
/// what the reference analyses through whatever it computed at build time.
const MEL_WINDOW_KEY: &str = "mel_window";

/// The mel's centring, as a pair — the mean and the standard deviation the
/// training run settled on.
const CENTRE_KEY: &str = "mel_centre";

/// The epsilon under the running variance, from the reference's `Normalizer`.
const CENTRE_EPS: f64 = 1e-9;

/// Converts the published archive into the two files the runtimes load.
fn convert(archive: &Path, dir: &Path) -> Result<(), EnhanceError> {
    let tensors =
        candle_core::pickle::read_all_with_key(archive, Some("module"))
            .map_err(|e| {
                EnhanceError::Checkpoint(format!(
                    "{}: cannot read its `module` state dictionary ({e})",
                    archive.display()
                ))
            })?;
    let raw: HashMap<String, Tensor> = tensors.into_iter().collect();

    for required in REQUIRED {
        if !raw.contains_key(*required) {
            return Err(EnhanceError::Checkpoint(format!(
                "the checkpoint has no `{required}`, so it is not \
                 resemble-enhance"
            )));
        }
    }

    let folded = fold_weight_norm(&raw)?;

    let mut denoiser = HashMap::new();
    for (name, tensor) in &folded {
        // Only the network: the checkpoint also carries the mel transform the
        // denoiser was trained with and never uses at inference.
        if let Some(rest) = name.strip_prefix("denoiser.") &&
            rest.starts_with("net.")
        {
            denoiser.insert(rest.to_owned(), tensor.clone());
        }
    }
    write(&denoiser, &dir.join(DENOISER_FILE))?;
    drop(denoiser);

    let mut enhancer = HashMap::new();
    for (name, tensor) in &folded {
        let keep = name.starts_with("denoiser.net.") ||
            name.starts_with("lcfm.ae.encoder.") ||
            name.starts_with("lcfm.ae.decoder.") ||
            name.starts_with("lcfm.cfm.net.") ||
            name.starts_with("vocoder.");
        if keep {
            enhancer.insert(name.clone(), tensor.clone());
        }
    }
    enhancer.insert(
        FILTERBANK_KEY.to_owned(),
        cast(&raw["mel_fn.melspec.mel_scale.fb"])?,
    );
    enhancer.insert(
        MEL_WINDOW_KEY.to_owned(),
        cast(&raw["mel_fn.melspec.spectrogram.window"])?,
    );
    enhancer.insert(CENTRE_KEY.to_owned(), centre(&raw)?);
    write(&enhancer, &dir.join(ENHANCER_FILE))
}

/// The suffix a normalized parameter's direction is stored under.
const DIRECTION: &str = ".weight_v";

/// And its length.
const LENGTH: &str = ".weight_g";

/// Multiplies weight normalization back into one kernel.
///
/// A normalized parameter is stored as a length per output channel and a
/// direction; the kernel is the direction scaled to that length. Everything
/// else is passed through, widened to `f32` — the reference loads these
/// half-precision values into full-precision parameters before it does the same
/// multiplication.
///
/// The names are the ones torch used before weight normalization became a
/// parametrization; the checkpoint predates the change, and the reference reads
/// it through torch's own compatibility hook.
fn fold_weight_norm(
    raw: &HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, EnhanceError> {
    let mut out = HashMap::with_capacity(raw.len());
    for (name, tensor) in raw {
        let Some(prefix) = name.strip_suffix(DIRECTION) else {
            if name.ends_with(LENGTH) {
                continue;
            }
            out.insert(name.clone(), cast(tensor)?);
            continue;
        };
        let length =
            raw.get(&format!("{prefix}{LENGTH}")).ok_or_else(|| {
                EnhanceError::Checkpoint(format!(
                    "{name} has a direction but no length"
                ))
            })?;
        out.insert(format!("{prefix}.weight"), fold(tensor, length)?);
    }
    Ok(out)
}

/// One folded kernel: `g · v / ‖v‖`, the norm taken over every axis but the
/// first.
///
/// The first axis is the output channels of an ordinary convolution and the
/// **input** channels of a transposed one — and it is the first either way,
/// because that is the axis torch normalizes over by default and the axis the
/// stored length is shaped for. Nothing here has to know which kind it is.
fn fold(direction: &Tensor, length: &Tensor) -> Result<Tensor, EnhanceError> {
    let checkpoint = |what: &str, e: candle_core::Error| {
        EnhanceError::Checkpoint(format!(
            "folding a normalized kernel: {what}: {e}"
        ))
    };
    let direction = cast(direction)?;
    let length = cast(length)?;
    let axes: Vec<usize> = (1..direction.rank()).collect();
    let norm = direction
        .sqr()
        .and_then(|t| t.sum_keepdim(axes))
        .and_then(|t| t.sqrt())
        .map_err(|e| checkpoint("the norm", e))?;
    direction
        .broadcast_div(&norm)
        .and_then(|t| t.broadcast_mul(&length))
        .map_err(|e| checkpoint("the scaling", e))
}

/// A tensor as `f32`.
fn cast(tensor: &Tensor) -> Result<Tensor, EnhanceError> {
    tensor.to_dtype(DType::F32).map_err(|e| {
        EnhanceError::Checkpoint(format!("widening a tensor to f32: {e}"))
    })
}

/// The mel's centring, as the mean and the standard deviation the runtimes
/// subtract and divide by. The checkpoint stores a **variance**, and the
/// reference takes its square root with a floor under it every time it is used.
fn centre(raw: &HashMap<String, Tensor>) -> Result<Tensor, EnhanceError> {
    let scalar = |name: &str| -> Result<f64, EnhanceError> {
        let value = raw[name]
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
            .map_err(|e| {
                EnhanceError::Checkpoint(format!("reading {name}: {e}"))
            })?;
        Ok(f64::from(value[0]))
    };
    let mean = scalar("normalizer.running_mean_unsafe")?;
    let variance = scalar("normalizer.running_var_unsafe")?;
    if !mean.is_finite() || !variance.is_finite() {
        return Err(EnhanceError::Checkpoint(
            "the checkpoint's mel centring was never estimated".into(),
        ));
    }
    Tensor::from_vec(
        vec![mean as f32, (variance + CENTRE_EPS).sqrt() as f32],
        2,
        &candle_core::Device::Cpu,
    )
    .map_err(|e| EnhanceError::Checkpoint(format!("the mel centring: {e}")))
}

/// Writes one converted file, through a temporary so that an interrupted run
/// leaves nothing that looks finished.
fn write(
    tensors: &HashMap<String, Tensor>,
    out: &Path,
) -> Result<(), EnhanceError> {
    let temp = out.with_extension("partial");
    candle_core::safetensors::save(tensors, &temp).map_err(|e| {
        EnhanceError::Checkpoint(format!("writing {}: {e}", temp.display()))
    })?;
    fs::rename(&temp, out).map_err(|e| {
        EnhanceError::Checkpoint(format!("finalizing {}: {e}", out.display()))
    })
}
