//! Model resolution, download, and the one-time conversion of the published
//! checkpoints into the file the runtimes load.
//!
//! Upstream publishes the pipeline as four separate `torch.save` archives —
//! 2.19 GB between them, all `f32`. They are downloaded once, folded into a
//! single `model.safetensors` with one prefix per network, and the archives are
//! deleted. Two things happen during the fold that cannot happen later:
//!
//! - the encoder's positional convolution is stored **weight-normalized**, as a
//!   direction and a per-tap gain; the conversion multiplies them back into one
//!   kernel, so no runtime has to know that the parameterization existed;
//! - the vocoder's Hann window rides along in the checkpoint as a buffer. It is
//!   dropped — [`istft::hann`](super::istft::hann) recomputes it, and a window
//!   that disagreed would show up in the very first parity check.
//!
//! Everything else is copied through under its own name, so a tensor in the
//! converted file can be found in the reference by stripping one prefix.

#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use candle_core::Tensor;

use crate::{
    download::{self, Download, Progress},
    enhance::EnhanceError,
};

/// The converted checkpoint every runtime loads.
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// The repository the checkpoints are published in.
const REPO: &str = "Xiaobin-Rong/unipase";

/// The only published model. Named so a second one — a lighter masking network,
/// say — can be added next to it without changing the command.
pub const DEFAULT_MODEL: &str = "unipase";

/// The published model names.
pub const KNOWN_MODELS: &[&str] = &[DEFAULT_MODEL];

/// One published archive: what it is called upstream, how big it is, its
/// digest, and the prefix its tensors take in the converted file.
struct Part {
    file: &'static str,
    prefix: &'static str,
    size: u64,
    blake3: &'static str,
}

/// The four networks of the pipeline, in the order they run.
///
/// Two of the five published archives are deliberately absent.
/// `Vocoder_WavLM-L24.pt` is upstream's way of reconstructing waveforms from
/// the encoder's deep layer while *evaluating* the encoder, and takes no part
/// in enhancement. `PostNet.pt` is the bandwidth extender, which this version
/// does not run — see the engine's own documentation for why, and what
/// `--sample-rate` does instead.
const PARTS: &[Part] = &[
    Part {
        file: "DeWavLM-Omni.pt",
        prefix: "wavlm",
        size: 1_261_992_190,
        blake3:
            "3dd5b257f5b11401177fd27254dbf9cf846a91fd3a3d6a1635519f8590de7174",
    },
    Part {
        file: "Adapter.pt",
        prefix: "adapter",
        size: 458_130_478,
        blake3:
            "645c6a2f8034d126864d52a02052928e68d09eb17c12fafea7949d29b53ac798",
    },
    Part {
        file: "Vocoder_DWO-L1.pt",
        prefix: "vocoder",
        size: 454_996_467,
        blake3:
            "d228c0f116dbf77e2dd17b0a3f59c9800df69ede71b3d4494ed37949cc3e2347",
    },
];

/// Bytes fetched on a cold install, for a caller that wants to say so before
/// the first byte moves.
#[must_use]
pub fn download_size() -> u64 { PARTS.iter().map(|part| part.size).sum() }

/// A resolved model: the directory holding its converted weights.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The directory with [`WEIGHTS_FILE`] in it.
    pub dir: PathBuf,
    /// The published name, when the model was named rather than pointed at.
    pub name: Option<String>,
}

/// Resolves `model` to a directory holding converted weights, downloading and
/// converting the published checkpoints on first use.
///
/// A path to a directory that already holds [`WEIGHTS_FILE`] is used as it is.
///
/// # Errors
///
/// Returns [`EnhanceError::InvalidModel`] for an unknown name,
/// [`EnhanceError::ModelDownload`] when a checkpoint cannot be fetched, and
/// [`EnhanceError::Checkpoint`] when one cannot be converted.
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

    let dir = models_dir.join("enhance").join("unipase");
    let weights = dir.join(WEIGHTS_FILE);
    if weights.is_file() {
        return Ok(ResolvedModel {
            dir,
            name: Some(model.to_owned()),
        });
    }

    download::create_dir(&dir)?;
    let mut archives = Vec::new();
    for part in PARTS {
        let archive = dir.join(part.file);
        if !archive.is_file() {
            let url = download::hugging_face_url(REPO, part.file);
            Download::new(&url, &archive)
                .label(part.file)
                .blake3(part.blake3)
                .fetch(progress)?;
        }
        archives.push((archive, part));
    }
    convert(&archives, &weights)?;
    for (archive, _) in &archives {
        let _ = fs::remove_file(archive);
    }

    Ok(ResolvedModel {
        dir,
        name: Some(model.to_owned()),
    })
}

/// The weight-normalized kernel's two halves, as the reference stores them.
const WEIGHT_G: &str = "wavlm.encoder.pos_conv.0.weight_g";
const WEIGHT_V: &str = "wavlm.encoder.pos_conv.0.weight_v";
/// The name the folded kernel takes.
const POS_CONV_WEIGHT: &str = "wavlm.encoder.pos_conv.0.weight";
/// The window the conversion drops.
const ISTFT_WINDOW: &str = "head.istft.window";

/// Folds the four archives into one safetensors file.
fn convert(
    archives: &[(PathBuf, &'static Part)],
    out: &Path,
) -> Result<(), EnhanceError> {
    let mut all: HashMap<String, Tensor> = HashMap::new();
    for (archive, part) in archives {
        let tensors = read_state_dict(archive)?;
        for (name, tensor) in tensors {
            if name.ends_with(ISTFT_WINDOW) {
                continue;
            }
            if tensor.dtype() != candle_core::DType::F32 {
                return Err(EnhanceError::Checkpoint(format!(
                    "{}: `{name}` is stored as {:?}, expected f32",
                    archive.display(),
                    tensor.dtype()
                )));
            }
            all.insert(format!("{}.{name}", part.prefix), tensor);
        }
    }

    fold_weight_norm(&mut all)?;

    for required in [
        "wavlm.mask_emb",
        POS_CONV_WEIGHT,
        "adapter.head.weight",
        "vocoder.head.out.weight",
    ] {
        if !all.contains_key(required) {
            return Err(EnhanceError::Checkpoint(format!(
                "the converted checkpoint has no `{required}`, so it is not \
                 the published pipeline"
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

/// Reads the `model` half of a published archive.
fn read_state_dict(
    archive: &Path,
) -> Result<Vec<(String, Tensor)>, EnhanceError> {
    candle_core::pickle::read_all_with_key(archive, Some("model")).map_err(
        |e| {
            EnhanceError::Checkpoint(format!(
                "{}: cannot read its `model` state dictionary ({e})",
                archive.display()
            ))
        },
    )
}

/// Multiplies the positional convolution's direction and gain into one kernel.
///
/// The parameterization is `weight = g · v / ‖v‖`, with the norm taken over
/// everything but the tap axis — which is why `g` is `[1, 1, 128]` and not a
/// scalar. Folding it here means the runtimes see an ordinary grouped
/// convolution.
fn fold_weight_norm(
    all: &mut HashMap<String, Tensor>,
) -> Result<(), EnhanceError> {
    let failed = |what: String| EnhanceError::Checkpoint(what);
    let (Some(gain), Some(direction)) =
        (all.remove(WEIGHT_G), all.remove(WEIGHT_V))
    else {
        return Err(failed(format!(
            "the encoder's positional convolution is missing its \
             weight-normalized halves (`{WEIGHT_G}`, `{WEIGHT_V}`)"
        )));
    };
    let dims = direction.dims3().map_err(|e| {
        failed(format!("`{WEIGHT_V}` is not a 3-D kernel: {e}"))
    })?;
    let build = || -> candle_core::Result<Tensor> {
        // ‖v‖ over the output and input axes, one value per tap.
        let norm = direction
            .sqr()?
            .sum_keepdim(0)?
            .sum_keepdim(1)?
            .sqrt()?
            .reshape((1, 1, dims.2))?;
        direction
            .broadcast_div(&norm)?
            .broadcast_mul(&gain.reshape((1, 1, dims.2))?)
    };
    let folded = build().map_err(|e| {
        failed(format!("folding the positional convolution's norm: {e}"))
    })?;
    all.insert(POS_CONV_WEIGHT.to_owned(), folded);
    Ok(())
}
