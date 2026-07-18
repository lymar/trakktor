//! Model and tokenizer resolution and download.
//!
//! A model is referred to either by a published `sat-*-sm` name or by a path to
//! a local checkpoint directory. Named models are cached under the model
//! directory as `text/structify/<name>/` (`config.json` + `model.safetensors`),
//! downloaded from the official repository on first use. The XLM-R tokenizer is
//! shared across models and cached once as `text/structify/tokenizer.json`.

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use super::error::StructifyError;

/// The weight-file names a checkpoint may carry: the `-sm` models publish
/// safetensors, the base models only a PyTorch checkpoint (loaded via candle's
/// pickle reader).
const SAFETENSORS: &str = "model.safetensors";
const PYTORCH_BIN: &str = "pytorch_model.bin";

/// Published model names and the weight file each ships. The repo slug equals
/// the name (all under `segment-any-text/`).
///
/// The `-sm` models score sentence boundaries; the base
/// `-no-limited-lookahead` model scores newline (paragraph) boundaries — a
/// higher `--threshold` there yields coarse, reader-style paragraphs.
pub const KNOWN_MODELS: &[(&str, &str)] = &[
    // Base (paragraph) family — full-context, `pytorch_model.bin`.
    ("sat-1l-no-limited-lookahead", PYTORCH_BIN),
    ("sat-3l-no-limited-lookahead", PYTORCH_BIN),
    ("sat-6l-no-limited-lookahead", PYTORCH_BIN),
    ("sat-9l-no-limited-lookahead", PYTORCH_BIN),
    ("sat-12l-no-limited-lookahead", PYTORCH_BIN),
    // Supervised (sentence) family — safetensors.
    ("sat-1l-sm", SAFETENSORS),
    ("sat-3l-sm", SAFETENSORS),
    ("sat-6l-sm", SAFETENSORS),
    ("sat-9l-sm", SAFETENSORS),
    ("sat-12l-sm", SAFETENSORS),
];

/// The weight file a checkpoint directory carries, when named. A local
/// directory is probed for either at load time.
#[must_use]
pub fn weight_file(model: &str) -> Option<&'static str> {
    KNOWN_MODELS
        .iter()
        .find(|(name, _)| *name == model)
        .map(|&(_, file)| file)
}

/// The Hugging Face repo of the shared XLM-R tokenizer.
const TOKENIZER_REPO: &str = "FacebookAI/xlm-roberta-base";
/// The tokenizer file name (both in the repo and in the cache).
const TOKENIZER_FILE: &str = "tokenizer.json";

/// A resolved model: where its checkpoint lives, and its canonical published
/// name when referred to by name (a plain directory has none).
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The checkpoint directory (`config.json` + `model.safetensors`).
    pub dir: PathBuf,
    /// The canonical published name, when known.
    pub name: Option<&'static str>,
}

/// The subdirectory of the model directory that holds structify state.
fn feature_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("text").join("structify")
}

/// Resolves `model` to a checkpoint directory, downloading a named model into
/// `<models_dir>/text/structify/<name>/` on first use.
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
///
/// # Errors
///
/// Returns [`StructifyError::InvalidModel`] for an unknown name and
/// [`StructifyError::ModelDownload`] when fetching the checkpoint fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<ResolvedModel, StructifyError> {
    // A path to a local checkpoint directory wins over the name table.
    let as_path = Path::new(model);
    if as_path.join("config.json").is_file() {
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            name: None,
        });
    }

    let Some(&(name, weight)) =
        KNOWN_MODELS.iter().find(|(known, _)| *known == model)
    else {
        let known: Vec<&str> =
            KNOWN_MODELS.iter().map(|&(name, _)| name).collect();
        return Err(StructifyError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a checkpoint \
             directory)",
            known.join(", ")
        )));
    };

    // The repo slug equals the name; each model ships `config.json` plus its
    // one weight file (safetensors or a PyTorch checkpoint).
    let files = ["config.json", weight];
    let dir = feature_dir(models_dir).join(name);
    if files.iter().all(|file| dir.join(file).is_file()) {
        return Ok(ResolvedModel {
            dir,
            name: Some(name),
        });
    }

    fs::create_dir_all(&dir).map_err(|e| {
        StructifyError::ModelDownload(format!(
            "creating {}: {e}",
            dir.display()
        ))
    })?;

    let client = http_client()?;
    for file in files {
        let target = dir.join(file);
        if target.is_file() {
            continue;
        }
        let url = format!(
            "https://huggingface.co/segment-any-text/{name}/resolve/main/{file}"
        );
        download_file(&client, &url, &target, file, progress)?;
    }

    Ok(ResolvedModel {
        dir,
        name: Some(name),
    })
}

/// Resolves the shared XLM-R tokenizer, downloading it into
/// `<models_dir>/text/structify/tokenizer.json` on first use.
///
/// # Errors
///
/// Returns [`StructifyError::ModelDownload`] when fetching the tokenizer fails.
pub fn resolve_tokenizer(
    models_dir: &Path,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<PathBuf, StructifyError> {
    let dir = feature_dir(models_dir);
    let target = dir.join(TOKENIZER_FILE);
    if target.is_file() {
        return Ok(target);
    }
    fs::create_dir_all(&dir).map_err(|e| {
        StructifyError::ModelDownload(format!(
            "creating {}: {e}",
            dir.display()
        ))
    })?;
    let client = http_client()?;
    let url = format!(
        "https://huggingface.co/{TOKENIZER_REPO}/resolve/main/{TOKENIZER_FILE}"
    );
    download_file(&client, &url, &target, TOKENIZER_FILE, progress)?;
    Ok(target)
}

/// Builds the blocking HTTP client used for downloads.
fn http_client() -> Result<reqwest::blocking::Client, StructifyError> {
    reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // Model files are large; only the connection phase is bounded.
        .timeout(None)
        .build()
        .map_err(|e| {
            StructifyError::ModelDownload(format!("building http client: {e}"))
        })
}

/// Streams `url` into `target` via a temporary file, so an interrupted download
/// never leaves a half-written file behind.
fn download_file(
    client: &reqwest::blocking::Client,
    url: &str,
    target: &Path,
    label: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<(), StructifyError> {
    let failed = |stage: &str, detail: String| {
        StructifyError::ModelDownload(format!("{stage} {url}: {detail}"))
    };

    let mut response = client
        .get(url)
        .send()
        .map_err(|e| failed("requesting", e.to_string()))?;
    if !response.status().is_success() {
        return Err(failed(
            "requesting",
            format!("http status {}", response.status()),
        ));
    }
    let total = response.content_length();

    let temp = target.with_extension("partial");
    let mut output = fs::File::create(&temp)
        .map_err(|e| failed("writing", e.to_string()))?;

    let mut buffer = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|e| failed("reading", e.to_string()))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|e| failed("writing", e.to_string()))?;
        done += read as u64;
        progress(label, done, total);
    }
    output
        .flush()
        .map_err(|e| failed("writing", e.to_string()))?;
    drop(output);

    fs::rename(&temp, target)
        .map_err(|e| failed("finalizing", e.to_string()))?;
    Ok(())
}
