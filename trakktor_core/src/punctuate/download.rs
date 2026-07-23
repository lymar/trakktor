//! Model resolution and download.
//!
//! A model is referred to either by a published name or by a path to a local
//! directory that already holds the extracted files. Named models are cached
//! under the model directory as `text/punctuate/<name>/`, downloaded on first
//! use: the SentencePiece model (`sp.model`) is fetched directly, and the
//! PyTorch weights (`model_weights.ckpt`) are extracted from the NeMo archive
//! (`*.nemo`, a tar), which is then removed.

use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use super::{error::PunctuateError, runtime::WEIGHTS_FILE};

/// The SentencePiece model file name, in the repo and in the cache.
pub(super) const SPE_FILE: &str = "sp.model";

/// A published model: its cache name, HF repo, and the two files fetched from
/// the repo root — the SentencePiece model and the NeMo archive to extract the
/// weights from.
struct ModelSpec {
    name: &'static str,
    repo: &'static str,
    nemo: &'static str,
}

/// The known models. For now, the multilingual XLM-R punctuator (47 languages,
/// Apache-2.0).
const KNOWN_MODELS: &[ModelSpec] = &[ModelSpec {
    name: "xlmr-47lang",
    repo: "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
    nemo: "pcs47_1jun.nemo",
}];

/// The published model names, for help text and error messages.
#[must_use]
pub fn known_model_names() -> Vec<&'static str> {
    KNOWN_MODELS.iter().map(|spec| spec.name).collect()
}

/// A resolved model: the directory holding `sp.model` and `model_weights.ckpt`,
/// and its canonical published name when referred to by name.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The model directory.
    pub dir: PathBuf,
    /// The canonical published name, when known.
    pub name: Option<&'static str>,
}

impl ResolvedModel {
    /// The `sp.model` path.
    #[must_use]
    pub fn spe_path(&self) -> PathBuf { self.dir.join(SPE_FILE) }
}

/// The subdirectory of the model directory that holds punctuate state.
fn feature_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("text").join("punctuate")
}

/// Resolves `model` to a directory holding `sp.model` and
/// `model_weights.ckpt`, downloading a named model into
/// `<models_dir>/text/punctuate/<name>/` on first use.
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
///
/// # Errors
///
/// Returns [`PunctuateError::InvalidModel`] for an unknown name and
/// [`PunctuateError::ModelDownload`] when fetching or extracting fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<ResolvedModel, PunctuateError> {
    // A local directory that already holds the weights wins over the name
    // table.
    let as_path = Path::new(model);
    if as_path.join(WEIGHTS_FILE).is_file() {
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            name: None,
        });
    }

    let Some(spec) = KNOWN_MODELS.iter().find(|spec| spec.name == model) else {
        return Err(PunctuateError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a model directory)",
            known_model_names().join(", ")
        )));
    };

    let dir = feature_dir(models_dir).join(spec.name);
    let spe = dir.join(SPE_FILE);
    let weights = dir.join(WEIGHTS_FILE);
    if spe.is_file() && weights.is_file() {
        return Ok(ResolvedModel {
            dir,
            name: Some(spec.name),
        });
    }

    fs::create_dir_all(&dir).map_err(|e| {
        PunctuateError::ModelDownload(format!(
            "creating {}: {e}",
            dir.display()
        ))
    })?;

    let client = http_client()?;
    if !spe.is_file() {
        let url = repo_url(spec.repo, SPE_FILE);
        download_file(&client, &url, &spe, SPE_FILE, progress)?;
    }
    if !weights.is_file() {
        // Fetch the NeMo archive, extract just the weights, then drop it.
        let nemo = dir.join(spec.nemo);
        let url = repo_url(spec.repo, spec.nemo);
        download_file(&client, &url, &nemo, spec.nemo, progress)?;
        extract_weights(&nemo, &weights)?;
        let _ = fs::remove_file(&nemo);
    }

    Ok(ResolvedModel {
        dir,
        name: Some(spec.name),
    })
}

/// The HF resolve URL for a repo file.
fn repo_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

/// Extracts `model_weights.ckpt` from a NeMo archive (`*.nemo`, a plain or gzip
/// tar) into `target`.
fn extract_weights(nemo: &Path, target: &Path) -> Result<(), PunctuateError> {
    let failed = |detail: String| {
        PunctuateError::ModelDownload(format!(
            "extracting {WEIGHTS_FILE} from {}: {detail}",
            nemo.display()
        ))
    };
    // NeMo archives are usually a plain tar; support gzip too (magic 1f 8b),
    // probed on the same handle and rewound.
    let mut file = fs::File::open(nemo).map_err(|e| failed(e.to_string()))?;
    let mut magic = [0u8; 2];
    let gzip = file.read_exact(&mut magic).is_ok() && magic == [0x1f, 0x8b];
    file.seek(SeekFrom::Start(0))
        .map_err(|e| failed(e.to_string()))?;
    let reader: Box<dyn Read> = if gzip {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut archive = tar::Archive::new(reader);
    let entries = archive.entries().map_err(|e| failed(e.to_string()))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| failed(e.to_string()))?;
        let is_weights = entry
            .path()
            .ok()
            .and_then(|p| {
                p.file_name().map(|n| n.to_string_lossy().into_owned())
            })
            .is_some_and(|name| name == WEIGHTS_FILE);
        if is_weights {
            let temp = target.with_extension("partial");
            let mut out =
                fs::File::create(&temp).map_err(|e| failed(e.to_string()))?;
            std::io::copy(&mut entry, &mut out)
                .map_err(|e| failed(e.to_string()))?;
            out.flush().map_err(|e| failed(e.to_string()))?;
            drop(out);
            fs::rename(&temp, target).map_err(|e| failed(e.to_string()))?;
            return Ok(());
        }
    }
    Err(failed(format!("no {WEIGHTS_FILE} member")))
}

/// Builds the blocking HTTP client used for downloads.
fn http_client() -> Result<reqwest::blocking::Client, PunctuateError> {
    reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // Model files are large; only the connection phase is bounded.
        .timeout(None)
        .build()
        .map_err(|e| {
            PunctuateError::ModelDownload(format!("building http client: {e}"))
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
) -> Result<(), PunctuateError> {
    let failed = |stage: &str, detail: String| {
        PunctuateError::ModelDownload(format!("{stage} {url}: {detail}"))
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
