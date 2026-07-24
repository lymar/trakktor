//! Model resolution and download.
//!
//! A model is referred to either by a published variant name or by a path to a
//! local checkpoint directory. Named variants are cached under the model
//! directory as `tts/qwen3-tts/<variant>/`, fetched on first use.
//!
//! Each checkpoint carries its own copy of the codec under `speech_tokenizer/`,
//! so there is nothing else to fetch alongside it.

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use super::{config::ModelType, error::Qwen3TtsError};

/// A published checkpoint: the name `--model` accepts, the repository it comes
/// from, and the conditioning path it was trained for.
#[derive(Debug, Clone, Copy)]
pub struct KnownModel {
    /// The name `--model` accepts.
    pub name: &'static str,
    /// The Hugging Face repository holding the checkpoint.
    pub repo: &'static str,
    /// How the voice is chosen for this checkpoint.
    pub model_type: ModelType,
}

/// The checkpoints this engine can serve.
///
/// The family also publishes VoiceDesign and Base checkpoints; they condition
/// the voice in ways this engine does not implement yet, so they are not
/// offered — listing them would only let a caller spend gigabytes on a
/// checkpoint that cannot then be used.
pub const KNOWN_MODELS: &[KnownModel] = &[
    KnownModel {
        name: "0.6b-customvoice",
        repo: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        model_type: ModelType::CustomVoice,
    },
    KnownModel {
        name: "1.7b-customvoice",
        repo: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        model_type: ModelType::CustomVoice,
    },
];

/// Files a checkpoint directory must hold for the engine to load it.
///
/// The text tokenizer ships as a raw byte-level BPE (`vocab.json` plus
/// `merges.txt`) rather than a packaged `tokenizer.json`, so both come along,
/// with `tokenizer_config.json` for the control tokens.
pub const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
];

/// A resolved checkpoint: where it lives and, when referred to by name, which
/// published variant it is.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The checkpoint directory.
    pub dir: PathBuf,
    /// The published variant, when the model was named rather than pathed.
    pub known: Option<KnownModel>,
}

impl ResolvedModel {
    /// The name to report in the output: the published variant, or the
    /// directory that was given.
    #[must_use]
    pub fn label(&self) -> String {
        self.known
            .map(|model| model.name.to_owned())
            .unwrap_or_else(|| self.dir.display().to_string())
    }
}

/// The subdirectory of the model directory that holds this engine's state.
fn engine_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("tts").join("qwen3-tts")
}

/// Resolves `model` to a checkpoint directory, downloading a named variant into
/// `<models_dir>/tts/qwen3-tts/<variant>/` on first use.
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
///
/// # Errors
///
/// Returns [`Qwen3TtsError::InvalidModel`] for an unknown name and
/// [`Qwen3TtsError::ModelDownload`] when fetching the checkpoint fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<ResolvedModel, Qwen3TtsError> {
    // A path to a local checkpoint directory wins over the name table.
    let as_path = Path::new(model);
    if as_path.join("config.json").is_file() {
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            known: None,
        });
    }

    let Some(&known) = KNOWN_MODELS.iter().find(|entry| entry.name == model)
    else {
        let names: Vec<&str> =
            KNOWN_MODELS.iter().map(|entry| entry.name).collect();
        return Err(Qwen3TtsError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a checkpoint \
             directory)",
            names.join(", ")
        )));
    };

    let dir = engine_dir(models_dir).join(known.name);
    if REQUIRED_FILES.iter().all(|file| dir.join(file).is_file()) {
        return Ok(ResolvedModel {
            dir,
            known: Some(known),
        });
    }

    let client = http_client()?;
    for file in REQUIRED_FILES {
        let target = dir.join(file);
        if target.is_file() {
            continue;
        }
        // `speech_tokenizer/…` nests, so create the parent of each file rather
        // than the checkpoint root alone.
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                Qwen3TtsError::ModelDownload(format!(
                    "creating {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{file}",
            known.repo
        );
        download_file(&client, &url, &target, file, progress)?;
    }

    Ok(ResolvedModel {
        dir,
        known: Some(known),
    })
}

/// Builds the blocking HTTP client used for downloads.
fn http_client() -> Result<reqwest::blocking::Client, Qwen3TtsError> {
    reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        // Model files are large; only the connection phase is bounded.
        .timeout(None)
        .build()
        .map_err(|e| {
            Qwen3TtsError::ModelDownload(format!("building http client: {e}"))
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
) -> Result<(), Qwen3TtsError> {
    let failed = |stage: &str, detail: String| {
        Qwen3TtsError::ModelDownload(format!("{stage} {url}: {detail}"))
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

#[cfg(test)]
mod tests;
