//! Model resolution, download, and the one-time conversion.
//!
//! The reference publishes a single 53.8 MB `torch.package` archive holding
//! both networks, their tensors, and four lookup tables. Keeping that format on
//! disk would mean re-walking a pickle tree on every run, so a download is
//! converted **once**: the tensors become safetensors, the tables become plain
//! text, and the archive is removed. This module is the only place that knows
//! the reference's save format at all.
//!
//! The archive is pinned to an upstream **tag**: on `master` the weights are
//! rewritten in place under a stable path, which would silently change results.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use candle_core::{Device, Tensor};

use super::{
    error::StressError,
    model::WEIGHTS_FILE,
    package::{self, Element, Package, StoredTensor},
    tables::{
        EXCEPTIONS_FILE, HOMOGRAPHS_FILE, NGRAMS_FILE, VOCAB_FILE,
        render_ngrams, render_vocab,
    },
};
use crate::download::{Download, Progress};

/// A published model: its cache name, where the archive comes from, and what
/// it should hash to.
struct ModelSpec {
    name: &'static str,
    url: &'static str,
    /// The file the archive lands in before it is converted.
    archive: &'static str,
    blake3: &'static str,
}

/// The models this feature can serve. One today — the Russian accentor of
/// `silero-stress`, MIT.
const KNOWN_MODELS: &[ModelSpec] = &[ModelSpec {
    name: "silero-ru",
    url: "https://raw.githubusercontent.com/snakers4/silero-stress/v1.4/src/\
          silero_stress/data/accentor.pt",
    archive: "accentor.pt",
    blake3: "228e7c5c26fee2035356ec0b2defed5640dbb657b61f4e6d4a15c64b3f2953eb",
}];

/// The published model names, for help text and error messages.
#[must_use]
pub fn known_model_names() -> Vec<&'static str> {
    KNOWN_MODELS.iter().map(|spec| spec.name).collect()
}

/// The default model.
pub const DEFAULT_MODEL: &str = "silero-ru";

/// A resolved model: the directory holding the converted files, and its
/// canonical published name when referred to by name.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The model directory.
    pub dir: PathBuf,
    /// The canonical published name, when known.
    pub name: Option<&'static str>,
}

impl ResolvedModel {
    /// The name to report in the output: the published one, or the directory
    /// that was given.
    #[must_use]
    pub fn label(&self) -> String {
        self.name
            .map(str::to_owned)
            .unwrap_or_else(|| self.dir.display().to_string())
    }
}

/// The subdirectory of the model directory that holds this feature's state.
fn feature_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("text").join("stress")
}

/// Every file a converted model directory must hold.
fn is_converted(dir: &Path) -> bool {
    [
        WEIGHTS_FILE,
        NGRAMS_FILE,
        EXCEPTIONS_FILE,
        HOMOGRAPHS_FILE,
        VOCAB_FILE,
    ]
    .iter()
    .all(|file| dir.join(file).is_file())
}

/// Resolves `model` to a converted model directory, downloading and converting
/// a named model into `<models_dir>/text/stress/<name>/` on first use.
///
/// `progress` is called as the download advances with
/// `(file name, bytes done, bytes total when known)`.
///
/// # Errors
///
/// Returns [`StressError::InvalidModel`] for an unknown name and
/// [`StressError::ModelDownload`] when fetching or converting fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    progress: Progress<'_>,
) -> Result<ResolvedModel, StressError> {
    // A local directory that already holds the converted files wins over the
    // name table.
    let as_path = Path::new(model);
    if is_converted(as_path) {
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            name: None,
        });
    }

    let Some(spec) = KNOWN_MODELS.iter().find(|spec| spec.name == model) else {
        return Err(StressError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a model directory)",
            known_model_names().join(", ")
        )));
    };

    let dir = feature_dir(models_dir).join(spec.name);
    if is_converted(&dir) {
        return Ok(ResolvedModel {
            dir,
            name: Some(spec.name),
        });
    }

    // The archive lands next to the files it becomes and is removed once the
    // conversion succeeds — a conversion that fails does not cost the download
    // a second time.
    let archive = dir.join(spec.archive);
    if !archive.is_file() {
        Download::new(spec.url, &archive)
            .label(spec.archive)
            .blake3(spec.blake3)
            .fetch(progress)?;
    }
    convert(&archive, &dir)?;
    let _ = std::fs::remove_file(&archive);

    Ok(ResolvedModel {
        dir,
        name: Some(spec.name),
    })
}

/// Wraps a conversion failure.
fn failed(detail: impl std::fmt::Display) -> StressError {
    StressError::ModelDownload(format!("converting the model: {detail}"))
}

/// Turns the published archive into the files every later run reads.
pub(super) fn convert(archive: &Path, dir: &Path) -> Result<(), StressError> {
    let mut package = Package::open(archive)?;

    // Which of the two TorchScript modules is which is decided by what they
    // hold, not by their position in the archive.
    let mut accentor = None;
    let mut homograph = None;
    for name in package.script_modules() {
        let object = package.pickle(&name)?;
        let tensors = package::tensors(&object);
        if tensors.contains_key("embedding.weight") {
            accentor = Some((object, tensors));
        } else if tensors.contains_key("bert.embeddings.word_embeddings.weight")
        {
            homograph = Some((object, tensors));
        }
    }
    let Some((accentor_object, accentor_tensors)) = accentor else {
        return Err(failed("no accentor network in the archive"));
    };
    let Some((_, homograph_tensors)) = homograph else {
        return Err(failed("no homograph network in the archive"));
    };

    let mut weights: HashMap<String, Tensor> = HashMap::new();
    copy_accentor(&mut package, &accentor_tensors, &mut weights)?;
    copy_homograph(&mut package, &homograph_tensors, &mut weights)?;

    // The tables: the n-grams live with the accentor network, the rest in the
    // object pickle next to it.
    let ngrams = package::field(
        package::path(&accentor_object, &["embedding"])
            .ok_or_else(|| failed("no embedding module"))?,
        "ngram_dict",
    )
    .and_then(package::string_int_map)
    .ok_or_else(|| failed("no n-gram table"))?;

    let tables = package.pickle("accentor_models/accentor")?;
    let exceptions = package::path(&tables, &["accentor", "exceptions"])
        .and_then(package::string_pair_map)
        .ok_or_else(|| failed("no exception table"))?;
    let homographs = package::path(&tables, &["homosolver", "homodict"])
        .and_then(package::string_list_map)
        .ok_or_else(|| failed("no homograph table"))?;
    let vocab = package::path(&tables, &["homosolver", "tokenizer", "vocab"])
        .and_then(package::string_int_map)
        .ok_or_else(|| failed("no word-piece vocabulary"))?;

    crate::download::create_dir(dir)?;
    write_atomically(dir, NGRAMS_FILE, &render_ngrams(&ngrams)?)?;
    write_atomically(dir, VOCAB_FILE, &render_vocab(&vocab)?)?;
    write_atomically(dir, EXCEPTIONS_FILE, &render_exceptions(&exceptions))?;
    write_atomically(dir, HOMOGRAPHS_FILE, &render_homographs(&homographs)?)?;

    let temp = dir.join(WEIGHTS_FILE).with_extension("partial");
    candle_core::safetensors::save(&weights, &temp)
        .map_err(|e| failed(format!("writing the weights: {e}")))?;
    std::fs::rename(&temp, dir.join(WEIGHTS_FILE))
        .map_err(|e| failed(format!("finalizing the weights: {e}")))?;
    Ok(())
}

/// Writes one table through a temporary file, so an interrupted write never
/// leaves a half-converted model behind.
fn write_atomically(
    dir: &Path,
    file: &str,
    contents: &str,
) -> Result<(), StressError> {
    let temp = dir.join(format!("{file}.partial"));
    std::fs::write(&temp, contents)
        .map_err(|e| failed(format!("writing {file}: {e}")))?;
    std::fs::rename(&temp, dir.join(file))
        .map_err(|e| failed(format!("finalizing {file}: {e}")))
}

/// `word stress-position yo-position` per line, `-1` when there is no `ё`.
fn render_exceptions(entries: &[(String, i64, i64)]) -> String {
    let mut out = String::new();
    for (word, stress, yo) in entries {
        out.push_str(&format!("{word} {stress} {yo}\n"));
    }
    out
}

/// `word first-variant second-variant` per line. The variants are ordered here,
/// once, because the reference sorts them on every use and the network's bit
/// picks between them by index.
fn render_homographs(
    entries: &[(String, Vec<String>)],
) -> Result<String, StressError> {
    let mut out = String::new();
    for (word, variants) in entries {
        let mut variants = variants.clone();
        variants.sort();
        let [first, second] = &variants[..] else {
            return Err(failed(format!(
                "homograph `{word}` has {} variants, expected 2",
                variants.len()
            )));
        };
        out.push_str(&format!("{word} {first} {second}\n"));
    }
    Ok(out)
}

/// Reads one stored tensor as an f32 candle tensor of the expected shape.
fn copy(
    package: &mut Package,
    tensors: &HashMap<String, StoredTensor>,
    key: &str,
    into: &mut HashMap<String, Tensor>,
    name: &str,
) -> Result<(), StressError> {
    let stored = tensors
        .get(key)
        .ok_or_else(|| failed(format!("{key}: not in the archive")))?;
    let values = package.values(stored)?;
    let tensor = Tensor::from_vec(values, stored.shape.clone(), &Device::Cpu)
        .map_err(|e| failed(format!("{key}: {e}")))?;
    into.insert(name.to_owned(), tensor);
    Ok(())
}

/// The accentor: its n-gram table (already dequantized in the archive) and the
/// two heads.
fn copy_accentor(
    package: &mut Package,
    tensors: &HashMap<String, StoredTensor>,
    into: &mut HashMap<String, Tensor>,
) -> Result<(), StressError> {
    copy(
        package,
        tensors,
        "embedding.weight",
        into,
        "accentor.embedding.weight",
    )?;
    for head in ["stress_clf", "yo_clf"] {
        for layer in [0, 2, 4, 6] {
            for part in ["weight", "bias"] {
                let key = format!("{head}.{layer}.{part}");
                copy(package, tensors, &key, into, &format!("accentor.{key}"))?;
            }
        }
    }
    Ok(())
}

/// The homograph solver: the encoder, its quantized word embeddings (kept
/// byte-sized — widening them to f32 would quadruple the model on disk), and
/// the binary head. The pooler is published but unused, so it is not copied.
fn copy_homograph(
    package: &mut Package,
    tensors: &HashMap<String, StoredTensor>,
    into: &mut HashMap<String, Tensor>,
) -> Result<(), StressError> {
    let key = "bert.embeddings.word_embeddings.weight";
    let stored = tensors
        .get(key)
        .ok_or_else(|| failed(format!("{key}: not in the archive")))?;
    if stored.element != Element::I8 {
        return Err(failed(format!(
            "{key}: expected the quantized (byte) table, got {:?}",
            stored.element
        )));
    }
    // Stored as signed bytes; biased into unsigned so safetensors can hold
    // them, and unbiased again at load.
    let biased: Vec<u8> = package
        .raw(stored)?
        .into_iter()
        .map(|byte| byte ^ 0x80)
        .collect();
    let table = Tensor::from_vec(biased, stored.shape.clone(), &Device::Cpu)
        .map_err(|e| failed(format!("{key}: {e}")))?;
    into.insert("homograph.word_embeddings.q".to_owned(), table);

    for (key, name) in [
        ("bert.scale", "homograph.word_embeddings.scale"),
        ("bert.zero_point", "homograph.word_embeddings.zero_point"),
    ] {
        let value = package::scalar(package, tensors, key)?;
        let tensor = Tensor::from_vec(vec![value], 1, &Device::Cpu)
            .map_err(|e| failed(format!("{key}: {e}")))?;
        into.insert(name.to_owned(), tensor);
    }

    let embeddings = "bert.embeddings";
    for tail in [
        "position_embeddings.weight",
        "token_type_embeddings.weight",
        "LayerNorm.weight",
        "LayerNorm.bias",
    ] {
        copy(
            package,
            tensors,
            &format!("{embeddings}.{tail}"),
            into,
            &format!("homograph.embeddings.{tail}"),
        )?;
    }

    // Layers are copied until the archive runs out of them, and the loader
    // checks the count against the declared geometry.
    let mut layer = 0;
    loop {
        let prefix = format!("bert.encoder.layer.{layer}");
        if !tensors
            .contains_key(&format!("{prefix}.attention.self.query.weight"))
        {
            break;
        }
        for tail in [
            "attention.self.query.weight",
            "attention.self.query.bias",
            "attention.self.key.weight",
            "attention.self.key.bias",
            "attention.self.value.weight",
            "attention.self.value.bias",
            "attention.output.dense.weight",
            "attention.output.dense.bias",
            "attention.output.LayerNorm.weight",
            "attention.output.LayerNorm.bias",
            "intermediate.dense.weight",
            "intermediate.dense.bias",
            "output.dense.weight",
            "output.dense.bias",
            "output.LayerNorm.weight",
            "output.LayerNorm.bias",
        ] {
            copy(
                package,
                tensors,
                &format!("{prefix}.{tail}"),
                into,
                &format!("homograph.layer.{layer}.{tail}"),
            )?;
        }
        layer += 1;
    }
    if layer == 0 {
        return Err(failed("no encoder layers in the archive"));
    }

    for (source, name) in [("0", "0"), ("3", "1")] {
        for part in ["weight", "bias"] {
            copy(
                package,
                tensors,
                &format!("homo_clf.{source}.{part}"),
                into,
                &format!("homograph.head.{name}.{part}"),
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
