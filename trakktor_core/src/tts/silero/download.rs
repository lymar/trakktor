//! Model resolution, download, and the one-time conversion.
//!
//! Upstream publishes each model as a `torch.package`: one ZIP holding its
//! Python frontend, an object pickle with the lookup tables, a TorchScript
//! module, and the tensors as flat storages. Keeping that on disk would mean
//! walking a pickle tree on every run, so a download is converted **once** —
//! the tensors become safetensors, the tables become one JSON file, and the
//! archive is removed. This module is the only place that knows the published
//! save format at all.
//!
//! Two things the conversion deliberately drops, because nothing downstream can
//! reach them: the accentor bundled with the Russian model (54 MB — the same
//! network `text stress` already runs), and the tensors the published
//! `forward` never touches, the dead half of the decoder's shortened stage
//! among them.
//!
//! Three properties of the download channel shape the pins below: it serves no
//! ranges (so an interrupted download restarts), it publishes no checksums (so
//! the BLAKE3 digests here are ours), and it **rewrites weights under stable
//! URLs** (so a digest mismatch is a model that changed, not a corrupt file).

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use candle_core::{Device, Tensor, pickle::Object};

use super::{
    config::{FRAME_SECONDS, SAMPLE_RATE},
    error::SileroError,
    tables::Tables,
};
use crate::{
    download::{Download, Progress},
    torch_package::{self as package, Package, StoredTensor},
};

/// The converted weights inside a model directory.
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// How a model may be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum License {
    /// The permissive pair: usable without a further decision.
    Mit,
    /// Non-commercial, share-alike. Available only behind an explicit flag.
    NonCommercial,
}

impl License {
    /// The identifier reported in help, in the output contract, and in error
    /// messages.
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            Self::Mit => "MIT",
            Self::NonCommercial => "CC BY-NC-SA 4.0",
        }
    }

    /// Whether the model is available without `--allow-non-commercial-models`.
    #[must_use]
    pub fn is_permissive(self) -> bool { matches!(self, Self::Mit) }
}

/// A published model: its name here, where it comes from, what it should hash
/// to, and how it may be used.
#[derive(Debug, Clone, Copy)]
pub struct KnownModel {
    /// The name `--model` takes.
    pub name: &'static str,
    /// The file it lands in before conversion, which is also upstream's own
    /// name for it.
    pub file: &'static str,
    url: &'static str,
    blake3: &'static str,
    /// How the model may be used.
    pub license: License,
    /// One line for `--help`.
    pub summary: &'static str,
}

/// The models this engine can serve.
///
/// Deliberately not every published one: the earlier Russian releases are
/// superseded by the one below, and the checkpoints for other model families
/// are not this architecture.
pub const KNOWN_MODELS: &[KnownModel] = &[
    KnownModel {
        name: "cis-base",
        file: "v5_cis_base.pt",
        url: "https://models.silero.ai/models/tts/ru/v5_cis_base.pt",
        blake3:
            "109c4db258a8f796c740f43d63842c08bdd30af6727dc5b3412ce1e4abc4319d",
        license: License::Mit,
        summary: "60 voices across 20 languages, 29 of them Russian; expects \
                  stress marks in every language",
    },
    KnownModel {
        name: "cis-base-nostress",
        file: "v5_cis_base_nostress.pt",
        url: "https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt",
        blake3:
            "ffdc35d09078a9a4f2bf4e28723f99cf5e5ceb5ce6c342efbf20212100548161",
        license: License::Mit,
        summary: "the same 60 voices, trained to need stress marks only for \
                  the Slavic languages",
    },
    KnownModel {
        name: "cis-ext",
        file: "v5_cis_ext.pt",
        url: "https://models.silero.ai/models/tts/ru/v5_cis_ext.pt",
        blake3:
            "c93433b755b2c08d41b1cb982b69fc361b154c26abe828dd26db186b44f9d5c8",
        license: License::NonCommercial,
        summary: "35 further voices for Chuvash, Kalmyk, Kazakh, Tatar, \
                  Ukrainian and Uzbek",
    },
    KnownModel {
        name: "ru-classic",
        file: "v5_5_ru.pt",
        url: "https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
        blake3:
            "363bee213628c0eed5fb67caf8dee3769db360ada76e35cf99d6667ada446cd7",
        license: License::NonCommercial,
        summary: "the five long-standing Russian voices (aidar, baya, \
                  kseniya, eugene, xenia), with question intonation",
    },
];

/// The default model: the permissive one that reads every language it knows.
pub const DEFAULT_MODEL: &str = "cis-base";

/// The models available without the non-commercial flag, comma-separated.
#[must_use]
pub fn permissive_model_names() -> String {
    KNOWN_MODELS
        .iter()
        .filter(|model| model.license.is_permissive())
        .map(|model| model.name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// A resolved model: where its converted files are, and what it is.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The directory holding the converted files.
    pub dir: PathBuf,
    /// The published model, when it was referred to by name.
    pub known: Option<KnownModel>,
}

impl ResolvedModel {
    /// The name to report in the output: the published one, or the directory
    /// that was given.
    #[must_use]
    pub fn label(&self) -> String {
        self.known
            .map(|model| model.name.to_owned())
            .unwrap_or_else(|| self.dir.display().to_string())
    }

    /// The licence to report, when it is known.
    #[must_use]
    pub fn license(&self) -> Option<&'static str> {
        self.known.map(|model| model.license.id())
    }
}

/// The subdirectory of the model directory that holds this engine's state.
fn engine_dir(models_dir: &Path) -> PathBuf {
    models_dir.join("tts").join("silero")
}

/// Whether a directory already holds a converted model.
fn is_converted(dir: &Path) -> bool {
    dir.join(WEIGHTS_FILE).is_file() &&
        dir.join(super::tables::TABLES_FILE).is_file()
}

/// Resolves `model` to a converted model directory, downloading and converting
/// a named model into `<models_dir>/tts/silero/<name>/` on first use.
///
/// A model whose licence is not permissive needs `allow_non_commercial`; the
/// check happens here, before anything is fetched.
///
/// # Errors
///
/// Returns [`SileroError::LicenseRestricted`] for a gated model without the
/// flag, [`SileroError::InvalidModel`] for an unknown name, and
/// [`SileroError::ModelDownload`] when fetching or converting fails.
pub fn resolve_model(
    models_dir: &Path,
    model: &str,
    allow_non_commercial: bool,
    progress: Progress<'_>,
) -> Result<ResolvedModel, SileroError> {
    // A local directory that already holds the converted files wins over the
    // name table; its licence is then the caller's business, not ours.
    let as_path = Path::new(model);
    if is_converted(as_path) {
        return Ok(ResolvedModel {
            dir: as_path.to_path_buf(),
            known: None,
        });
    }

    let Some(known) = KNOWN_MODELS.iter().find(|known| known.name == model)
    else {
        return Err(SileroError::InvalidModel(format!(
            "unknown model `{model}` (known: {}; or pass a model directory)",
            KNOWN_MODELS
                .iter()
                .map(|known| known.name)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    };
    if !known.license.is_permissive() && !allow_non_commercial {
        return Err(SileroError::LicenseRestricted {
            model: known.name.to_owned(),
            license: known.license.id(),
            allowed: permissive_model_names(),
        });
    }

    let dir = engine_dir(models_dir).join(known.name);
    if is_converted(&dir) {
        return Ok(ResolvedModel {
            dir,
            known: Some(*known),
        });
    }

    // The archive lands next to the files it becomes and is removed once the
    // conversion succeeds — a conversion that fails does not cost the download
    // a second time.
    let archive = dir.join(known.file);
    if !archive.is_file() {
        Download::new(known.url, &archive)
            .label(known.file)
            .blake3(known.blake3)
            .fetch(progress)?;
    }
    convert(&archive, &dir)?;
    let _ = std::fs::remove_file(&archive);

    Ok(ResolvedModel {
        dir,
        known: Some(*known),
    })
}

/// Wraps a conversion failure.
fn failed(detail: impl std::fmt::Display) -> SileroError {
    SileroError::ModelDownload(format!("converting the model: {detail}"))
}

/// The switches of the reference this port hardcodes rather than reads.
///
/// A model whose module says otherwise would convert cleanly and then be read
/// **wrong** — a pitch scale silently applied twice, durations that are not
/// logarithms, an inverse transform with the other padding. Upstream rewrites
/// weights under stable URLs, so this is a real future, not paranoia; the
/// mismatch fails here, loudly, at install.
fn verify_assumptions(module: &Object) -> Result<(), SileroError> {
    let number =
        |names: &[&str]| package::path(module, names).and_then(package::number);
    let text =
        |names: &[&str]| package::path(module, names).and_then(package::string);
    let flag = |names: &[&str]| {
        package::path(module, names).and_then(package::boolean)
    };
    let pin = |what: &str, held: bool| {
        if held {
            Ok(())
        } else {
            Err(failed(format!(
                "the model breaks an assumption this port hardcodes: {what}; \
                 the published architecture has changed"
            )))
        }
    };

    pin(
        "the pitch projection is added unscaled (pitch_strength = 1)",
        number(&["tacotron", "pitch_strength"]) == Some(1.0),
    )?;
    pin(
        "the pitch is added before the length regulator (pitch_emb_type = \
         \"default\")",
        text(&["tacotron", "pitch_emb_type"]) == Some("default"),
    )?;
    pin(
        "the duration head predicts log(1 + frames) (log_dur = true)",
        flag(&["tacotron", "log_dur"]) == Some(true),
    )?;
    pin(
        "the inverse transform uses \"same\" padding",
        text(&["vocoder", "head", "istft", "padding"]) == Some("same"),
    )?;
    let hop = (FRAME_SECONDS * f64::from(SAMPLE_RATE)).round();
    pin(
        "one mel frame is 12.5 ms at 48 kHz (hop_length = 600)",
        number(&["vocoder", "head", "istft", "hop_length"]) == Some(hop),
    )?;
    pin(
        "the analysis window spans the whole transform (win_length = n_fft)",
        match (
            number(&["vocoder", "head", "istft", "win_length"]),
            number(&["vocoder", "head", "istft", "n_fft"]),
        ) {
            (Some(win), Some(n_fft)) => win == n_fft,
            _ => false,
        },
    )?;
    Ok(())
}

/// Turns a published archive into the files every later run reads.
pub(super) fn convert(archive: &Path, dir: &Path) -> Result<(), SileroError> {
    let mut archive = Package::open(archive)?;
    let tables = read_tables(&mut archive)?;

    // Which scripted module is the synthesizer is decided by what it holds,
    // not by its position: the Russian model packs its accentor alongside, and
    // that one is not ported — `text stress` is the same network.
    let mut synthesizer = None;
    for name in archive.script_modules() {
        let module = archive.pickle(&name)?;
        let stored = package::tensors(&module);
        if stored.contains_key("tacotron.embedding.weight") {
            synthesizer = Some((module, stored));
            break;
        }
    }
    let Some((module, stored)) = synthesizer else {
        return Err(failed("no acoustic model in the archive"));
    };
    verify_assumptions(&module)?;
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    copy_acoustic(&mut archive, &stored, &mut weights)?;
    copy_predictors(&mut archive, &stored, &mut weights)?;
    copy_vocoder(&mut archive, &stored, &mut weights)?;

    // The per-speaker pitch range is an attribute of the module, not a tensor.
    let ranges = package::field(&module, "mean_std_coef")
        .and_then(package::list)
        .ok_or_else(|| failed("no per-speaker pitch range"))?
        .iter()
        .map(|value| package::number(value).map(|value| value as f32))
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| failed("the pitch ranges are not numbers"))?;
    weights.insert(
        "pitch.mean_std_coef".to_owned(),
        Tensor::from_vec(ranges.clone(), ranges.len(), &Device::Cpu)
            .map_err(|e| failed(format!("the pitch ranges: {e}")))?,
    );

    crate::download::create_dir(dir)?;
    tables.store(dir)?;
    let temp = dir.join(WEIGHTS_FILE).with_extension("partial");
    candle_core::safetensors::save(&weights, &temp)
        .map_err(|e| failed(format!("writing the weights: {e}")))?;
    std::fs::rename(&temp, dir.join(WEIGHTS_FILE))
        .map_err(|e| failed(format!("finalizing the weights: {e}")))?;
    Ok(())
}

/// Reads the alphabet, the speakers and the transliteration tables out of the
/// archive's object pickle.
fn read_tables(archive: &mut Package) -> Result<Tables, SileroError> {
    let model = archive.pickle("tts_models/model")?;
    let packages = package::path(&model, &["packages"])
        .and_then(package::list)
        .ok_or_else(|| failed("no packages in the archive"))?;
    let [part] = packages else {
        return Err(failed(format!(
            "expected one packaged model, found {}",
            packages.len()
        )));
    };

    let symbols = ordered(
        package::field(part, "symbol_to_id")
            .and_then(package::string_int_map)
            .ok_or_else(|| failed("no symbol table"))?,
        "symbol",
    )?;
    let speaker_tables = package::field(part, "speaker_to_ids")
        .and_then(package::list)
        .ok_or_else(|| failed("no speaker table"))?;
    let [speakers] = speaker_tables else {
        return Err(failed(format!(
            "expected one speaker table, found {}",
            speaker_tables.len()
        )));
    };
    let speakers = ordered(
        package::string_int_map(speakers)
            .ok_or_else(|| failed("the speaker table is not a mapping"))?,
        "speaker",
    )?;

    let letters = package::field(part, "alphabet")
        .and_then(package::list)
        .map(|letters| {
            letters
                .iter()
                .filter_map(package::string)
                .collect::<String>()
        })
        .unwrap_or_default();
    let translit = package::field(part, "ext_alph")
        .and_then(package::nested_string_map)
        .map(|tables| {
            tables
                .into_iter()
                .map(|(language, table)| {
                    (language, table.into_iter().collect())
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(Tables {
        symbols,
        alphabet: package::field(part, "symbols")
            .and_then(package::string)
            .ok_or_else(|| failed("no alphabet string"))?
            .to_owned(),
        letters,
        sos: package::field(part, "sos_token")
            .and_then(package::string)
            .unwrap_or("|")
            .to_owned(),
        eos: package::field(part, "eos_token")
            .and_then(package::string)
            .unwrap_or("~")
            .to_owned(),
        speakers,
        translit,
    })
}

/// Turns a `name → id` mapping into a list indexed by id.
///
/// The ids are **not** always `0..n`: one published model names thirty-five
/// speakers over thirty-six rows, leaving a hole where a row belongs to
/// something it does not name. A hole becomes an empty name, which nothing can
/// then ask for by accident.
fn ordered(
    entries: Vec<(String, i64)>,
    what: &str,
) -> Result<Vec<String>, SileroError> {
    let highest = entries
        .iter()
        .map(|(_, id)| *id)
        .max()
        .ok_or_else(|| failed(format!("the {what} table is empty")))?;
    let size = usize::try_from(highest)
        .map_err(|_| failed(format!("a {what} has a negative id")))? +
        1;
    let mut out = vec![String::new(); size];
    for (name, id) in entries {
        let slot = usize::try_from(id)
            .ok()
            .and_then(|id| out.get_mut(id))
            .ok_or_else(|| {
                failed(format!("{what} `{name}` has an id outside the table"))
            })?;
        *slot = name;
    }
    Ok(out)
}

/// Reads one stored tensor as an f32 candle tensor, reshaped if asked.
fn copy(
    archive: &mut Package,
    stored: &HashMap<String, StoredTensor>,
    key: &str,
    into: &mut HashMap<String, Tensor>,
    name: &str,
    shape: Option<Vec<usize>>,
) -> Result<(), SileroError> {
    let tensor = stored
        .get(key)
        .ok_or_else(|| failed(format!("{key}: not in the archive")))?;
    let values = archive.values(tensor)?;
    let shape = shape.unwrap_or_else(|| tensor.shape.clone());
    let tensor = Tensor::from_vec(values, shape, &Device::Cpu)
        .map_err(|e| failed(format!("{key}: {e}")))?;
    into.insert(name.to_owned(), tensor);
    Ok(())
}

/// Copies one FFT block, folding away the pointwise convolution's spare axis.
fn copy_block(
    archive: &mut Package,
    stored: &HashMap<String, StoredTensor>,
    from: &str,
    to: &str,
    into: &mut HashMap<String, Tensor>,
) -> Result<(), SileroError> {
    for tail in [
        "self_attn.in_proj_weight",
        "self_attn.in_proj_bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "conv1.weight",
        "conv1.bias",
        "conv2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    ] {
        copy(
            archive,
            stored,
            &format!("{from}.{tail}"),
            into,
            &format!("{to}.{tail}"),
            None,
        )?;
    }
    // A convolution with a kernel of one is a matrix; stored as one, it needs
    // no rearranging at load and no transposes at run time.
    let key = format!("{from}.conv2.weight");
    let shape = stored
        .get(&key)
        .map(|tensor| tensor.shape[..2].to_vec())
        .ok_or_else(|| failed(format!("{key}: not in the archive")))?;
    copy(
        archive,
        stored,
        &key,
        into,
        &format!("{to}.conv2.weight"),
        Some(shape),
    )
}

/// Copies a positional table, dropping the singleton batch axis it is stored
/// with.
fn copy_positional(
    archive: &mut Package,
    stored: &HashMap<String, StoredTensor>,
    from: &str,
    to: &str,
    into: &mut HashMap<String, Tensor>,
) -> Result<(), SileroError> {
    let key = format!("{from}.pe");
    let shape = stored
        .get(&key)
        .map(|tensor| vec![tensor.shape[0], tensor.shape[2]])
        .ok_or_else(|| failed(format!("{key}: not in the archive")))?;
    copy(
        archive,
        stored,
        &key,
        into,
        &format!("{to}.pe"),
        Some(shape),
    )?;
    copy(
        archive,
        stored,
        &format!("{from}.scale"),
        into,
        &format!("{to}.scale"),
        None,
    )
}

/// The acoustic model: the encoder, the pitch projection and the hourglass
/// decoder.
fn copy_acoustic(
    archive: &mut Package,
    stored: &HashMap<String, StoredTensor>,
    into: &mut HashMap<String, Tensor>,
) -> Result<(), SileroError> {
    for tail in [
        "embedding.weight",
        "speaker_embedding.weight",
        "pitch_proj.weight",
        "pitch_proj.bias",
        "encoder.norm.weight",
        "encoder.norm.bias",
        "decoder.norm.weight",
        "decoder.norm.bias",
        "decoder.upsample.proj.weight",
        "decoder.upsample.proj.bias",
        "lin.weight",
        "lin.bias",
    ] {
        let key = format!("tacotron.{tail}");
        copy(archive, stored, &key, into, &key, None)?;
    }
    for group in ["encoder", "decoder"] {
        copy_positional(
            archive,
            stored,
            &format!("tacotron.{group}.pos_encoder"),
            &format!("tacotron.{group}.pos_encoder"),
            into,
        )?;
    }
    let mut layer = 0;
    while stored
        .contains_key(&format!("tacotron.encoder.layers.{layer}.conv1.weight"))
    {
        let key = format!("tacotron.encoder.layers.{layer}");
        copy_block(archive, stored, &key, &key, into)?;
        layer += 1;
    }
    if layer == 0 {
        return Err(failed("no encoder blocks in the archive"));
    }
    // Of the decoder's two shortened blocks only the second is ever used: the
    // published forward applies both to the same input and keeps that one. The
    // first is not copied, so nothing later can accidentally run it.
    for block in [
        "decoder.pre_vanilla_layers.0",
        "decoder.shorten_layers.1",
        "decoder.post_vanilla_layers.0",
    ] {
        let key = format!("tacotron.{block}");
        copy_block(archive, stored, &key, &key, into)?;
    }
    Ok(())
}

/// The duration and pitch heads.
fn copy_predictors(
    archive: &mut Package,
    stored: &HashMap<String, StoredTensor>,
    into: &mut HashMap<String, Tensor>,
) -> Result<(), SileroError> {
    for head in ["dur_predictor.dur_pred", "pitch_predictor.pitch_pred"] {
        for tail in [
            "embedding.weight",
            "speaker_embedding.weight",
            "lin.weight",
            "lin.bias",
            "transformer.norm.weight",
            "transformer.norm.bias",
        ] {
            let key = format!("{head}.{tail}");
            copy(archive, stored, &key, into, &key, None)?;
        }
        copy_positional(
            archive,
            stored,
            &format!("{head}.transformer.pos_encoder"),
            &format!("{head}.transformer.pos_encoder"),
            into,
        )?;
        // The intonation table exists on one model only.
        let types = format!("{head}.type_embedding.weight");
        if stored.contains_key(&types) {
            copy(archive, stored, &types, into, &types, None)?;
        }
        let mut layer = 0;
        while stored.contains_key(&format!(
            "{head}.transformer.layers.{layer}.conv1.weight"
        )) {
            let key = format!("{head}.transformer.layers.{layer}");
            copy_block(archive, stored, &key, &key, into)?;
            layer += 1;
        }
        if layer == 0 {
            return Err(failed(format!("no blocks under {head}")));
        }
    }
    Ok(())
}

/// The vocoder, its analysis window, and the filterbanks the lower rates come
/// from.
fn copy_vocoder(
    archive: &mut Package,
    stored: &HashMap<String, StoredTensor>,
    into: &mut HashMap<String, Tensor>,
) -> Result<(), SileroError> {
    for tail in [
        "backbone.embed.weight",
        "backbone.embed.bias",
        "backbone.norm.weight",
        "backbone.norm.bias",
        "backbone.final_layer_norm.weight",
        "backbone.final_layer_norm.bias",
        "head.out.weight",
        "head.out.bias",
        "head.istft.window",
    ] {
        let key = format!("vocoder.{tail}");
        copy(archive, stored, &key, into, &key, None)?;
    }
    let mut block = 0;
    while stored
        .contains_key(&format!("vocoder.backbone.convnext.{block}.gamma"))
    {
        for tail in [
            "dwconv.weight",
            "dwconv.bias",
            "norm.weight",
            "norm.bias",
            "pwconv1.weight",
            "pwconv1.bias",
            "pwconv2.weight",
            "pwconv2.bias",
            "gamma",
        ] {
            let key = format!("vocoder.backbone.convnext.{block}.{tail}");
            copy(archive, stored, &key, into, &key, None)?;
        }
        block += 1;
    }
    if block == 0 {
        return Err(failed("no ConvNeXt blocks in the archive"));
    }
    // The analysis filterbanks are stored with a singleton input axis; only
    // their taps matter.
    for bank in ["pqmf_2", "pqmf_6"] {
        let key = format!("vocoder.{bank}.H");
        let Some(shape) = stored
            .get(&key)
            .map(|tensor| vec![tensor.shape[0], tensor.shape[2]])
        else {
            continue;
        };
        copy(
            archive,
            stored,
            &key,
            into,
            &format!("vocoder.{bank}.filters"),
            Some(shape),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
