//! Marking the stress before speaking.
//!
//! Russian does not write stress, and both Russian engines read it as a real
//! input: without a mark the model guesses, and it guesses wrong often enough
//! to hear. The marking itself belongs to `text stress` — a second
//! implementation would only drift from it — so what lives here is the wiring:
//! the same runtime and device the synthesis runs on, and a model loaded
//! **lazily**, so `--stress off` never downloads it.
//!
//! Marks the caller wrote themselves are never moved: that is guaranteed by the
//! operation, not by this module.

use std::path::Path;

use trakktor_core::stress::{
    self, Dictionary, StressModel, StressOptions, StressRuntime, Stressor,
};

use crate::{
    cli::{DeviceArg, RuntimeArg, StressArg},
    error::CliError,
};

/// Where a marker should run: the same place the synthesis does.
#[derive(Clone, Copy)]
pub(crate) struct Where {
    pub mode: StressArg,
    pub runtime: RuntimeArg,
    pub device: DeviceArg,
}

/// The text of one synthesis run, marked.
pub(crate) struct Marked {
    /// The paragraphs to speak.
    pub paragraphs: Vec<String>,
    /// The reference transcript, marked the same way — otherwise the model
    /// aligns an unmarked recording against a marked text.
    pub ref_text: String,
}

/// Marks the paragraphs and the reference transcript, loading the model on
/// first use.
///
/// Returns `None` when `--stress off`, so the caller can tell "not asked for"
/// from "asked for and produced this".
pub(crate) fn mark(
    at: Where,
    model_dir: &Path,
    paragraphs: &[String],
    ref_text: &str,
) -> Result<Option<Marked>, CliError> {
    let Some(stressor) = marker(at, model_dir)? else {
        return Ok(None);
    };
    // No user dictionary here: this is the engine's convenience path, and a
    // caller who needs one runs `text stress` and feeds the result back.
    let dictionary = Dictionary::default();
    // The engine reads `+`, so the mark form is not a choice at this end.
    let options = StressOptions::default();

    let mut marked = Vec::with_capacity(paragraphs.len());
    for paragraph in paragraphs {
        marked.push(stressor.mark(paragraph, &dictionary, &options)?.text);
    }
    Ok(Some(Marked {
        paragraphs: marked,
        ref_text: stressor.mark(ref_text, &dictionary, &options)?.text,
    }))
}

/// Loads the marker, downloading and converting the model on first use.
///
/// Returns `None` when marking was not asked for, so a caller can tell "not
/// asked for" from "asked for and produced this".
pub(crate) fn marker(
    at: Where,
    model_dir: &Path,
) -> Result<Option<Stressor>, CliError> {
    if !matches!(at.mode, StressArg::Auto) {
        return Ok(None);
    }
    eprintln!("marking the stress; loading the model...");
    let resolved = stress::resolve_model(
        model_dir,
        stress::DEFAULT_MODEL,
        &mut crate::asr::progress::download_progress(),
    )?;
    // Full precision regardless of the engine's own `--precision`: both
    // networks here are small enough that it costs nothing, and every decision
    // they make is a threshold that half precision can flip.
    let precision = stress::Precision::F32;
    let runtime: Box<dyn StressModel> = match at.runtime {
        RuntimeArg::Candle => match at.device {
            DeviceArg::Cpu => {
                Box::new(StressRuntime::load_cpu(&resolved.dir, precision)?)
            },
            DeviceArg::Metal => Box::new(load_metal(&resolved.dir, precision)?),
        },
        RuntimeArg::Burn => load_burn(&resolved.dir, at.device, precision)?,
    };
    Ok(Some(Stressor::load(&resolved.dir, runtime)?))
}

/// Marks a list of paragraphs with an already-loaded marker.
///
/// # Errors
///
/// Returns [`CliError`] on a backend failure.
pub(crate) fn mark_paragraphs(
    stressor: &Stressor,
    paragraphs: &[String],
) -> Result<Vec<String>, CliError> {
    // No user dictionary here: this is the engine's convenience path, and a
    // caller who needs one runs `text stress` and feeds the result back.
    let dictionary = Dictionary::default();
    let options = StressOptions::default();
    paragraphs
        .iter()
        .map(|paragraph| {
            Ok(stressor.mark(paragraph, &dictionary, &options)?.text)
        })
        .collect()
}

/// Loads the model on Metal (builds with the `metal` feature).
#[cfg(feature = "metal")]
fn load_metal(
    model_dir: &Path,
    precision: stress::Precision,
) -> Result<StressRuntime, CliError> {
    Ok(StressRuntime::load_metal(model_dir, precision)?)
}

/// Without the `metal` feature, `--device metal` is a validation error.
#[cfg(not(feature = "metal"))]
fn load_metal(
    _model_dir: &Path,
    _precision: stress::Precision,
) -> Result<StressRuntime, CliError> {
    Err(stress::StressError::InvalidOptions(
        "this build has no Metal support; install or build trakktor with the \
         `metal` feature"
            .into(),
    )
    .into())
}

/// Loads the model on the burn runtime (builds with the `burn` feature).
#[cfg(feature = "burn")]
fn load_burn(
    model_dir: &Path,
    device: DeviceArg,
    precision: stress::Precision,
) -> Result<Box<dyn StressModel>, CliError> {
    use trakktor_core::stress::StressBurnRuntime;
    let runtime = match device {
        DeviceArg::Cpu => StressBurnRuntime::load_cpu(model_dir, precision)?,
        DeviceArg::Metal => {
            crate::burn_notice::announce_cold_gpu_start();
            StressBurnRuntime::load_metal(model_dir, precision)?
        },
    };
    Ok(Box::new(runtime))
}

/// Without the `burn` feature, `--runtime burn` is a validation error.
#[cfg(not(feature = "burn"))]
fn load_burn(
    _model_dir: &Path,
    _device: DeviceArg,
    _precision: stress::Precision,
) -> Result<Box<dyn StressModel>, CliError> {
    Err(CliError::from(stress::StressError::InvalidOptions(
        "this build has no burn runtime; install or build trakktor with the \
         `burn` feature"
            .into(),
    )))
}
